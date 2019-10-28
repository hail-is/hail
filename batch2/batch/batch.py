import json
import logging
import aiohttp

from .globals import complete_states, tasks
from .database import check_call_procedure

log = logging.getLogger('batch')


async def batch_record_to_dict(db, record, include_jobs=False):
    if record['n_failed'] > 0:
        state = 'failure'
    elif record['n_cancelled'] > 0:
        state = 'cancelled'
    elif record['closed'] and record['n_succeeded'] == record['n_jobs']:
        state = 'success'
    else:
        state = 'running'

    complete = record['closed'] and record['n_completed'] == record['n_jobs']

    d = {
        'id': record['id'],
        'state': state,
        'complete': complete,
        'closed': record['closed']
    }

    attributes = json.loads(record['attributes'])
    if attributes:
        d['attributes'] = attributes

    if include_jobs:
        jobs = [
            job_record_to_dict(record)
            async for record
            in db.execute_and_fetchall(
                'SELECT * FROM jobs where batch_id = %s',
                (record['id'],))
        ]
        d['jobs'] = sorted(jobs, key=lambda j: j['job_id'])

    return d


async def notify_batch_job_complete(db, batch_id):
    record = await db.execute_and_fetchone(
        '''
SELECT *
FROM batch
WHERE id = %s AND closed AND n_completed = n_jobs;
''',
        (batch_id,))

    if not record:
        return
    callback = record['callback']
    if not callback:
        return

    log.info(f'making callback for batch {batch_id}: {callback}')

    try:
        async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            await session.post(callback, json=await batch_record_to_dict(db, record))
            log.info(f'callback for batch {batch_id} successful')
    except Exception:
        log.exception(f'callback for batch {batch_id} failed, will not retry.')


async def mark_job_complete(
        db, scheduler_state_changed, inst_pool,
        batch_id, job_id, new_state, status):
    id = (batch_id, job_id)

    rv = await check_call_procedure(
        db,
        'CALL mark_job_complete(%s, %s, %s, %s);',
        (batch_id, job_id, new_state,
         json.dumps(status) if status is not None else None))

    log.info(f'mark_job_complete returned {rv} for job {id}')

    old_state = rv['old_state']
    if old_state in complete_states:
        # already complete, do nothing
        return

    log.info(f'job {id} changed state: {rv["old_state"]} => {new_state}')

    instance_id = rv['instance_id']
    if instance_id:
        instance = inst_pool.id_instance.get(instance_id)
        log.info(f'updating instance: {instance}')
        if instance:
            inst_pool.adjust_for_remove_instance(instance)
            instance.free_cores_mcpu += rv['cores_mcpu']
            inst_pool.adjust_for_add_instance(instance)

    scheduler_state_changed.set()

    await notify_batch_job_complete(db, batch_id)


def job_record_to_dict(record):
    def getopt(obj, attr):
        if obj is None:
            return None
        return obj.get(attr)

    result = {
        'batch_id': record['batch_id'],
        'job_id': record['job_id'],
        'state': record['state']
    }
    # FIXME can't change this yet, batch and batch2 share client
    if record['status']:
        status = json.loads(record['status'])

        if 'error' in status:
            result['error'] = status['error']
        result['exit_code'] = {
            k: getopt(getopt(status['container_statuses'][k], 'container_status'), 'exit_code') for
            k in tasks
        }
        result['duration'] = {k: getopt(status['container_statuses'][k]['timing'], 'runtime') for k in tasks}
        result['message'] = {
            # the execution of the job container might have
            # failed, or the docker container might have completed
            # (wait returned) but had status error
            k: (getopt(status['container_statuses'][k], 'error') or
                getopt(getopt(status['container_statuses'][k], 'container_status'), 'error'))
            for k in tasks
        }

    spec = json.loads(record['spec'])
    attributes = spec.get('attributes')
    if attributes:
        result['attributes'] = attributes
    return result
