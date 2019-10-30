import json
import logging
import aiohttp
from gear import execute_and_fetchone, execute_and_fetchall

from .globals import complete_states, tasks
from .database import check_call_procedure

log = logging.getLogger('batch')


# FIXME remove batch.attributes, query attributes
def batch_record_to_dict(record):
    # FIXME open job should have state = open
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
    return d


async def notify_batch_job_complete(db, batch_id):
    # FIXME fix batch_to_dict and just query callback here
    record = await execute_and_fetchone(
        db.pool, '''
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
            await session.post(callback, json=batch_record_to_dict(record))
            log.info(f'callback for batch {batch_id} successful')
    except Exception:
        log.exception(f'callback for batch {batch_id} failed, will not retry.')


# FIXME move
# FIXME if already in complete state, just ignore/do nothing
async def mark_job_complete(
        db, scheduler_state_changed, inst_pool,
        batch_id, job_id, new_state, status):
    id = (batch_id, job_id)

    rv = await check_call_procedure(
        db.pool,
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
        if 'error' in record['status']:
            result['error'] = record['status']['error']
        result['exit_code'] = {
            k: getopt(getopt(record['status']['container_statuses'][k], 'container_status'), 'exit_code') for
            k in tasks
        }
        result['duration'] = {k: getopt(record['status']['container_statuses'][k]['timing'], 'runtime') for k in tasks}
        result['message'] = {
            # the execution of the job container might have
            # failed, or the docker container might have completed
            # (wait returned) but had status error
            k: (getopt(record['status']['container_statuses'][k], 'error') or
                getopt(getopt(record['status']['container_statuses'][k], 'container_status'), 'error'))
            for k in tasks
        }
    if record['attributes']:
        result['attributes'] = record['attributes']
    return result


class Batch:
    @staticmethod
    def from_record(db, record, deleted=False):
        if record is not None:
            if not deleted:
                assert not record['deleted']
            attributes = json.loads(record['attributes'])
            userdata = json.loads(record['userdata'])

            if record['n_failed'] > 0:
                state = 'failure'
            elif record['n_cancelled'] > 0:
                state = 'cancelled'
            elif record['closed'] and record['n_succeeded'] == record['n_jobs']:
                state = 'success'
            else:
                state = 'running'

            complete = record['closed'] and record['n_completed'] == record['n_jobs']

            return Batch(db,
                         id=record['id'],
                         attributes=attributes,
                         callback=record['callback'],
                         userdata=userdata,
                         user=record['user'],
                         state=state,
                         complete=complete,
                         deleted=record['deleted'],
                         cancelled=record['cancelled'],
                         closed=record['closed'])
        return None

    @staticmethod
    async def from_db(db, ids, user):
        batches = await Batch.from_db_multiple(db, ids, user)
        if len(batches) == 1:
            return batches[0]
        return None

    @staticmethod
    async def from_db_multiple(db, ids, user):
        records = await db.batch.get_undeleted_records(ids, user)
        batches = [Batch.from_record(db, record) for record in records]
        return batches

    @staticmethod
    async def create_batch(db, attributes, callback, userdata, n_jobs):
        user = userdata['username']

        id = await db.batch.new_record(
            attributes=json.dumps(attributes),
            callback=callback,
            userdata=json.dumps(userdata),
            user=user,
            deleted=False,
            cancelled=False,
            closed=False,
            n_jobs=n_jobs)

        batch = Batch(db, id=id, attributes=attributes, callback=callback,
                      userdata=userdata, user=user, state='running',
                      complete=False, deleted=False, cancelled=False,
                      closed=False)

        if attributes is not None:
            items = [{'batch_id': id, 'key': k, 'value': v} for k, v in attributes.items()]
            await db.batch_attributes.new_records(items)

        return batch

    def __init__(self, db, id, attributes, callback, userdata, user,
                 state, complete, deleted, cancelled, closed):
        self.db = db
        self.id = id
        self.attributes = attributes
        self.callback = callback
        self.userdata = userdata
        self.user = user
        self.state = state
        self.complete = complete
        self.deleted = deleted
        self.cancelled = cancelled
        self.closed = closed

    # FIXME move these to front-end, directly use database
    # called by front end
    async def cancel(self):
        await self.db.batch.update_record(self.id, cancelled=True)
        self.cancelled = True
        log.info(f'{self} cancelled')

    # called by front end
    async def close(self):
        await check_call_procedure(
            self.db.pool, 'CALL close_batch(%s);', (self.id,))
        self.closed = True
        log.info(f'{self} closed')

    # called by front end
    async def mark_deleted(self):
        await self.cancel()
        await self.db.batch.update_record(self.id, cancelled=True, deleted=True)
        self.cancelled = True
        self.deleted = True
        log.info(f'{self} marked for deletion')

    def is_complete(self):
        return self.complete

    def is_successful(self):
        return self.state == 'success'

    async def to_dict(self, include_jobs=False):
        result = {
            'id': self.id,
            'state': self.state,
            'complete': self.complete,
            'closed': self.closed
        }
        if self.attributes:
            result['attributes'] = self.attributes
        if include_jobs:
            jobs = [
                job_record_to_dict(record)
                async for record
                in execute_and_fetchall(
                    'SELECT * FROM jobs where batch_id = %s',
                    (self.id,))
            ]
            result['jobs'] = sorted(jobs, key=lambda j: j['job_id'])
        return result

    def __str__(self):
        return f'batch {self.id}'
