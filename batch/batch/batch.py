import json
import logging

from gear import Database, transaction
from hailtop.utils import humanize_timedelta_msecs, time_msecs_str

from .batch_format_version import BatchFormatVersion
from .exceptions import NonExistentJobGroupError, OpenBatchError
from .utils import coalesce

log = logging.getLogger('batch')


def _time_msecs_str(t):
    if t:
        return time_msecs_str(t)
    return None


def job_group_record_to_dict(record):
    if record['state'] == 'open':
        state = 'open'
    elif record['n_failed'] > 0:
        state = 'failure'
    elif record['cancelled']:
        state = 'cancelled'
    elif record['state'] == 'complete':
        if record['n_cancelled'] > 0:
            state = 'cancelled'
        else:
            assert record['n_succeeded'] == record['n_jobs']
            state = 'success'
    else:
        state = 'running'

    time_created = _time_msecs_str(record['time_created'])
    time_closed = None
    time_completed = _time_msecs_str(record['time_completed'])

    if record['time_created'] and record['time_completed']:
        duration = humanize_timedelta_msecs(record['time_completed'] - record['time_created'])
    else:
        duration = None

    d = {
        'id': record['batch_id'],
        'batch_id': record['batch_id'],
        'job_group_id': record['job_group_id'],
        'user': record['user'],
        'billing_project': record['billing_project'],
        'token': record['token'],
        'state': state,
        'complete': record['state'] == 'complete',
        'closed': record['state'] != 'open',
        'n_jobs': record['n_jobs'],
        'n_completed': record['n_completed'],
        'n_succeeded': record['n_succeeded'],
        'n_failed': record['n_failed'],
        'n_cancelled': record['n_cancelled'],
        'time_created': time_created,
        'time_closed': time_closed,
        'time_completed': time_completed,
        'duration': duration,
        'msec_mcpu': record['msec_mcpu'],
        'cost': coalesce(record['cost'], 0),
        'path': record['path'],
    }

    try:
        attributes = json.loads(record['attributes'])
    except KeyError:
        raise
    if attributes:
        d['attributes'] = attributes

    return d


def job_record_to_dict(record, name):
    format_version = BatchFormatVersion(record['format_version'])

    db_status = record['status']
    if db_status:
        db_status = json.loads(db_status)
        exit_code, duration = format_version.get_status_exit_code_duration(db_status)
    else:
        exit_code = None
        duration = None

    result = {
        'batch_id': record['batch_id'],
        'job_id': record['job_id'],
        'job_group_id': record['job_group_id'],
        'job_group': record['path'],
        'name': name,
        'user': record['user'],
        'billing_project': record['billing_project'],
        'state': record['state'],
        'exit_code': exit_code,
        'duration': duration,
        'cost': coalesce(record['cost'], 0),
        'msec_mcpu': record['msec_mcpu'],
    }

    return result


async def cancel_job_group_in_db(db: Database, batch_id: int, job_group_id: int):
    @transaction(db)
    async def cancel(tx):
        record = await tx.execute_and_fetchone(
            '''
SELECT job_groups.`state`
FROM job_groups
LEFT JOIN batches ON batches.id = job_groups.batch_id
WHERE batch_id = %s AND job_group_id = %s AND NOT deleted
FOR UPDATE;
''',
            (batch_id, job_group_id),
        )
        if not record:
            raise NonExistentJobGroupError(batch_id, job_group_id)

        if record['state'] == 'open':
            raise OpenBatchError(batch_id)

        await tx.just_execute('CALL cancel_job_group(%s, %s);', (batch_id, job_group_id))

    await cancel()
