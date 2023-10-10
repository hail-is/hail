import json
import logging
from typing import Any, Dict, List, Optional

from gear import transaction
from hailtop.batch_client.types import CostBreakdownEntry, JobListEntryV1Alpha
from hailtop.utils import humanize_timedelta_msecs, time_msecs_str

from .batch_format_version import BatchFormatVersion
from .exceptions import NonExistentBatchError, OpenBatchError
from .utils import coalesce

log = logging.getLogger('batch')


def cost_breakdown_to_dict(cost_breakdown: Dict[str, float]) -> List[CostBreakdownEntry]:
    return [{'resource': resource, 'cost': cost} for resource, cost in cost_breakdown.items()]


def batch_record_to_dict(record: Dict[str, Any]) -> Dict[str, Any]:
    if record['state'] == 'open':
        state = 'open'
    elif record['n_failed'] > 0:
        state = 'failure'
    elif record['cancelled'] or record['n_cancelled'] > 0:
        state = 'cancelled'
    elif record['state'] == 'complete':
        assert record['n_succeeded'] == record['n_jobs']
        state = 'success'
    else:
        state = 'running'

    def _time_msecs_str(t):
        if t:
            return time_msecs_str(t)
        return None

    time_created = _time_msecs_str(record['time_created'])
    time_closed = _time_msecs_str(record['time_closed'])
    time_completed = _time_msecs_str(record['time_completed'])

    if record['time_created'] and record['time_completed']:
        duration_ms = record['time_completed'] - record['time_created']
        duration = humanize_timedelta_msecs(duration_ms)
    else:
        duration_ms = None
        duration = None

    if record['cost_breakdown'] is not None:
        record['cost_breakdown'] = cost_breakdown_to_dict(json.loads(record['cost_breakdown']))

    d = {
        'id': record['id'],
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
        'duration_ms': duration_ms,
        'duration': duration,
        'msec_mcpu': record['msec_mcpu'],
        'cost': coalesce(record.get('cost'), 0),
        'cost_breakdown': record['cost_breakdown'],
    }

    attributes = json.loads(record['attributes'])
    if attributes:
        d['attributes'] = attributes

    return d


def job_record_to_dict(record: Dict[str, Any], name: Optional[str]) -> JobListEntryV1Alpha:
    format_version = BatchFormatVersion(record['format_version'])

    db_status = record['status']
    if db_status:
        db_status = json.loads(db_status)
        exit_code, duration = format_version.get_status_exit_code_duration(db_status)
    else:
        exit_code = None
        duration = None

    if record['cost_breakdown'] is not None:
        record['cost_breakdown'] = cost_breakdown_to_dict(json.loads(record['cost_breakdown']))

    return {
        'batch_id': record['batch_id'],
        'job_id': record['job_id'],
        'name': name,
        'user': record['user'],
        'billing_project': record['billing_project'],
        'state': record['state'],
        'exit_code': exit_code,
        'duration': duration,
        'cost': coalesce(record.get('cost'), 0),
        'msec_mcpu': record['msec_mcpu'],
        'cost_breakdown': record['cost_breakdown'],
    }


async def cancel_batch_in_db(db, batch_id):
    @transaction(db)
    async def cancel(tx):
        record = await tx.execute_and_fetchone(
            '''
SELECT `state` FROM batches
WHERE id = %s AND NOT deleted
FOR UPDATE;
''',
            (batch_id,),
        )
        if not record:
            raise NonExistentBatchError(batch_id)

        if record['state'] == 'open':
            raise OpenBatchError(batch_id)

        await tx.just_execute('CALL cancel_batch(%s);', (batch_id,))

    await cancel()
