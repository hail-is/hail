import json
import logging
import re
from typing import Any, Dict

from gear import Database
from hailtop.aiocloud import aiogoogle
from hailtop.utils import parse_timestamp_msecs

from ....driver.instance_collection import InstanceCollectionManager
from .zones import ZoneSuccessRate

log = logging.getLogger('activity_logs')


RESOURCE_NAME_REGEX = re.compile('projects/(?P<project>[^/]+)/zones/(?P<zone>[^/]+)/instances/(?P<name>.+)')


def parse_resource_name(resource_name: str) -> Dict[str, str]:
    match = RESOURCE_NAME_REGEX.fullmatch(resource_name)
    assert match
    return match.groupdict()


async def handle_activity_log_event(
    event: Dict[str, Any],
    db: Database,
    inst_coll_manager: InstanceCollectionManager,
    zone_success_rate: ZoneSuccessRate,
    machine_name_prefix: str,
):
    payload = event.get('protoPayload')
    if payload is None:
        log.warning(f'event has no payload {json.dumps(event)}')
        return

    timestamp_msecs = parse_timestamp_msecs(event['timestamp'])

    resource_type = event['resource']['type']
    if resource_type != 'gce_instance':
        log.warning(f'unknown event resource type {resource_type} {json.dumps(event)}')
        return

    operation = event.get('operation')
    if operation is None:
        # occurs when deleting a worker that does not exist
        log.info(f'received an event with no operation {json.dumps(event)}')
        return

    operation_started = operation.get('first', False)
    if operation_started:
        event_type = 'STARTED'
    else:
        event_type = 'COMPLETED'

    event_subtype = payload['methodName']
    resource = event['resource']
    name = parse_resource_name(payload['resourceName'])['name']

    log.info(f'event {resource_type} {event_type} {event_subtype} {name}')

    if not name.startswith(machine_name_prefix):
        log.warning(f'event for unknown machine {name}')
        return

    if event_subtype == 'v1.compute.instances.insert':
        if event_type == 'COMPLETED':
            severity = event['severity']
            operation_id = event['operation']['id']
            success = severity != 'ERROR'
            zone_success_rate.push(resource['labels']['zone'], operation_id, success)
    else:
        instance = inst_coll_manager.get_instance(name)
        if not instance:
            record = await db.select_and_fetchone('SELECT name FROM instances WHERE name = %s;', (name,))
            if not record:
                log.error(f'event for unknown instance {name}: {json.dumps(event)}')
            return

        if event_subtype == 'compute.instances.preempted':
            await instance.inst_coll.call_delete_instance(instance, 'preempted', timestamp=timestamp_msecs)
        elif event_subtype == 'v1.compute.instances.delete':
            if event_type == 'COMPLETED':
                await instance.inst_coll.remove_instance(instance, 'deleted', timestamp_msecs)
            elif event_type == 'STARTED':
                await instance.mark_deleted('deleted', timestamp_msecs)


async def process_activity_log_events_since(
    db: Database,
    inst_coll_manager: InstanceCollectionManager,
    activity_logs_client: aiogoogle.GoogleLoggingClient,
    zone_success_rate: ZoneSuccessRate,
    machine_name_prefix: str,
    project: str,
    mark: str,
) -> str:
    filter = f"""
(logName="projects/{project}/logs/cloudaudit.googleapis.com%2Factivity" OR
logName="projects/{project}/logs/cloudaudit.googleapis.com%2Fsystem_event"
) AND
resource.type=gce_instance AND
protoPayload.resourceName:"{machine_name_prefix}" AND
timestamp >= "{mark}"
"""

    body = {
        'resourceNames': [f'projects/{project}'],
        'orderBy': 'timestamp asc',
        'pageSize': 100,
        'filter': filter,
    }

    async for event in await activity_logs_client.list_entries(body=body):
        mark = event['timestamp']
        await handle_activity_log_event(event, db, inst_coll_manager, zone_success_rate, machine_name_prefix)
    return mark
