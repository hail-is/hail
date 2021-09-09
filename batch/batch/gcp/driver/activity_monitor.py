from typing import TYPE_CHECKING
import re
import json
import logging

from hailtop import aiogoogle
from hailtop.utils import parse_timestamp_msecs, RateLimit

from ...batch_configuration import PROJECT
from ...driver.activity_monitor import BaseActivityMonitor

if TYPE_CHECKING:
    from .compute_manager import ComputeManager  # pylint: disable=cyclic-import

log = logging.getLogger('activity_monitor')


RESOURCE_NAME_REGEX = re.compile('projects/(?P<project>[^/]+)/zones/(?P<zone>[^/]+)/instances/(?P<name>.+)')


def parse_resource_name(resource_name):
    match = RESOURCE_NAME_REGEX.fullmatch(resource_name)
    assert match
    return match.groupdict()


class ActivityMonitor(BaseActivityMonitor):
    def __init__(self, compute_manager: 'ComputeManager'):
        super().__init__(compute_manager)

        self.activity_logs_client = aiogoogle.LoggingClient(
            credentials=compute_manager.credentials,
            # The project-wide logging quota is 60 request/m.  The event
            # loop sleeps 15s per iteration, so the max rate is 4
            # iterations/m.  Note, the event loop could make multiple
            # logging requests per iteration, so these numbers are not
            # quite comparable.  I didn't want to consume the entire quota
            # since there will be other users of the logging API (us at
            # the web console, test deployments, etc.)
            rate_limit=RateLimit(10, 60),
        )

    async def handle_event(self, event):
        payload = event.get('protoPayload')
        if payload is None:
            log.warning(f'event has no payload {json.dumps(event)}')
            return

        timestamp_msecs = self.event_timestamp_msecs(event)

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

        if not name.startswith(self.compute_manager.machine_name_prefix):
            log.warning(f'event for unknown machine {name}')
            return

        if event_subtype == 'v1.compute.instances.insert':
            if event_type == 'COMPLETED':
                severity = event['severity']
                operation_id = event['operation']['id']
                success = severity != 'ERROR'
                self.zone_monitor.zone_success_rate.push(resource['labels']['zone'], operation_id, success)
        else:
            instance = self.app['inst_coll_manager'].get_instance(name)
            if not instance:
                record = await self.db.select_and_fetchone('SELECT name FROM instances WHERE name = %s;', (name,))
                if not record:
                    log.error(f'event for unknown instance {name}: {json.dumps(event)}')
                return

            if event_subtype == 'compute.instances.preempted':
                log.info(f'event handler: handle preempt {instance}')
                await self.handle_preempt_event(instance, timestamp_msecs)
            elif event_subtype == 'v1.compute.instances.delete':
                if event_type == 'COMPLETED':
                    log.info(f'event handler: delete {instance} done')
                    await self.handle_delete_done_event(instance, timestamp_msecs)
                elif event_type == 'STARTED':
                    log.info(f'event handler: handle call delete {instance}')
                    await self.handle_call_delete_event(instance, timestamp_msecs)

    async def list_events(self, mark):
        filter = f'''
(logName="projects/{PROJECT}/logs/cloudaudit.googleapis.com%2Factivity" OR
logName="projects/{PROJECT}/logs/cloudaudit.googleapis.com%2Fsystem_event"
) AND
resource.type=gce_instance AND
protoPayload.resourceName:"{self.compute_manager.machine_name_prefix}" AND
timestamp >= "{mark}"
'''

        return await self.activity_logs_client.list_entries(
            body={
                'resourceNames': [f'projects/{PROJECT}'],
                'orderBy': 'timestamp asc',
                'pageSize': 100,
                'filter': filter,
            }
        )

    def event_timestamp_msecs(self, event):
        return parse_timestamp_msecs(event['timestamp'])

    async def shutdown(self):
        try:
            await super().shutdown()
        finally:
            await self.activity_logs_client.close()
