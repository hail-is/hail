import re
import json
import logging
import datetime
import aiohttp

from gear import Database
from hailtop import aiotools, aiogoogle
from hailtop.utils import periodically_call, time_msecs, parse_timestamp_msecs

from ..batch_configuration import PROJECT, DEFAULT_NAMESPACE
from .zone_monitor import ZoneMonitor
from .instance_collection_manager import InstanceCollectionManager

log = logging.getLogger('gce_event_monitor')

RESOURCE_NAME_REGEX = re.compile('projects/(?P<project>[^/]+)/zones/(?P<zone>[^/]+)/instances/(?P<name>.+)')


def parse_resource_name(resource_name):
    match = RESOURCE_NAME_REGEX.fullmatch(resource_name)
    assert match
    return match.groupdict()


class GCEEventMonitor:
    def __init__(self, app, machine_name_prefix):
        self.app = app
        self.db: Database = app['db']
        self.zone_monitor: ZoneMonitor = app['zone_monitor']
        self.compute_client: aiogoogle.ComputeClient = app['compute_client']
        self.logging_client: aiogoogle.LoggingClient = app['logging_client']
        self.inst_coll_manager: InstanceCollectionManager = app['inst_coll_manager']
        self.machine_name_prefix = machine_name_prefix

        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(self.event_loop())
        self.task_manager.ensure_future(self.delete_orphaned_disks_loop())

    def shutdown(self):
        self.task_manager.shutdown()

    async def handle_preempt_event(self, instance, timestamp):
        await instance.inst_coll.call_delete_instance(instance, 'preempted', timestamp=timestamp)

    async def handle_delete_done_event(self, instance, timestamp):
        await instance.inst_coll.remove_instance(instance, 'deleted', timestamp)

    async def handle_call_delete_event(self, instance, timestamp):
        await instance.mark_deleted('deleted', timestamp)

    async def handle_event(self, event):
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

        if not name.startswith(self.machine_name_prefix):
            log.warning(f'event for unknown machine {name}')
            return

        if event_subtype == 'v1.compute.instances.insert':
            if event_type == 'COMPLETED':
                severity = event['severity']
                operation_id = event['operation']['id']
                success = severity != 'ERROR'
                self.zone_monitor.zone_success_rate.push(resource['labels']['zone'], operation_id, success)
        else:
            instance = self.inst_coll_manager.get_instance(name)
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

    async def handle_events(self):
        row = await self.db.select_and_fetchone('SELECT * FROM `gevents_mark`;')
        mark = row['mark']
        if mark is None:
            mark = datetime.datetime.utcnow().isoformat() + 'Z'
            await self.db.execute_update('UPDATE `gevents_mark` SET mark = %s;', (mark,))

        filter = f'''
(logName="projects/{PROJECT}/logs/cloudaudit.googleapis.com%2Factivity" OR
logName="projects/{PROJECT}/logs/cloudaudit.googleapis.com%2Fsystem_event"
) AND
resource.type=gce_instance AND
protoPayload.resourceName:"{self.machine_name_prefix}" AND
timestamp >= "{mark}"
'''

        log.info(f'querying logging client with mark {mark}')
        mark = None
        async for event in await self.logging_client.list_entries(
            body={
                'resourceNames': [f'projects/{PROJECT}'],
                'orderBy': 'timestamp asc',
                'pageSize': 100,
                'filter': filter,
            }
        ):
            # take the last, largest timestamp
            mark = event['timestamp']
            await self.handle_event(event)

        if mark is not None:
            await self.db.execute_update('UPDATE `gevents_mark` SET mark = %s;', (mark,))

    async def event_loop(self):
        await periodically_call(15, self.handle_events)

    async def delete_orphaned_disks(self):
        log.info('deleting orphaned disks')

        params = {'filter': f'(labels.namespace = {DEFAULT_NAMESPACE})'}

        for zone in self.zone_monitor.zones:
            async for disk in await self.compute_client.list(f'/zones/{zone}/disks', params=params):
                disk_name = disk['name']
                instance_name = disk['labels']['instance-name']
                instance = self.inst_coll_manager.get_instance(instance_name)

                creation_timestamp_msecs = parse_timestamp_msecs(disk.get('creationTimestamp'))
                last_attach_timestamp_msecs = parse_timestamp_msecs(disk.get('lastAttachTimestamp'))
                last_detach_timestamp_msecs = parse_timestamp_msecs(disk.get('lastDetachTimestamp'))

                now_msecs = time_msecs()
                if instance is None:
                    log.exception(f'deleting disk {disk_name} from instance that no longer exists')
                elif (last_attach_timestamp_msecs is None
                        and now_msecs - creation_timestamp_msecs > 60 * 60 * 1000):
                    log.exception(f'deleting disk {disk_name} that has not attached within 60 minutes')
                elif (last_detach_timestamp_msecs is not None
                        and now_msecs - last_detach_timestamp_msecs > 5 * 60 * 1000):
                    log.exception(f'deleting detached disk {disk_name} that has not been cleaned up within 5 minutes')
                else:
                    continue

                try:
                    await self.compute_client.delete_disk(f'/zones/{zone}/disks/{disk_name}')
                except aiohttp.ClientResponseError as e:
                    if e.status == 404:
                        continue
                    log.exception(f'error while deleting orphaned disk {disk_name}')

    async def delete_orphaned_disks_loop(self):
        await periodically_call(60, self.delete_orphaned_disks)
