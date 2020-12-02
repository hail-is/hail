import re
import asyncio
import aiohttp
import logging
import datetime
import sortedcontainers
import dateutil.parser

from hailtop import aiotools
from hailtop.utils import time_msecs, secret_alnum_string

from .instance import Instance
from ..batch_configuration import PROJECT

log = logging.getLogger('instance_monitor')


def parse_resource_name(resource_name):
    match = RESOURCE_NAME_REGEX.fullmatch(resource_name)
    assert match
    return match.groupdict()


RESOURCE_NAME_REGEX = re.compile('projects/(?P<project>[^/]+)/zones/(?P<zone>[^/]+)/instances/(?P<name>.+)')


class InstanceMonitor:
    def __init__(self, app, machine_name_prefix):
        self.app = app
        self.db = app['db']
        self.compute_client = app['compute_client']
        self.logging_client = app['logging_client']
        self.log_store = app['log_store']
        self.zone_monitor = app['zone_monitor']

        self.machine_name_prefix = machine_name_prefix

        self.instances_by_last_updated = sortedcontainers.SortedSet(
            key=lambda instance: instance.last_updated)

        self.name_instance = {}

        self.n_instances_by_state = {
            'pending': 0,
            'active': 0,
            'inactive': 0,
            'deleted': 0
        }

        # pending and active
        self.live_free_cores_mcpu = 0
        self.live_total_cores_mcpu = 0

        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        log.info('initializing instance monitor')

        async for record in self.db.select_and_fetchall(
                'SELECT * FROM instances WHERE removed = 0;'):
            instance = Instance.from_record(self.app, record)
            self.add_instance(instance)

        log.info('finished initializing instance monitor')

    async def run(self):
        self.task_manager.ensure_future(self.event_loop())
        self.task_manager.ensure_future(self.instance_monitoring_loop())

    def shutdown(self):
        self.task_manager.shutdown()

    def unique_machine_name(self):
        while True:
            # 36 ** 5 = ~60M
            suffix = secret_alnum_string(5, case='lower')
            machine_name = f'{self.machine_name_prefix}{suffix}'
            if machine_name not in self.name_instance:
                break
        return machine_name

    def adjust_for_remove_instance(self, instance):
        assert instance in self.instances_by_last_updated

        self.instances_by_last_updated.remove(instance)
        self.n_instances_by_state[instance.state] -= 1

        if instance.state in ('pending', 'active'):
            self.live_free_cores_mcpu -= max(0, instance.free_cores_mcpu)
            self.live_total_cores_mcpu -= instance.cores_mcpu

        instance.inst_pool.adjust_for_remove_instance(instance)

    async def remove_instance(self, instance, reason, timestamp=None):
        await instance.deactivate(reason, timestamp)

        await self.db.just_execute(
            'UPDATE instances SET removed = 1 WHERE name = %s;', (instance.name,))

        self.adjust_for_remove_instance(instance)
        del self.name_instance[instance.name]

    def adjust_for_add_instance(self, instance):
        assert instance not in self.instances_by_last_updated

        self.instances_by_last_updated.add(instance)
        self.n_instances_by_state[instance.state] += 1

        if instance.state in ('pending', 'active'):
            self.live_free_cores_mcpu += max(0, instance.free_cores_mcpu)
            self.live_total_cores_mcpu += instance.cores_mcpu

        instance.inst_pool.adjust_for_add_instance(instance)

    def add_instance(self, instance):
        assert instance.name not in self.name_instance
        self.name_instance[instance.name] = instance
        self.adjust_for_add_instance(instance)

    async def call_delete_instance(self, instance, reason, timestamp=None, force=False):
        if instance.state == 'deleted' and not force:
            return
        if instance.state not in ('inactive', 'deleted'):
            await instance.deactivate(reason, timestamp)

        try:
            await self.compute_client.delete(
                f'/zones/{instance.zone}/instances/{instance.name}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                log.info(f'{instance} already delete done')
                await self.remove_instance(instance, reason, timestamp)
                return
            raise

    async def handle_preempt_event(self, instance, timestamp):
        await self.call_delete_instance(instance, 'preempted', timestamp=timestamp)

    async def handle_delete_done_event(self, instance, timestamp):
        await self.remove_instance(instance, 'deleted', timestamp)

    async def handle_call_delete_event(self, instance, timestamp):
        await instance.mark_deleted('deleted', timestamp)

    async def handle_event(self, event):
        payload = event.get('protoPayload')
        if payload is None:
            log.warning('event has no payload')
            return

        timestamp = dateutil.parser.isoparse(event['timestamp']).timestamp() * 1000

        resource_type = event['resource']['type']
        if resource_type != 'gce_instance':
            log.warning(f'unknown event resource type {resource_type}')
            return

        operation_started = event['operation'].get('first', False)
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
                success = (severity != 'ERROR')
                self.zone_monitor.zone_success_rate.push(resource['labels']['zone'], operation_id, success)
        else:
            instance = self.name_instance.get(name)
            if not instance:
                log.warning(f'event for unknown instance {name}')
                return

            if event_subtype == 'v1.compute.instances.preempted':
                log.info(f'event handler: handle preempt {instance}')
                await self.handle_preempt_event(instance, timestamp)
            elif event_subtype == 'v1.compute.instances.delete':
                if event_type == 'COMPLETED':
                    log.info(f'event handler: delete {instance} done')
                    await self.handle_delete_done_event(instance, timestamp)
                elif event_type == 'STARTED':
                    log.info(f'event handler: handle call delete {instance}')
                    await self.handle_call_delete_event(instance, timestamp)

    async def event_loop(self):
        log.info('starting event loop')
        while True:
            try:
                row = await self.db.select_and_fetchone('SELECT * FROM `gevents_mark`;')
                mark = row['mark']
                if mark is None:
                    mark = datetime.datetime.utcnow().isoformat() + 'Z'
                    await self.db.execute_update(
                        'UPDATE `gevents_mark` SET mark = %s;',
                        (mark,))

                filter = f'''
logName="projects/{PROJECT}/logs/cloudaudit.googleapis.com%2Factivity" AND
resource.type=gce_instance AND
protoPayload.resourceName:"{self.machine_name_prefix}" AND
timestamp >= "{mark}"
'''

                new_mark = None
                async for event in await self.logging_client.list_entries(
                        body={
                            'resourceNames': [f'projects/{PROJECT}'],
                            'orderBy': 'timestamp asc',
                            'pageSize': 100,
                            'filter': filter
                        }):
                    # take the last, largest timestamp
                    new_mark = event['timestamp']
                    await self.handle_event(event)

                if new_mark is not None:
                    await self.db.execute_update(
                        'UPDATE `gevents_mark` SET mark = %s;',
                        (new_mark,))
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:
                log.exception('in event loop')
            await asyncio.sleep(15)

    async def check_on_instance(self, instance):
        active_and_healthy = await instance.check_is_active_and_healthy()
        if active_and_healthy:
            return

        try:
            spec = await self.compute_client.get(
                f'/zones/{instance.zone}/instances/{instance.name}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                await self.remove_instance(instance, 'does_not_exist')
                return
            raise

        # PROVISIONING, STAGING, RUNNING, STOPPING, TERMINATED
        gce_state = spec['status']
        log.info(f'{instance} gce_state {gce_state}')

        if gce_state in ('STOPPING', 'TERMINATED'):
            log.info(f'{instance} live but stopping or terminated, deactivating')
            await instance.deactivate('terminated')

        if (gce_state in ('STAGING', 'RUNNING')
                and instance.state == 'pending'
                and time_msecs() - instance.time_created > 5 * 60 * 1000):
            # FIXME shouldn't count time in PROVISIONING
            log.info(f'{instance} did not activate within 5m, deleting')
            await self.call_delete_instance(instance, 'activation_timeout')

        if instance.state == 'inactive':
            log.info(f'{instance} is inactive, deleting')
            await self.call_delete_instance(instance, 'inactive')

        await instance.update_timestamp()

    async def instance_monitoring_loop(self):
        log.info('starting instance monitoring loop')

        while True:
            try:
                if self.instances_by_last_updated:
                    # 0 is the smallest (oldest)
                    instance = self.instances_by_last_updated[0]
                    since_last_updated = time_msecs() - instance.last_updated
                    if since_last_updated > 60 * 1000:
                        log.info(f'checking on {instance}, last updated {since_last_updated / 1000}s ago')
                        await self.check_on_instance(instance)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:
                log.exception('in monitor instances loop')

            await asyncio.sleep(1)
