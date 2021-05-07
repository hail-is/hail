import aiohttp
import sortedcontainers
import logging
import dateutil.parser
from typing import Dict

from hailtop.utils import time_msecs, secret_alnum_string, periodically_call
from hailtop import aiotools, aiogoogle
from gear import Database

from .instance import Instance
from .zone_monitor import ZoneMonitor

log = logging.getLogger('inst_collection')


class InstanceCollection:
    def __init__(self, app, name, machine_name_prefix, is_pool):
        self.app = app
        self.db: Database = app['db']
        self.compute_client: aiogoogle.ComputeClient = self.app['compute_client']
        self.zone_monitor: ZoneMonitor = self.app['zone_monitor']

        self.name = name
        self.machine_name_prefix = f'{machine_name_prefix}{self.name}-'
        self.is_pool = is_pool

        self.name_instance: Dict[str, Instance] = {}

        self.instances_by_last_updated = sortedcontainers.SortedSet(key=lambda instance: instance.last_updated)

        self.n_instances_by_state = {'pending': 0, 'active': 0, 'inactive': 0, 'deleted': 0}

        # pending and active
        self.live_free_cores_mcpu = 0
        self.live_total_cores_mcpu = 0

        self.boot_disk_size_gb = None
        self.max_instances = None
        self.max_live_instances = None

        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(self.monitor_instances_loop())

    def shutdown(self):
        self.task_manager.shutdown()

    @property
    def n_instances(self):
        return len(self.name_instance)

    def generate_machine_name(self):
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

    async def remove_instance(self, instance, reason, timestamp=None):
        await instance.deactivate(reason, timestamp)

        await self.db.just_execute('UPDATE instances SET removed = 1 WHERE name = %s;', (instance.name,))

        self.adjust_for_remove_instance(instance)
        del self.name_instance[instance.name]

    def adjust_for_add_instance(self, instance):
        assert instance not in self.instances_by_last_updated

        self.n_instances_by_state[instance.state] += 1

        self.instances_by_last_updated.add(instance)
        if instance.state in ('pending', 'active'):
            self.live_free_cores_mcpu += max(0, instance.free_cores_mcpu)
            self.live_total_cores_mcpu += instance.cores_mcpu

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
            await self.compute_client.delete(f'/zones/{instance.zone}/instances/{instance.name}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                log.info(f'{instance} already delete done')
                await self.remove_instance(instance, reason, timestamp)
                return
            raise

    async def check_on_instance(self, instance):
        active_and_healthy = await instance.check_is_active_and_healthy()
        if active_and_healthy:
            return

        try:
            spec = await self.compute_client.get(f'/zones/{instance.zone}/instances/{instance.name}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                await self.remove_instance(instance, 'does_not_exist')
                return
            raise

        # PROVISIONING, STAGING, RUNNING, STOPPING, TERMINATED
        gce_state = spec['status']

        log.info(f'{instance} gce_state {gce_state}')

        if (
            gce_state == 'PROVISIONING'
            and instance.state == 'pending'
            and time_msecs() - instance.time_created > 5 * 60 * 1000
        ):
            log.exception(f'{instance} did not provision within 5m after creation, deleting')
            await self.call_delete_instance(instance, 'activation_timeout')

        if gce_state in ('STOPPING', 'TERMINATED'):
            log.info(f'{instance} live but stopping or terminated, deactivating')
            await instance.deactivate('terminated')

        if gce_state in ('STAGING', 'RUNNING'):
            last_start_timestamp = spec.get('lastStartTimestamp')
            assert last_start_timestamp is not None, f'lastStartTimestamp does not exist {spec}'
            last_start_time_msecs = dateutil.parser.isoparse(last_start_timestamp).timestamp() * 1000

            if instance.state == 'pending' and time_msecs() - last_start_time_msecs > 5 * 60 * 1000:
                log.exception(f'{instance} did not activate within 5m after starting, deleting')
                await self.call_delete_instance(instance, 'activation_timeout')

        if instance.state == 'inactive':
            log.info(f'{instance} is inactive, deleting')
            await self.call_delete_instance(instance, 'inactive')

        await instance.update_timestamp()

    async def monitor_instances(self):
        if self.instances_by_last_updated:
            # 0 is the smallest (oldest)
            instance = self.instances_by_last_updated[0]
            since_last_updated = time_msecs() - instance.last_updated
            if since_last_updated > 60 * 1000:
                log.info(f'checking on {instance}, last updated {since_last_updated / 1000}s ago')
                await self.check_on_instance(instance)

    async def monitor_instances_loop(self):
        await periodically_call(1, self.monitor_instances)
