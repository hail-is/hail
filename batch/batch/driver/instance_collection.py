import asyncio
import sortedcontainers
import logging
import dateutil.parser
import collections
from typing import Dict

from hailtop.utils import time_msecs, secret_alnum_string, periodically_call, time_msecs_str
from hailtop import aiotools
from gear import Database

from .instance import Instance
from .resource_manager import CloudResourceManager, VMState, VMDoesNotExist

log = logging.getLogger('inst_collection')


class InstanceCollection:
    def __init__(self, app, cloud, name, machine_name_prefix, is_pool):
        self.app = app
        self.db: Database = app['db']
        self.resource_manager: CloudResourceManager = app['resource_manager']

        self.cloud = cloud
        self.name = name
        self.machine_name_prefix = f'{machine_name_prefix}{self.name}-'
        self.is_pool = is_pool

        self.name_instance: Dict[str, Instance] = {}
        self.live_free_cores_mcpu_by_location: Dict[str, int] = collections.defaultdict(int)

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
            self.live_free_cores_mcpu_by_location[instance.location] -= max(0, instance.free_cores_mcpu)

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
            self.live_free_cores_mcpu_by_location[instance.location] += max(0, instance.free_cores_mcpu)

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
            await self.resource_manager.delete_vm(instance)
        except VMDoesNotExist:
            log.info(f'{instance} delete already done')
            await self.remove_instance(instance, reason, timestamp)

    async def check_on_instance(self, instance):
        active_and_healthy = await instance.check_is_active_and_healthy()
        if active_and_healthy:
            return

        if (instance.state == 'active'
                and instance.failed_request_count > 5
                and time_msecs() - instance.last_updated > 5 * 60 * 1000):
            log.exception(f'deleting {instance} with {instance.failed_request_count} failed request counts after more than 5 minutes')
            await self.call_delete_instance(instance, 'not_responding')
            return

        try:
            vm_state = await self.resource_manager.get_vm_state(instance)
        except VMDoesNotExist:
            await self.remove_instance(instance, 'does_not_exist')
            return

        log.info(f'{instance} vm_state {vm_state}')

        if (
            vm_state.state == VMState.CREATING
            and instance.state == 'pending'
            and time_msecs() - instance.time_created > 5 * 60 * 1000
        ):
            log.exception(f'{instance} did not provision within 5m after creation, deleting')
            await self.call_delete_instance(instance, 'activation_timeout')

        if vm_state.state == VMState.TERMINATED:
            log.info(f'{instance} live but stopping or terminated, deactivating')
            await instance.deactivate('terminated')

        if vm_state.state == VMState.RUNNING:
            last_start_timestamp = vm_state.last_start_timestamp
            if last_start_timestamp is not None:
                last_start_time_msecs = dateutil.parser.isoparse(last_start_timestamp).timestamp() * 1000
                elapsed_time = time_msecs() - last_start_time_msecs
                if instance.state == 'pending' and elapsed_time > 5 * 60 * 1000:
                    log.exception(f'{instance} did not activate within 5m after starting, deleting')
                    await self.call_delete_instance(instance, 'activation_timeout')
            else:
                elapsed_time = time_msecs() - instance.time_created
                if instance.state == 'pending' and elapsed_time > 5 * 60 * 1000:
                    log.warning(f'{instance} did not activate within {time_msecs_str(elapsed_time)}, ignoring {vm_state.full_spec}')

        if instance.state == 'inactive':
            log.info(f'{instance} is inactive, deleting')
            await self.call_delete_instance(instance, 'inactive')

        await instance.update_timestamp()

    async def monitor_instances(self):
        if self.instances_by_last_updated:
            # [:50] are the fifty smallest (oldest)
            instances = self.instances_by_last_updated[:50]

            async def check(instance):
                since_last_updated = time_msecs() - instance.last_updated
                if since_last_updated > 60 * 1000:
                    log.info(f'checking on {instance}, last updated {since_last_updated / 1000}s ago')
                    await self.check_on_instance(instance)

            await asyncio.gather(*[check(instance) for instance in instances])

    async def monitor_instances_loop(self):
        await periodically_call(1, self.monitor_instances)
