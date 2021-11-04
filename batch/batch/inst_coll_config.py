from typing import List, Dict, Optional, Any, Tuple
import asyncio
import re
import logging

from gear import Database

from .globals import MAX_PERSISTENT_SSD_SIZE_GIB, valid_machine_types, WORKER_CONFIG_VERSION
from .utils import (
    adjust_cores_for_memory_request,
    adjust_cores_for_packability,
    round_storage_bytes_to_gib,
    cores_mcpu_to_memory_bytes,
)
from .worker_config import WorkerConfig, Disk, DiskType


log = logging.getLogger('inst_coll_config')


class PreemptibleNotSupportedError(Exception):
    pass


MACHINE_TYPE_REGEX = re.compile('(?P<machine_family>[^-]+)-(?P<machine_type>[^-]+)-(?P<cores>\\d+)')


def machine_type_to_dict(machine_type: str) -> Optional[Dict[str, Any]]:
    match = MACHINE_TYPE_REGEX.search(machine_type)
    if match is None:
        return None
    return match.groupdict()


def requested_storage_bytes_to_actual_storage_gib(storage_bytes, allow_zero_storage):
    if storage_bytes > MAX_PERSISTENT_SSD_SIZE_GIB * 1024 ** 3:
        return None
    if allow_zero_storage and storage_bytes == 0:
        return storage_bytes
    # minimum storage for a GCE instance is 10Gi
    return max(10, round_storage_bytes_to_gib(storage_bytes))


class InstanceCollectionConfig:
    pass


class PoolConfig(InstanceCollectionConfig):
    @staticmethod
    def from_record(record):
        return PoolConfig(
            name=record['name'],
            worker_type=record['worker_type'],
            worker_cores=record['worker_cores'],
            worker_local_ssd_data_disk=record['worker_local_ssd_data_disk'],
            worker_pd_ssd_data_disk_size_gb=record['worker_pd_ssd_data_disk_size_gb'],
            enable_standing_worker=record['enable_standing_worker'],
            standing_worker_cores=record['standing_worker_cores'],
            boot_disk_size_gb=record['boot_disk_size_gb'],
            max_instances=record['max_instances'],
            max_live_instances=record['max_live_instances'],
        )

    def __init__(
        self,
        name,
        worker_type,
        worker_cores,
        worker_local_ssd_data_disk,
        worker_pd_ssd_data_disk_size_gb,
        enable_standing_worker,
        standing_worker_cores,
        boot_disk_size_gb,
        max_instances,
        max_live_instances,
    ):
        self.name = name
        self.worker_type = worker_type
        self.worker_cores = worker_cores
        self.worker_local_ssd_data_disk = worker_local_ssd_data_disk
        self.worker_pd_ssd_data_disk_size_gb = worker_pd_ssd_data_disk_size_gb
        self.enable_standing_worker = enable_standing_worker
        self.standing_worker_cores = standing_worker_cores
        self.boot_disk_size_gb = boot_disk_size_gb
        self.max_instances = max_instances
        self.max_live_instances = max_live_instances

        self.worker_config = self.to_worker_config()

    def to_worker_config(self) -> WorkerConfig:
        disks: List[Disk] = [
            {
                'boot': True,
                'project': None,
                'zone': None,
                'type': 'pd-ssd',
                'size': self.boot_disk_size_gb,
                'image': None,
            }
        ]

        typ: DiskType
        if self.worker_local_ssd_data_disk:
            typ = 'local-ssd'
            size = 375
        else:
            typ = 'pd-ssd'
            size = self.worker_pd_ssd_data_disk_size_gb

        disks.append({'boot': False, 'project': None, 'zone': None, 'type': typ, 'size': size, 'image': None})

        config = {
            'version': WORKER_CONFIG_VERSION,
            'instance': {
                'project': None,
                'zone': None,
                'family': 'n1',  # FIXME: need to figure out how to handle variable family types
                'type': self.worker_type,
                'cores': self.worker_cores,
                'preemptible': True,
            },
            'disks': disks,
            'job-private': False,
        }

        return WorkerConfig(config)

    def convert_requests_to_resources(self, cores_mcpu, memory_bytes, storage_bytes):
        storage_gib = requested_storage_bytes_to_actual_storage_gib(storage_bytes, allow_zero_storage=True)
        if storage_gib is None:
            return None

        cores_mcpu = adjust_cores_for_memory_request(cores_mcpu, memory_bytes, self.worker_type)
        cores_mcpu = adjust_cores_for_packability(cores_mcpu)

        memory_bytes = cores_mcpu_to_memory_bytes(cores_mcpu, self.worker_type)

        if cores_mcpu <= self.worker_cores * 1000:
            return (cores_mcpu, memory_bytes, storage_gib)

        return None

    def cost_per_hour(self, resource_rates, cores_mcpu, memory_bytes, storage_gib):
        cost_per_hour = self.worker_config.cost_per_hour(resource_rates, cores_mcpu, memory_bytes, storage_gib)
        return cost_per_hour


class JobPrivateInstanceManagerConfig(InstanceCollectionConfig):
    @staticmethod
    def from_record(record):
        return JobPrivateInstanceManagerConfig(
            record['name'], record['boot_disk_size_gb'], record['max_instances'], record['max_live_instances']
        )

    def __init__(self, name, boot_disk_size_gb, max_instances, max_live_instances):
        self.name = name
        self.boot_disk_size_gb = boot_disk_size_gb
        self.max_instances = max_instances
        self.max_live_instances = max_live_instances

    def convert_requests_to_resources(self, machine_type, storage_bytes):
        storage_gib = requested_storage_bytes_to_actual_storage_gib(storage_bytes, allow_zero_storage=False)
        if storage_gib is None:
            return None

        machine_type_dict = machine_type_to_dict(machine_type)
        if machine_type_dict is None:
            raise ValueError(f'bad machine_type {machine_type}')
        cores = int(machine_type_dict['cores'])
        cores_mcpu = cores * 1000

        memory_bytes = cores_mcpu_to_memory_bytes(cores_mcpu, machine_type_dict['machine_type'])

        return (self.name, cores_mcpu, memory_bytes, storage_gib)


class InstanceCollectionConfigs:
    @staticmethod
    async def create(db: Database):
        (name_pool_config, jpim_config), resource_rates = await asyncio.gather(
            InstanceCollectionConfigs.instance_collections_from_db(db),
            InstanceCollectionConfigs.resource_rates_from_db(db))
        return InstanceCollectionConfigs(name_pool_config, jpim_config, resource_rates)

    @staticmethod
    async def instance_collections_from_db(db: Database) -> Tuple[Dict[str, PoolConfig], JobPrivateInstanceManagerConfig]:
        records = db.execute_and_fetchall(
            '''
SELECT inst_colls.*, pools.*
FROM inst_colls
LEFT JOIN pools ON inst_colls.name = pools.name;
'''
        )
        name_pool_config: Dict[str, PoolConfig] = {}
        jpim_config: Optional[JobPrivateInstanceManagerConfig] = None
        async for record in records:
            if record['is_pool']:
                config = PoolConfig.from_record(record)
                name_pool_config[config.name] = config
            else:
                config = JobPrivateInstanceManagerConfig.from_record(record)
                jpim_config = config
        assert jpim_config is not None
        return name_pool_config, jpim_config

    @staticmethod
    async def resource_rates_from_db(db: Database) -> Dict[str, float]:
        return {
            record['resource']: record['rate']
            async for record in db.execute_and_fetchall('SELECT * FROM resources;')}

    def __init__(self,
                 name_pool_config: Dict[str, PoolConfig],
                 jpim_config: JobPrivateInstanceManagerConfig,
                 resource_rates: Dict[str, float]):
        self.name_pool_config = name_pool_config
        self.jpim_config = jpim_config
        self.resource_rates = resource_rates

    async def refresh(self, db: Database):
        configs, resource_rates = await asyncio.gather(
            InstanceCollectionConfigs.instance_collections_from_db(db),
            InstanceCollectionConfigs.resource_rates_from_db(db))
        self.name_pool_config, self.jpim_config = configs
        self.resource_rates = resource_rates

    def select_pool_from_cost(self, cores_mcpu, memory_bytes, storage_bytes):
        assert self.resource_rates is not None

        optimal_result = None
        optimal_cost = None
        for pool in self.name_pool_config.values():
            result = pool.convert_requests_to_resources(cores_mcpu, memory_bytes, storage_bytes)
            if result:
                maybe_cores_mcpu, maybe_memory_bytes, maybe_storage_gib = result
                maybe_cost = pool.cost_per_hour(
                    self.resource_rates, maybe_cores_mcpu, maybe_memory_bytes, maybe_storage_gib
                )
                if optimal_cost is None or maybe_cost < optimal_cost:
                    optimal_cost = maybe_cost
                    optimal_result = (pool.name, maybe_cores_mcpu, maybe_memory_bytes, maybe_storage_gib)
        return optimal_result

    def select_pool_from_worker_type(self, worker_type, cores_mcpu, memory_bytes, storage_bytes):
        for pool in self.name_pool_config.values():
            if pool.worker_type == worker_type:
                result = pool.convert_requests_to_resources(cores_mcpu, memory_bytes, storage_bytes)
                if result:
                    actual_cores_mcpu, actual_memory_bytes, acutal_storage_gib = result
                    return (pool.name, actual_cores_mcpu, actual_memory_bytes, acutal_storage_gib)
        return None

    def select_job_private(self, machine_type, storage_bytes):
        return self.jpim_config.convert_requests_to_resources(machine_type, storage_bytes)

    def select_inst_coll(
        self, machine_type, preemptible, worker_type, req_cores_mcpu, req_memory_bytes, req_storage_bytes
    ):
        if worker_type is not None and machine_type is None:
            if not preemptible:
                return (
                    None,
                    PreemptibleNotSupportedError('nonpreemptible machines are not supported without a machine type'),
                )
            result = self.select_pool_from_worker_type(
                worker_type=worker_type,
                cores_mcpu=req_cores_mcpu,
                memory_bytes=req_memory_bytes,
                storage_bytes=req_storage_bytes,
            )
        elif worker_type is None and machine_type is None:
            if not preemptible:
                return (
                    None,
                    PreemptibleNotSupportedError('nonpreemptible workers are not supported without a machine type'),
                )
            result = self.select_pool_from_cost(
                cores_mcpu=req_cores_mcpu, memory_bytes=req_memory_bytes, storage_bytes=req_storage_bytes
            )
        else:
            assert machine_type and machine_type in valid_machine_types
            assert worker_type is None
            result = self.select_job_private(machine_type=machine_type, storage_bytes=req_storage_bytes)
        return (result, None)
