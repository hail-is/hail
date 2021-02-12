import re
import logging

from typing import Dict, Optional, Any

from gear import Database

from .globals import MAX_PERSISTENT_SSD_SIZE_BYTES
from .utils import (adjust_cores_for_memory_request, adjust_cores_for_packability,
                    adjust_cores_for_storage_request, round_storage_bytes_to_gib)


log = logging.getLogger('inst_coll_config')


MACHINE_TYPE_REGEX = re.compile('(?P<machine_family>[^-]+)-(?P<machine_type>[^-]+)-(?P<cores>\\d+)')


def machine_type_to_dict(machine_type: str) -> Optional[Dict[str, Any]]:
    match = MACHINE_TYPE_REGEX.search(machine_type)
    return match.groupdict()


class InstanceCollectionConfig:
    pass


class PoolConfig(InstanceCollectionConfig):
    @staticmethod
    def from_record(record):
        return PoolConfig(name=record['name'],
                          worker_type=record['worker_type'],
                          worker_cores=record['worker_cores'],
                          worker_local_ssd_data_disk=record['worker_local_ssd_data_disk'],
                          worker_pd_ssd_data_disk_size_gb=record['worker_pd_ssd_data_disk_size_gb'],
                          enable_standing_worker=record['enable_standing_worker'],
                          standing_worker_cores=record['standing_worker_cores'],
                          boot_disk_size_gb=record['boot_disk_size_gb'],
                          max_instances=record['max_instances'],
                          max_live_instances=record['max_live_instances'])

    def __init__(self, name, worker_type, worker_cores, worker_local_ssd_data_disk,
                 worker_pd_ssd_data_disk_size_gb, enable_standing_worker, standing_worker_cores,
                 boot_disk_size_gb, max_instances, max_live_instances):
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

    def resources_to_cores_mcpu(self, cores_mcpu, memory_bytes, storage_bytes):
        cores_mcpu = adjust_cores_for_memory_request(cores_mcpu, memory_bytes, self.worker_type)
        cores_mcpu = adjust_cores_for_storage_request(cores_mcpu, storage_bytes, self.worker_cores,
                                                      self.worker_local_ssd_data_disk, self.worker_pd_ssd_data_disk_size_gb)
        cores_mcpu = adjust_cores_for_packability(cores_mcpu)

        if cores_mcpu < self.worker_cores * 1000:
            return cores_mcpu
        return None


class JobPrivateInstanceManagerConfig(InstanceCollectionConfig):
    @staticmethod
    def from_record(record):
        return JobPrivateInstanceManagerConfig(record['name'], record['boot_disk_size_gb'],
                                               record['max_instances'], record['max_live_instances'])

    @staticmethod
    def convert_requests_to_resources(machine_type, storage_bytes):
        if storage_bytes > MAX_PERSISTENT_SSD_SIZE_BYTES:
            return None
        storage_gib = max(10, round_storage_bytes_to_gib(storage_bytes))

        machine_type_dict = machine_type_to_dict(machine_type)
        cores = int(machine_type_dict['cores'])
        cores_mcpu = cores * 1000

        return (cores_mcpu, storage_gib)

    def __init__(self, name, boot_disk_size_gb, max_instances, max_live_instances):
        self.name = name
        self.boot_disk_size_gb = boot_disk_size_gb
        self.max_instances = max_instances
        self.max_live_instances = max_live_instances


class InstanceCollectionConfigs:
    def __init__(self, app):
        self.app = app
        self.db: Database = app['db']
        self.name_config: Dict[str, InstanceCollectionConfig] = {}
        self.name_pool_config: Dict[str, PoolConfig] = {}
        self.jpim_config: Optional[JobPrivateInstanceManagerConfig] = None

    async def async_init(self):
        records = self.db.execute_and_fetchall('''
SELECT inst_colls.*, pools.*
FROM inst_colls
LEFT JOIN pools ON inst_colls.name = pools.name;
''')
        async for record in records:
            is_pool = bool(record['is_pool'])
            if is_pool:
                config = PoolConfig.from_record(record)
                self.name_pool_config[config.name] = config
            else:
                assert self.jpim_config is None
                config = JobPrivateInstanceManagerConfig.from_record(record)
                self.jpim_config = config

            self.name_config[config.name] = config

        assert self.jpim_config is not None

    def select_pool(self, worker_type, cores_mcpu, memory_bytes, storage_bytes):
        for pool in self.name_pool_config.values():
            if pool.worker_type == worker_type:
                maybe_cores_mcpu = pool.resources_to_cores_mcpu(cores_mcpu, memory_bytes, storage_bytes)
                if maybe_cores_mcpu is not None:
                    return (pool.name, maybe_cores_mcpu)
        return None

    def select_job_private(self, machine_type, storage_bytes):  # pylint: disable=no-self-use
        return JobPrivateInstanceManagerConfig.convert_requests_to_resources(machine_type, storage_bytes)
