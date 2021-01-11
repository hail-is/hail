import re
import logging

from gear import Database

from ..globals import MAX_PERSISTENT_SSD_SIZE_BYTES
from ..utils import (adjust_cores_for_memory_request, adjust_cores_for_packability,
                     adjust_cores_for_storage_request, round_storage_bytes_to_gib)


MACHINE_TYPE_REGEX = re.compile('(?P<machine_family>[^-]+)-(?P<machine_type>[^-]+)-(?P<cores>\\d+)')

log = logging.getLogger('inst_coll_selector')


class PoolConfig:
    @staticmethod
    def from_record(record):
        return PoolConfig(record['name'], record['worker_type'], record['worker_cores'],
                          record['boot_disk_size_gb'], record['worker_local_ssd_data_disk'],
                          record['worker_pd_ssd_data_disk_size_gb'])

    def __init__(self, name, worker_type, worker_cores, boot_disk_size_gb,
                 worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb):
        self.name = name
        self.worker_type = worker_type
        self.worker_cores = worker_cores
        self.worker_local_ssd_data_disk = worker_local_ssd_data_disk
        self.worker_pd_ssd_data_disk_size_gb = worker_pd_ssd_data_disk_size_gb
        self.boot_disk_size_gb = boot_disk_size_gb

    def resources_to_cores_mcpu(self, cores_mcpu, memory_bytes, storage_bytes):
        cores_mcpu = adjust_cores_for_memory_request(cores_mcpu, memory_bytes, self.worker_type)
        cores_mcpu = adjust_cores_for_storage_request(cores_mcpu, storage_bytes, self.worker_cores,
                                                      self.worker_local_ssd_data_disk, self.worker_pd_ssd_data_disk_size_gb)
        cores_mcpu = adjust_cores_for_packability(cores_mcpu)

        if cores_mcpu < self.worker_cores * 1000:
            return cores_mcpu
        return None


class PoolSelector:
    def __init__(self, app):
        self.app = app
        self.db: Database = app['db']
        self.pool_configs = {}

    async def async_init(self):
        records = self.db.execute_and_fetchall('''
SELECT pools.name, worker_type, worker_cores, boot_disk_size_gb,
  worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb
FROM pools
LEFT JOIN inst_colls ON pools.name = inst_colls.name;
''')
        async for record in records:
            self.pool_configs[record['name']] = PoolConfig.from_record(record)

    def select_pool(self, worker_type, cores_mcpu, memory_bytes, storage_bytes):
        for pool in self.pool_configs.values():
            if pool.worker_type == worker_type:
                maybe_cores_mcpu = pool.resources_to_cores_mcpu(cores_mcpu, memory_bytes, storage_bytes)
                if maybe_cores_mcpu is not None:
                    return (pool.name, maybe_cores_mcpu)
        return None


class JobPrivateInstanceSelector:
    @staticmethod
    def parse_machine_spec_from_resource_requests(machine_type, storage_bytes):
        if storage_bytes > MAX_PERSISTENT_SSD_SIZE_BYTES:
            return None
        storage_gib = max(10, round_storage_bytes_to_gib(storage_bytes))

        machine_type_dict = MACHINE_TYPE_REGEX.match(machine_type)
        if not machine_type_dict:
            return None

        cores = int(machine_type_dict.groupdict()['cores'])
        cores_mcpu = cores * 1000

        return ('job-private', cores_mcpu, storage_gib)
