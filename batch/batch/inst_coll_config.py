import logging

from typing import Dict, Optional

from gear import Database

from .cloud.resource_utils import (
    adjust_cores_for_memory_request,
    adjust_cores_for_packability,
    machine_type_to_worker_type_cores,
    requested_storage_bytes_to_actual_storage_gib,
    cores_mcpu_to_memory_bytes,
    valid_machine_types,
)
from .cloud.utils import instance_config_from_pool_config


log = logging.getLogger('inst_coll_config')


class PreemptibleNotSupportedError(Exception):
    pass


class InstanceCollectionConfig:
    pass


class PoolConfig(InstanceCollectionConfig):
    @staticmethod
    def from_record(record):
        cloud = record.get('cloud', 'gcp')
        return PoolConfig(
            name=record['name'],
            cloud=cloud,
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
        cloud,
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
        self.cloud = cloud
        self.worker_type = worker_type
        self.worker_cores = worker_cores
        self.worker_local_ssd_data_disk = worker_local_ssd_data_disk
        self.worker_pd_ssd_data_disk_size_gb = worker_pd_ssd_data_disk_size_gb
        self.enable_standing_worker = enable_standing_worker
        self.standing_worker_cores = standing_worker_cores
        self.boot_disk_size_gb = boot_disk_size_gb
        self.max_instances = max_instances
        self.max_live_instances = max_live_instances

        self.instance_config = instance_config_from_pool_config(self)

    def convert_requests_to_resources(self, cores_mcpu, memory_bytes, storage_bytes):
        storage_gib = requested_storage_bytes_to_actual_storage_gib(self.cloud, storage_bytes, allow_zero_storage=True)
        if storage_gib is None:
            return None

        cores_mcpu = adjust_cores_for_memory_request(self.cloud, cores_mcpu, memory_bytes, self.worker_type)
        cores_mcpu = adjust_cores_for_packability(cores_mcpu)

        memory_bytes = cores_mcpu_to_memory_bytes(self.cloud, cores_mcpu, self.worker_type)

        if cores_mcpu <= self.worker_cores * 1000:
            return (cores_mcpu, memory_bytes, storage_gib)

        return None

    def cost_per_hour(self, resource_rates, cores_mcpu, memory_bytes, storage_gib):
        cost_per_hour = self.instance_config.cost_per_hour(resource_rates, cores_mcpu, memory_bytes, storage_gib)
        return cost_per_hour


class JobPrivateInstanceManagerConfig(InstanceCollectionConfig):
    @staticmethod
    def from_record(record):
        cloud = record.get('cloud', 'gcp')
        return JobPrivateInstanceManagerConfig(
            record['name'], cloud, record['boot_disk_size_gb'], record['max_instances'], record['max_live_instances']
        )

    def __init__(self, name, cloud, boot_disk_size_gb, max_instances, max_live_instances):
        self.name = name
        self.cloud = cloud
        self.boot_disk_size_gb = boot_disk_size_gb
        self.max_instances = max_instances
        self.max_live_instances = max_live_instances

    def convert_requests_to_resources(self, machine_type, storage_bytes):
        storage_gib = requested_storage_bytes_to_actual_storage_gib(self.cloud, storage_bytes, allow_zero_storage=False)
        if storage_gib is None:
            return None

        worker_type, cores = machine_type_to_worker_type_cores(self.cloud, machine_type)
        cores_mcpu = cores * 1000
        memory_bytes = cores_mcpu_to_memory_bytes(self.cloud, cores_mcpu, worker_type)

        return (self.name, cores_mcpu, memory_bytes, storage_gib)


class InstanceCollectionConfigs:
    @staticmethod
    async def create(app):
        icc = InstanceCollectionConfigs(app)
        await icc.refresh()
        return icc

    def __init__(self, app):
        self.app = app
        self.db: Database = app['db']
        self.name_config: Dict[str, InstanceCollectionConfig] = {}
        self.name_pool_config: Dict[str, PoolConfig] = {}
        self.resource_rates: Optional[Dict[str, float]] = None
        self.jpim_config: Optional[JobPrivateInstanceManagerConfig] = None

    async def refresh(self):
        records = self.db.execute_and_fetchall(
            '''
SELECT inst_colls.*, pools.*
FROM inst_colls
LEFT JOIN pools ON inst_colls.name = pools.name;
'''
        )
        async for record in records:
            is_pool = bool(record['is_pool'])
            if is_pool:
                config = PoolConfig.from_record(record)
                self.name_pool_config[config.name] = config
            else:
                config = JobPrivateInstanceManagerConfig.from_record(record)
                self.jpim_config = config

            self.name_config[config.name] = config

        assert self.jpim_config is not None

        records = self.db.execute_and_fetchall(
            '''
SELECT * FROM resources;
'''
        )
        self.resource_rates = {record['resource']: record['rate'] async for record in records}

    def select_pool_from_cost(self, cloud, cores_mcpu, memory_bytes, storage_bytes):
        assert self.resource_rates is not None

        optimal_result = None
        optimal_cost = None
        for pool in self.name_pool_config.values():
            if pool.cloud != cloud:
                continue

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

    def select_pool_from_worker_type(self, cloud, worker_type, cores_mcpu, memory_bytes, storage_bytes):
        for pool in self.name_pool_config.values():
            if pool.cloud == cloud and pool.worker_type == worker_type:
                result = pool.convert_requests_to_resources(cores_mcpu, memory_bytes, storage_bytes)
                if result:
                    actual_cores_mcpu, actual_memory_bytes, acutal_storage_gib = result
                    return (pool.name, actual_cores_mcpu, actual_memory_bytes, acutal_storage_gib)
        return None

    def select_job_private(self, cloud, machine_type, storage_bytes):
        if self.jpim_config.cloud != cloud:
            return None
        return self.jpim_config.convert_requests_to_resources(machine_type, storage_bytes)

    def select_inst_coll(
        self, cloud, machine_type, preemptible, worker_type, req_cores_mcpu, req_memory_bytes, req_storage_bytes
    ):
        if worker_type is not None and machine_type is None:
            if not preemptible:
                return (
                    None,
                    PreemptibleNotSupportedError('nonpreemptible machines are not supported without a machine type'),
                )
            result = self.select_pool_from_worker_type(
                cloud=cloud,
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
                cloud=cloud,
                cores_mcpu=req_cores_mcpu,
                memory_bytes=req_memory_bytes,
                storage_bytes=req_storage_bytes
            )
        else:
            assert machine_type and machine_type in valid_machine_types(cloud)
            assert worker_type is None
            result = self.select_job_private(cloud=cloud, machine_type=machine_type, storage_bytes=req_storage_bytes)
        return (result, None)
