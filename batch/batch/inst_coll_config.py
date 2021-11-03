from typing import Dict, Optional, Tuple
import asyncio
import logging

from gear import Database

from .cloud.gcp.instance_config import GCPSlimInstanceConfig
from .cloud.gcp.resource_utils import family_worker_type_cores_to_gcp_machine_type, GCP_MACHINE_FAMILY
from .cloud.azure.resource_utils import azure_worker_properties_to_machine_type
from .cloud.azure.instance_config import AzureSlimInstanceConfig
from .instance_config import InstanceConfig
from .cloud.resource_utils import (
    adjust_cores_for_memory_request,
    adjust_cores_for_packability,
    machine_type_to_worker_type_cores,
    requested_storage_bytes_to_actual_storage_gib,
    cores_mcpu_to_memory_bytes,
    valid_machine_types,
    local_ssd_size
)

log = logging.getLogger('inst_coll_config')


def instance_config_from_pool_config(pool_config: 'PoolConfig') -> InstanceConfig:
    cloud = pool_config.cloud
    if cloud == 'gcp':
        machine_type = family_worker_type_cores_to_gcp_machine_type(
            GCP_MACHINE_FAMILY, pool_config.worker_type, pool_config.worker_cores)
        return GCPSlimInstanceConfig(machine_type=machine_type,
                                     preemptible=True,
                                     local_ssd_data_disk=pool_config.worker_local_ssd_data_disk,
                                     data_disk_size_gb=pool_config.data_disk_size_gb,
                                     boot_disk_size_gb=pool_config.boot_disk_size_gb,
                                     job_private=False)
    else:
        assert cloud == 'azure'
        machine_type = azure_worker_properties_to_machine_type(
            pool_config.worker_type, pool_config.worker_cores, pool_config.worker_local_ssd_data_disk
        )
        return AzureSlimInstanceConfig(machine_type=machine_type,
                                       preemptible=True,
                                       local_ssd_data_disk=pool_config.worker_local_ssd_data_disk,
                                       data_disk_size_gb=pool_config.data_disk_size_gb,
                                       boot_disk_size_gb=pool_config.boot_disk_size_gb,
                                       job_private=False)


class PreemptibleNotSupportedError(Exception):
    pass


class InstanceCollectionConfig:
    pass


class PoolConfig(InstanceCollectionConfig):
    @staticmethod
    def from_record(record):
        return PoolConfig(
            name=record['name'],
            cloud=record['cloud'],
            worker_type=record['worker_type'],
            worker_cores=record['worker_cores'],
            worker_local_ssd_data_disk=record['worker_local_ssd_data_disk'],
            worker_external_ssd_data_disk_size_gb=record['worker_external_ssd_data_disk_size_gb'],
            enable_standing_worker=record['enable_standing_worker'],
            standing_worker_cores=record['standing_worker_cores'],
            boot_disk_size_gb=record['boot_disk_size_gb'],
            max_instances=record['max_instances'],
            max_live_instances=record['max_live_instances'],
        )

    async def update_database(self, db: Database):
        await db.just_execute(
            '''
UPDATE pools
INNER JOIN inst_colls ON pools.name = inst_colls.name
SET worker_cores = %s,
    worker_local_ssd_data_disk = %s,
    worker_external_ssd_data_disk_size_gb = %s,
    enable_standing_worker = %s,
    standing_worker_cores = %s,
    boot_disk_size_gb = %s,
    max_instances = %s,
    max_live_instances = %s
WHERE pools.name = %s;
''',
            (
                self.worker_cores,
                self.worker_local_ssd_data_disk,
                self.worker_external_ssd_data_disk_size_gb,
                self.enable_standing_worker,
                self.standing_worker_cores,
                self.boot_disk_size_gb,
                self.max_instances,
                self.max_live_instances,
                self.name,
            ),
        )

    def __init__(
        self,
        name: str,
        cloud: str,
        worker_type: str,
        worker_cores: int,
        worker_local_ssd_data_disk: bool,
        worker_external_ssd_data_disk_size_gb: int,
        enable_standing_worker: bool,
        standing_worker_cores: int,
        boot_disk_size_gb: int,
        max_instances: int,
        max_live_instances: int,
    ):
        self.name = name
        self.cloud = cloud
        self.worker_type = worker_type
        self.worker_cores = worker_cores
        self.worker_local_ssd_data_disk = worker_local_ssd_data_disk
        self.worker_external_ssd_data_disk_size_gb = worker_external_ssd_data_disk_size_gb
        self.enable_standing_worker = enable_standing_worker
        self.standing_worker_cores = standing_worker_cores
        self.boot_disk_size_gb = boot_disk_size_gb
        self.max_instances = max_instances
        self.max_live_instances = max_live_instances

        self.instance_config = instance_config_from_pool_config(self)

    @property
    def data_disk_size_gb(self) -> int:
        if self.worker_local_ssd_data_disk:
            return local_ssd_size(self.cloud, self.worker_type, self.worker_cores)
        return self.worker_external_ssd_data_disk_size_gb

    @property
    def data_disk_size_standing_gb(self) -> int:
        if self.worker_local_ssd_data_disk:
            return local_ssd_size(self.cloud, self.worker_type, self.standing_worker_cores)
        return self.worker_external_ssd_data_disk_size_gb

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
        return JobPrivateInstanceManagerConfig(
            record['name'], record['cloud'], record['boot_disk_size_gb'], record['max_instances'], record['max_live_instances']
        )

    def __init__(self, name, cloud, boot_disk_size_gb: int, max_instances, max_live_instances):
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
    async def create(db: Database):
        (name_pool_config, jpim_config), resource_rates = await asyncio.gather(
            InstanceCollectionConfigs.instance_collections_from_db(db),
            InstanceCollectionConfigs.resource_rates_from_db(db))
        return InstanceCollectionConfigs(name_pool_config, jpim_config, resource_rates)

    @staticmethod
    async def instance_collections_from_db(db: Database) -> Tuple[Dict[str, PoolConfig], JobPrivateInstanceManagerConfig]:
        records = db.execute_and_fetchall('''
SELECT inst_colls.*, pools.*
FROM inst_colls
LEFT JOIN pools ON inst_colls.name = pools.name;
''')

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
