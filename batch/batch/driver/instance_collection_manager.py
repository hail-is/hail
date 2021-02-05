import re
import collections
import logging
from typing import Dict

from gear import Database

from .instance_collection import InstanceCollection
from .job_private import JobPrivateInstanceManager
from .pool import Pool

log = logging.getLogger('inst_coll_manager')


class InstanceCollectionManager:
    def __init__(self, app, machine_name_prefix):
        self.app = app
        self.db: Database = app['db']
        self.machine_name_prefix = machine_name_prefix
        self.inst_coll_regex = re.compile(f'{self.machine_name_prefix}(?P<inst_coll>.*)-.*')

        self.name_inst_coll: Dict[str, InstanceCollection] = {}
        self.name_pool: Dict[str, Pool] = {}
        self.job_private_inst_manager: JobPrivateInstanceManager = None

    async def async_init(self):
        inst_coll_records = self.db.execute_and_fetchall('''
SELECT inst_colls.*, pools.*
FROM inst_colls
LEFT JOIN pools ON inst_colls.name = pools.name;
''')

        async for record in inst_coll_records:
            inst_coll_name = record['name']
            is_pool = record['is_pool']

            if is_pool:
                inst_coll = Pool.from_record(self.app, self.machine_name_prefix, record)
                self.name_pool[inst_coll_name] = inst_coll
            else:
                inst_coll = JobPrivateInstanceManager.from_record(self.app, self.machine_name_prefix, record)
                assert self.job_private_inst_manager is None
                self.job_private_inst_manager = inst_coll

            self.name_inst_coll[inst_coll_name] = inst_coll

    async def run(self):
        for inst_coll in self.name_inst_coll.values():
            await inst_coll.run()
        log.info('finished initializing instance collections')

    def shutdown(self):
        for inst_coll in self.name_inst_coll.values():
            inst_coll.shutdown()

    @property
    def pools(self):
        return self.name_pool

    @property
    def name_instance(self):
        result = {}
        for inst_coll in self.name_inst_coll.values():
            result.update(inst_coll.name_instance)
        return result

    @property
    def global_live_total_cores_mcpu(self):
        return sum([inst_coll.live_total_cores_mcpu for inst_coll in self.name_inst_coll.values()])

    @property
    def global_live_free_cores_mcpu(self):
        return sum([inst_coll.live_free_cores_mcpu for inst_coll in self.name_inst_coll.values()])

    @property
    def global_n_instances_by_state(self):
        counters = [collections.Counter(inst_coll.n_instances_by_state)
                    for inst_coll in self.name_inst_coll.values()]
        result = collections.Counter({})
        for counter in counters:
            result += counter
        return result

    def get_inst_coll(self, inst_coll_name):
        return self.name_inst_coll.get(inst_coll_name)

    def get_instance(self, inst_name):
        inst_coll_name = None

        match = re.search(self.inst_coll_regex, inst_name)
        if match:
            inst_coll_name = match.groupdict()['inst_coll']
        elif inst_name.startswith(self.machine_name_prefix):
            inst_coll_name = 'standard'

        inst_coll = self.name_inst_coll.get(inst_coll_name)
        if inst_coll:
            return inst_coll.name_instance.get(inst_name)
        return None

    def select_pool(self, worker_type, cores_mcpu, memory_bytes, storage_bytes):
        for pool in self.pools.values():
            if pool.worker_type == worker_type:
                maybe_cores_mcpu = pool.resources_to_cores_mcpu(cores_mcpu, memory_bytes, storage_bytes)
                if maybe_cores_mcpu is not None:
                    return (pool.name, maybe_cores_mcpu)
        return None

    def select_job_private(self, machine_type, storage_bytes):
        result = JobPrivateInstanceManager.convert_requests_to_resources(machine_type, storage_bytes)
        if result:
            cores_mcpu, storage_gib = result
            return (self.job_private_inst_manager.name, cores_mcpu, storage_gib)
        return None
