import asyncio
import re
import collections
import logging
from typing import Dict

from gear import Database

from ..inst_coll_config import InstanceCollectionConfigs
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

    async def async_init(self, config_manager: InstanceCollectionConfigs):
        jpim = JobPrivateInstanceManager(self.app, self.machine_name_prefix, config_manager.jpim_config)
        self.job_private_inst_manager = jpim
        self.name_inst_coll[jpim.name] = jpim

        for pool_name, config in config_manager.name_pool_config.items():
            pool = Pool(self.app, self.machine_name_prefix, config)
            self.name_pool[pool_name] = pool
            self.name_inst_coll[pool_name] = pool

        await asyncio.gather(*[inst_coll.async_init() for inst_coll in self.name_inst_coll.values()])

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
        counters = [collections.Counter(inst_coll.n_instances_by_state) for inst_coll in self.name_inst_coll.values()]
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
