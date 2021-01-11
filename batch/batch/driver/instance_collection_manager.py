import re
import collections
import logging

from gear import Database

from .job_private import JobPrivateInstanceCollection
from .pool import Pool
from .zone_monitor import ZoneMonitor

log = logging.getLogger('inst_coll_manager')


class InstanceCollectionManager:
    def __init__(self, app, machine_name_prefix):
        self.app = app
        self.db: Database = app['db']
        self.zone_monitor: ZoneMonitor = app['zone_monitor']
        self.machine_name_prefix = machine_name_prefix
        self.inst_coll_regex = re.compile(f'{self.machine_name_prefix}(?P<inst_coll>.*)-.*')

        self.name_inst_coll = {}
        self.name_pool = {}
        self.name_job_private = {}

    async def async_init(self):
        inst_coll_records = self.db.execute_and_fetchall('''
SELECT * FROM inst_colls;
''')

        async for record in inst_coll_records:
            inst_coll_name = record['name']
            is_pool = record['is_pool']

            if is_pool:
                inst_coll = Pool(self.app, inst_coll_name, self.machine_name_prefix)
                self.name_pool[inst_coll_name] = inst_coll
            else:
                inst_coll = JobPrivateInstanceCollection(self.app, inst_coll_name, self.machine_name_prefix)
                self.name_job_private[inst_coll_name] = inst_coll

            self.name_inst_coll[inst_coll_name] = inst_coll

            await inst_coll.async_init()

        log.info('finished initializing instance collections')

    def shutdown(self):
        for inst_coll in self.name_inst_coll.values():
            inst_coll.shutdown()

    @property
    def pools(self):
        return self.name_pool

    @property
    def job_private_inst_colls(self):
        return self.name_job_private

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
