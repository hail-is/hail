import asyncio
import aiohttp
import datetime
import logging
import secrets
import humanize
from hailtop.utils import time_msecs, time_msecs_str
from gear import Database

from ..database import check_call_procedure
from ..globals import INSTANCE_VERSION

from .instance_pool import InstancePool

log = logging.getLogger('instance')


class Instance:
    @staticmethod
    def from_record(app, record):
        return Instance(
            app, record['name'], record['state'],
            record['cores_mcpu'], record['free_cores_mcpu'],
            record['time_created'], record['failed_request_count'],
            record['last_updated'], record['ip_address'], record['version'],
            record['zone'])

    @staticmethod
    async def create(app, name, activation_token, worker_cores_mcpu, zone):
        db = app['db']

        state = 'pending'
        now = time_msecs()
        token = secrets.token_urlsafe(32)
        await db.just_execute(
            '''
INSERT INTO instances (name, state, activation_token, token, cores_mcpu, free_cores_mcpu, time_created, last_updated, version, zone)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
''',
            (name, state, activation_token, token, worker_cores_mcpu,
             worker_cores_mcpu, now, now, INSTANCE_VERSION, zone))
        return Instance(
            app, name, state, worker_cores_mcpu, worker_cores_mcpu, now,
            0, now, None, INSTANCE_VERSION, zone)

    def __init__(self, app, name: str, state: str, cores_mcpu: int, free_cores_mcpu: int,
                 time_created, failed_request_count: int, last_updated, ip_address: str,
                 version: int, zone: str):
        self.db: Database = app['db']
        self.instance_pool: InstancePool = app['inst_pool']

        self.name = name
        self.cores_mcpu = cores_mcpu
        self.time_created = time_created
        self.ip_address = ip_address
        self.version = version
        self.zone = zone

        self._state = state  # pending, active, inactive, deleted
        self._free_cores_mcpu = free_cores_mcpu
        self._failed_request_count = failed_request_count
        self._last_updated = last_updated

    @property
    def machine_type(self):
        return f'n1-{self.inst_coll.worker_type}-{self.cores}'

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self.instance_pool.n_instances_by_state[self._state] -= 1
        self.instance_pool.n_instances_by_state[new_state] += 1
        self._state = new_state
        if new_state in ('active', 'inactive'):
            self.instance_pool.scheduler_state_changed.set()

    @property
    def free_cores_mcpu(self):
        return self._free_cores_mcpu

    @free_cores_mcpu.setter
    def free_cores_mcpu(self, new_free_cores_mcpu):
        self.instance_pool.live_free_cores_mcpu += new_free_cores_mcpu - self._free_cores_mcpu
        self._free_cores_mcpu = new_free_cores_mcpu
        self.healthy_instances_by_free_cores.remove(self)
        if self.is_healthy():
            self.healthy_instances_by_free_cores.add(self)

    @property
    def last_updated(self):
        return self._last_updated

    @last_updated.setter
    def last_updated(self, new_last_updated):
        self._last_updated = new_last_updated
        self.instance_pool.instances_by_last_updated.remove(self)
        self.instance_pool.instances_by_last_updated.add(self)

    @property
    def failed_request_count(self):
        return self._failed_request_count

    @failed_request_count.setter
    def failed_request_count(self, new_failed_request_count):
        self._failed_request_count = new_failed_request_count
        if new_failed_request_count < 2 and self.state == 'active':
            self.pool.healthy_instances_by_free_cores.add(instance)
        else:
            self.pool.healthy_instances_by_free_cores.discard(instance)

    def is_healthy(self):
        return self.state == 'active' and self.failed_request_count < 2

    def adjust_free_cores_in_memory(self, delta_mcpu):
        self.free_cores_mcpu += delta_mcpu

    async def activate(self, ip_address, timestamp):
        await self.instance_pool.activate(self, ip_address, timestamp)

    async def deactivate(self, reason, timestamp=None):
        await self.instance_pool.deactivate(self, reason, timestamp)

    async def mark_healthy(self):
        if self.state != 'active':
            return

        now = time_msecs()
        changed = (self.failed_request_count > 1) or (now - self.last_updated) > 5000
        if not changed:
            return

        await self.db.execute_update(
            '''
UPDATE instances
SET last_updated = %s,
  failed_request_count = 0
WHERE name = %s;
''',
            (now, self.name))

        self.failed_request_count = 0
        self.last_updated = now

    async def incr_failed_request_count(self):
        await self.db.execute_update(
            '''
UPDATE instances
SET failed_request_count = failed_request_count + 1 WHERE name = %s;
''',
            (self.name,))

        self.failed_request_count += 1


    async def update_timestamp(self):
        now = time_msecs()
        await self.db.execute_update(
            'UPDATE instances SET last_updated = %s WHERE name = %s;',
            (now, self.name))

        self.last_updated = now

    def time_created_str(self):
        return time_msecs_str(self.time_created)

    def last_updated_str(self):
        return humanize.naturaldelta(
            datetime.timedelta(milliseconds=(time_msecs() - self.last_updated)))

    def __str__(self):
        return f'instance {self.name}'
