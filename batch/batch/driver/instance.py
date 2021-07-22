import aiohttp
import datetime
import logging
import secrets
import humanize
import base64
import json
from typing import Optional

from hailtop.utils import time_msecs, time_msecs_str, retry_transient_errors
from gear import Database

from ..database import check_call_procedure
from ..globals import INSTANCE_VERSION
from ..worker_config import WorkerConfig

log = logging.getLogger('instance')


class Instance:
    @staticmethod
    def from_record(app, inst_coll, record):
        config = record.get('config')
        if config:
            worker_config = WorkerConfig(json.loads(base64.b64decode(config).decode()))
        else:
            worker_config = None

        return Instance(
            app,
            inst_coll,
            record['name'],
            record['state'],
            record['cores_mcpu'],
            record['free_cores_mcpu'],
            record['time_created'],
            record['failed_request_count'],
            record['last_updated'],
            record['ip_address'],
            record['version'],
            record['zone'],
            record['machine_type'],
            bool(record['preemptible']),
            worker_config,
        )

    @staticmethod
    async def create(app, inst_coll, name, activation_token, worker_cores_mcpu, zone, machine_type, preemptible, worker_config: WorkerConfig):
        db: Database = app['db']

        state = 'pending'
        now = time_msecs()
        token = secrets.token_urlsafe(32)
        await db.just_execute(
            '''
INSERT INTO instances (name, state, activation_token, token, cores_mcpu, free_cores_mcpu,
  time_created, last_updated, version, zone, inst_coll, machine_type, preemptible, config)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
''',
            (
                name,
                state,
                activation_token,
                token,
                worker_cores_mcpu,
                worker_cores_mcpu,
                now,
                now,
                INSTANCE_VERSION,
                zone,
                inst_coll.name,
                machine_type,
                preemptible,
                base64.b64encode(json.dumps(worker_config.config).encode()).decode()
            ),
        )
        return Instance(
            app,
            inst_coll,
            name,
            state,
            worker_cores_mcpu,
            worker_cores_mcpu,
            now,
            0,
            now,
            None,
            INSTANCE_VERSION,
            zone,
            machine_type,
            preemptible,
            worker_config
        )

    def __init__(
        self,
        app,
        inst_coll,
        name,
        state,
        cores_mcpu,
        free_cores_mcpu,
        time_created,
        failed_request_count,
        last_updated,
        ip_address,
        version,
        zone,
        machine_type,
        preemptible,
        worker_config: Optional[WorkerConfig],
    ):
        self.db: Database = app['db']
        self.inst_coll = inst_coll
        # pending, active, inactive, deleted
        self._state = state
        self.name = name
        self.cores_mcpu = cores_mcpu
        self._free_cores_mcpu = free_cores_mcpu
        self.time_created = time_created
        self._failed_request_count = failed_request_count
        self._last_updated = last_updated
        self.ip_address = ip_address
        self.version = version
        self.zone = zone
        self.machine_type = machine_type
        self.preemptible = preemptible
        self.worker_config = worker_config

    @property
    def state(self):
        return self._state

    async def activate(self, ip_address, timestamp):
        assert self._state == 'pending'

        rv = await check_call_procedure(
            self.db, 'CALL activate_instance(%s, %s, %s);', (self.name, ip_address, timestamp)
        )

        self.inst_coll.adjust_for_remove_instance(self)
        self._state = 'active'
        self.ip_address = ip_address
        self.inst_coll.adjust_for_add_instance(self)
        self.inst_coll.scheduler_state_changed.set()

        return rv['token']

    async def deactivate(self, reason, timestamp=None):
        if self._state in ('inactive', 'deleted'):
            return

        if not timestamp:
            timestamp = time_msecs()

        rv = await self.db.execute_and_fetchone('CALL deactivate_instance(%s, %s, %s);', (self.name, reason, timestamp))

        if rv['rc'] == 1:
            log.info(f'{self} with in-memory state {self._state} was already deactivated; {rv}')
            assert rv['cur_state'] in ('inactive', 'deleted')

        self.inst_coll.adjust_for_remove_instance(self)
        self._state = 'inactive'
        self._free_cores_mcpu = self.cores_mcpu
        self.inst_coll.adjust_for_add_instance(self)

        # there might be jobs to reschedule
        self.inst_coll.scheduler_state_changed.set()

    async def kill(self):
        async def make_request():
            if self._state in ('inactive', 'deleted'):
                return
            try:
                async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=30)
                ) as session:
                    url = f'http://{self.ip_address}:5000' f'/api/v1alpha/kill'
                    await session.post(url)
            except aiohttp.ClientResponseError as err:
                if err.status == 403:
                    log.info(f'cannot kill {self} -- does not exist at {self.ip_address}')
                    return
                raise

        await retry_transient_errors(make_request)

    async def mark_deleted(self, reason, timestamp):
        if self._state == 'deleted':
            return
        if self._state != 'inactive':
            await self.deactivate(reason, timestamp)

        rv = await self.db.execute_and_fetchone('CALL mark_instance_deleted(%s);', (self.name,))

        if rv['rc'] == 1:
            log.info(f'{self} with in-memory state {self._state} could not be marked deleted; {rv}')
            assert rv['cur_state'] == 'deleted'

        self.inst_coll.adjust_for_remove_instance(self)
        self._state = 'deleted'
        self.inst_coll.adjust_for_add_instance(self)

    @property
    def free_cores_mcpu(self):
        return self._free_cores_mcpu

    def adjust_free_cores_in_memory(self, delta_mcpu):
        self.inst_coll.adjust_for_remove_instance(self)
        self._free_cores_mcpu += delta_mcpu
        self.inst_coll.adjust_for_add_instance(self)

    @property
    def failed_request_count(self):
        return self._failed_request_count

    async def check_is_active_and_healthy(self):
        if self._state == 'active' and self.ip_address:
            try:
                async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)
                ) as session:
                    async with session.get(f'http://{self.ip_address}:5000/healthcheck') as resp:
                        actual_name = (await resp.json()).get('name')
                        if actual_name and actual_name != self.name:
                            return False
                    await self.mark_healthy()
                    return True
            except Exception:
                log.exception(f'while requesting {self} /healthcheck')
                await self.incr_failed_request_count()
        return False

    async def mark_healthy(self):
        if self._state != 'active':
            return

        now = time_msecs()
        changed = (self._failed_request_count > 1) or (now - self._last_updated) > 5000
        if not changed:
            return

        await self.db.execute_update(
            '''
UPDATE instances
SET last_updated = %s,
  failed_request_count = 0
WHERE name = %s;
''',
            (now, self.name),
        )

        self.inst_coll.adjust_for_remove_instance(self)
        self._failed_request_count = 0
        self._last_updated = now
        self.inst_coll.adjust_for_add_instance(self)

    async def incr_failed_request_count(self):
        await self.db.execute_update(
            '''
UPDATE instances
SET failed_request_count = failed_request_count + 1 WHERE name = %s;
''',
            (self.name,),
        )

        self.inst_coll.adjust_for_remove_instance(self)
        self._failed_request_count += 1
        self.inst_coll.adjust_for_add_instance(self)

    @property
    def last_updated(self):
        return self._last_updated

    async def update_timestamp(self):
        now = time_msecs()
        await self.db.execute_update('UPDATE instances SET last_updated = %s WHERE name = %s;', (now, self.name))

        self.inst_coll.adjust_for_remove_instance(self)
        self._last_updated = now
        self.inst_coll.adjust_for_add_instance(self)

    def time_created_str(self):
        return time_msecs_str(self.time_created)

    def last_updated_str(self):
        return humanize.naturaldelta(datetime.timedelta(milliseconds=(time_msecs() - self.last_updated)))

    def __str__(self):
        return f'instance {self.name}'
