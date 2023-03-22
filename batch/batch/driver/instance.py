import base64
import datetime
import json
import logging
import secrets
from typing import Dict, Optional

import aiohttp
import humanize

from gear import Database, transaction
from hailtop import httpx
from hailtop.utils import retry_transient_errors, time_msecs, time_msecs_str

from ..cloud.utils import instance_config_from_config_dict
from ..globals import INSTANCE_VERSION
from ..instance_config import InstanceConfig

log = logging.getLogger('instance')


class Instance:
    @staticmethod
    def from_record(app, inst_coll, record):
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
            record['location'],
            record['machine_type'],
            record['preemptible'],
            instance_config_from_config_dict(json.loads(base64.b64decode(record['instance_config']).decode())),
        )

    @staticmethod
    async def create(
        app,
        inst_coll,
        name: str,
        activation_token,
        cores: int,
        location: str,
        machine_type: str,
        preemptible: bool,
        instance_config: InstanceConfig,
    ) -> 'Instance':
        db: Database = app['db']

        state = 'pending'
        now = time_msecs()
        token = secrets.token_urlsafe(32)

        worker_cores_mcpu = cores * 1000

        @transaction(db)
        async def insert(tx):
            await tx.just_execute(
                '''
INSERT INTO instances (name, state, activation_token, token, cores_mcpu,
  time_created, last_updated, version, location, inst_coll, machine_type, preemptible, instance_config)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
''',
                (
                    name,
                    state,
                    activation_token,
                    token,
                    worker_cores_mcpu,
                    now,
                    now,
                    INSTANCE_VERSION,
                    location,
                    inst_coll.name,
                    machine_type,
                    preemptible,
                    base64.b64encode(json.dumps(instance_config.to_dict()).encode()).decode(),
                ),
            )
            await tx.just_execute(
                '''
INSERT INTO instances_free_cores_mcpu (name, free_cores_mcpu)
VALUES (%s, %s);
''',
                (
                    name,
                    worker_cores_mcpu,
                ),
            )

        await insert()  # pylint: disable=no-value-for-parameter

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
            location,
            machine_type,
            preemptible,
            instance_config,
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
        last_updated: int,
        ip_address,
        version,
        location: str,
        machine_type: str,
        preemptible: bool,
        instance_config: InstanceConfig,
    ):
        self.db: Database = app['db']
        self.client_session: httpx.ClientSession = app['client_session']
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
        self.location = location
        self.machine_type = machine_type
        self.preemptible = preemptible
        self.instance_config = instance_config

    @property
    def state(self):
        return self._state

    async def activate(self, ip_address, timestamp):
        assert self._state == 'pending'

        rv = await self.db.check_call_procedure(
            'CALL activate_instance(%s, %s, %s);', (self.name, ip_address, timestamp), 'activate_instance'
        )

        self.inst_coll.adjust_for_remove_instance(self)
        self._state = 'active'
        self.ip_address = ip_address
        self.inst_coll.adjust_for_add_instance(self)
        self.inst_coll.scheduler_state_changed.set()

        return rv['token']

    async def deactivate(self, reason: str, timestamp: Optional[int] = None):
        if self._state in ('inactive', 'deleted'):
            return

        if not timestamp:
            timestamp = time_msecs()

        rv = await self.db.execute_and_fetchone(
            'CALL deactivate_instance(%s, %s, %s);', (self.name, reason, timestamp), 'deactivate_instance'
        )

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
                await self.client_session.post(
                    f'http://{self.ip_address}:5000/api/v1alpha/kill', timeout=aiohttp.ClientTimeout(total=30)
                )
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
        """A possibly negative measure of the free cores in millicpu.

        See free_cores_mcpu_nonnegative for a more useful property.
        """
        return self._free_cores_mcpu

    @property
    def free_cores_mcpu_nonnegative(self):
        """A nonnegative measure of the free cores in millicpu.

        free_cores_mcpu can be negative temporarily if the worker is oversubscribed.
        """
        return max(0, self.free_cores_mcpu)

    @property
    def used_cores_mcpu_nonnegative(self):
        """A nonnegative measure of the used cores in millicpu.

        The free_cores_mcpu can be negative temporarily if the worker is oversubscribed, so this
        property uses free_cores_mcpu_nonnegative to calculate used cores.

        """
        return self.cores_mcpu - self.free_cores_mcpu_nonnegative

    @property
    def percent_cores_used(self) -> float:
        """The percent of cores currently in use."""
        return self.used_cores_mcpu_nonnegative / self.cores_mcpu

    def cost_per_hour(self, resource_rates: Dict[str, float]) -> float:
        """The instantaneous cost (in USD per hour) generated by this instance (WITH CAVEATS).

        The instantaneous cost of this instance, ignoring attached disks.

        """
        return self.instance_config.actual_cost_per_hour(resource_rates)

    def revenue_per_hour(self, resource_rates: Dict[str, float]) -> float:
        """The instantaneous revenue (in USD per hour) generated by this instance (WITH CAVEATS).

        The instantaneous revenue this instance generates, ignoring attached disks, based on the
        cores in use.

        """
        return self.instance_config.cost_per_hour_from_cores(resource_rates, self.used_cores_mcpu_nonnegative)

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
                async with self.client_session.get(f'http://{self.ip_address}:5000/healthcheck') as resp:
                    actual_name = (await resp.json()).get('name')
                    if actual_name and actual_name != self.name:
                        return False
                await self.mark_healthy()
                return True
            except Exception:
                if (time_msecs() - self.last_updated) / 1000 > 300:
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

        self.inst_coll.adjust_for_remove_instance(self)
        self._failed_request_count = 0
        self._last_updated = now
        self.inst_coll.adjust_for_add_instance(self)

        await self.db.execute_update(
            '''
UPDATE instances
SET last_updated = %s,
  failed_request_count = 0
WHERE name = %s;
''',
            (now, self.name),
            'mark_healthy',
        )

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
        return humanize.naturaldelta(datetime.timedelta(milliseconds=time_msecs() - self.last_updated))

    @property
    def region(self):
        return self.instance_config.region_for(self.location)

    def __str__(self):
        return f'instance {self.name}'
