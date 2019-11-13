import time
import datetime
import secrets
import humanize

from ..database import check_call_procedure


def fmt_timestamp(t):
    return datetime.datetime.utcfromtimestamp(t).strftime(
        '%Y-%m-%dT%H:%M:%SZ')


class Instance:
    @staticmethod
    def from_record(app, record):
        return Instance(
            app, record['name'], record['state'],
            record['cores_mcpu'], record['free_cores_mcpu'],
            record['time_created'], record['failed_request_count'],
            record['last_updated'], record['ip_address'])

    @staticmethod
    async def create(app, name, activation_token, worker_cores_mcpu):
        db = app['db']

        state = 'pending'
        now = time.time()
        token = secrets.token_urlsafe(32)
        await db.just_execute(
            '''
INSERT INTO instances (name, state, activation_token, token, cores_mcpu, free_cores_mcpu, time_created, last_updated)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
''',
            (name, state, activation_token, token, worker_cores_mcpu,
             worker_cores_mcpu, now, now))
        return Instance(
            app, name, state, worker_cores_mcpu, worker_cores_mcpu, now, 0, now, None)

    def __init__(self, app, name, state, cores_mcpu, free_cores_mcpu,
                 time_created, failed_request_count, last_updated, ip_address):
        self.db = app['db']
        self.instance_pool = app['inst_pool']
        self.scheduler_state_changed = app['scheduler_state_changed']
        # pending, active, inactive, deleted
        self._state = state
        self.name = name
        self.cores_mcpu = cores_mcpu
        self._free_cores_mcpu = free_cores_mcpu
        self.time_created = time_created
        self._failed_request_count = failed_request_count
        self._last_updated = last_updated
        self.ip_address = ip_address

    @property
    def state(self):
        return self._state

    async def activate(self, ip_address):
        assert self._state == 'pending'

        rv = await check_call_procedure(
            self.db,
            'CALL activate_instance(%s, %s);',
            (self.name, ip_address))

        self.instance_pool.adjust_for_remove_instance(self)
        self._state = 'active'
        self.ip_address = ip_address
        self.instance_pool.adjust_for_add_instance(self)

        self.scheduler_state_changed.set()

        return rv['token']

    async def deactivate(self):
        if self._state in ('inactive', 'deleted'):
            return

        await check_call_procedure(
            self.db,
            'CALL deactivate_instance(%s);',
            (self.name,))

        self.instance_pool.adjust_for_remove_instance(self)
        self._state = 'inactive'
        self._free_cores_mcpu = self.cores_mcpu
        self.instance_pool.adjust_for_add_instance(self)

        # there might be jobs to reschedule
        self.scheduler_state_changed.set()

    async def mark_deleted(self):
        if self._state == 'deleted':
            return
        if self._state != 'inactive':
            await self.deactivate()

        await check_call_procedure(
            self.db,
            'CALL mark_instance_deleted(%s);',
            (self.name,))

        self.instance_pool.adjust_for_remove_instance(self)
        self._state = 'deleted'
        self.instance_pool.adjust_for_add_instance(self)

    @property
    def free_cores_mcpu(self):
        return self._free_cores_mcpu

    def adjust_free_cores_in_memory(self, delta_mcpu):
        self.instance_pool.adjust_for_remove_instance(self)
        self._free_cores_mcpu += delta_mcpu
        self.instance_pool.adjust_for_add_instance(self)

    @property
    def failed_request_count(self):
        return self._failed_request_count

    async def mark_healthy(self):
        now = time.time()
        await self.db.execute_update(
            '''
UPDATE instances
SET last_updated = %s,
  failed_request_count = 0
WHERE name = %s;
''',
            (now, self.name))

        self.instance_pool.adjust_for_remove_instance(self)
        self._failed_request_count = 0
        self._last_updated = now
        self.instance_pool.adjust_for_add_instance(self)

    async def incr_failed_request_count(self):
        await self.db.execute_update(
            '''
UPDATE instances
SET failed_request_count = failed_request_count + 1 WHERE name = %s;
''',
            (self.name,))

        self.instance_pool.adjust_for_remove_instance(self)
        self._failed_request_count += 1
        self.instance_pool.adjust_for_add_instance(self)

    @property
    def last_updated(self):
        return self._last_updated

    async def update_timestamp(self):
        now = time.time()
        await self.db.execute_update(
            'UPDATE instances SET last_updated = %s WHERE name = %s;',
            (now, self.name))

        self.instance_pool.adjust_for_remove_instance(self)
        self._last_updated = now
        self.instance_pool.adjust_for_add_instance(self)

    def time_created_str(self):
        return datetime.datetime.utcfromtimestamp(self.time_created).strftime(
            '%Y-%m-%dT%H:%M:%SZ')

    def last_updated_str(self):
        return humanize.naturaldelta(
            datetime.datetime.utcnow() - datetime.datetime.utcfromtimestamp(self.last_updated))

    def __str__(self):
        return f'instance {self.name}'
