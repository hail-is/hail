from ..database import check_call_procedure


class Instance:
    @staticmethod
    def from_record(app, record):
        return Instance(
            app, record['id'], record['state'], record['name'],
            record['token'], record['cores_mcpu'],
            record['free_cores_mcpu'], record['ip_address'])

    @staticmethod
    async def create(app, machine_name, inst_token, worker_cores_mcpu):
        db = app['db']

        state = 'pending'
        id = await db.execute_insertone(
            '''
INSERT INTO instances (state, name, token, cores_mcpu, free_cores_mcpu)
VALUES (%s, %s, %s, %s, %s);
''',
            (state, machine_name, inst_token, worker_cores_mcpu,
             worker_cores_mcpu))
        return Instance(
            app, id, state, machine_name, inst_token, worker_cores_mcpu,
            worker_cores_mcpu, None)

    def __init__(self, app, id, state, name, token, cores_mcpu, free_cores_mcpu, ip_address):
        self.db = app['db']
        self.instance_pool = app['inst_pool']
        self.scheduler_state_changed = app['scheduler_state_changed']
        self.id = id
        # pending, active, inactive, deleted
        self._state = state
        self.name = name
        self.token = token
        self.cores_mcpu = cores_mcpu
        self._free_cores_mcpu = free_cores_mcpu
        self.ip_address = ip_address

    @property
    def state(self):
        return self._state

    async def activate(self, ip_address):
        assert self._state == 'pending'

        await check_call_procedure(
            self.db,
            'CALL activate_instance(%s, %s);',
            (self.id, ip_address))

        self.instance_pool.adjust_for_remove_instance(self)
        self._state = 'active'
        self.ip_address = ip_address
        self.instance_pool.adjust_for_add_instance(self)

        self.scheduler_state_changed.set()

    async def deactivate(self):
        if self._state in ('inactive', 'deleted'):
            return

        await check_call_procedure(
            self.db,
            'CALL deactivate_instance(%s);',
            (self.id,))

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
            (self.id,))

        self.instance_pool.adjust_for_remove_instance(self)
        self._state = 'deleted'
        self.instance_pool.adjust_for_add_instance(self)

    @property
    def free_cores_mcpu(self):
        return self._free_cores_mcpu

    def adjust_free_cores(self, delta_mcpu):
        self.instance_pool.adjust_for_remove_instance(self)
        self._free_cores_mcpu += delta_mcpu
        self.instance_pool.adjust_for_add_instance(self)

    def __str__(self):
        return f'instance {self.id} {self.name}'
