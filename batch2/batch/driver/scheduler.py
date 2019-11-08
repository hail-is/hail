import asyncio
import logging

from ..batch import schedule_job, unschedule_job, mark_job_complete

log = logging.getLogger('driver')


class Scheduler:
    def __init__(self, app):
        self.app = app
        self.scheduler_state_changed = app['scheduler_state_changed']
        self.cancel_state_changed = app['cancel_state_changed']
        self.db = app['db']
        self.inst_pool = app['inst_pool']

    async def async_init(self):
        asyncio.ensure_future(self.loop('schedule_loop', self.scheduler_state_changed, self.schedule_1))
        asyncio.ensure_future(self.loop('cancel_loop', self.cancel_state_changed, self.cancel_1))
        asyncio.ensure_future(self.bump_loop())

    async def bump_loop(self):
        while True:
            self.scheduler_state_changed.set()
            self.cancel_state_changed.set()
            await asyncio.sleep(60)

    async def loop(self, name, changed, body):
        changed.clear()
        while True:
            should_wait = False
            try:
                should_wait = await body()
            except Exception:
                # FIXME back off?
                log.exception(f'in {name}')
            if should_wait:
                await changed.wait()
                changed.clear()

    async def cancel_1(self):
        records = self.db.execute_and_fetchall(
            '''
SELECT job_id, batch_id, cores_mcpu, instance_name
FROM jobs
INNER JOIN batches ON batches.id = jobs.batch_id
WHERE jobs.state = 'Running' AND (NOT jobs.always_run) AND batches.closed AND batches.cancelled
LIMIT 50;
''')

        should_wait = True
        async for record in records:
            should_wait = False
            await unschedule_job(self.app, record)

        return should_wait

    async def schedule_1(self):
        records = self.db.execute_and_fetchall(
            '''
SELECT job_id, batch_id, spec, cores_mcpu,
  ((jobs.cancelled OR batches.cancelled) AND NOT always_run) AS cancel,
  userdata, user
FROM jobs
INNER JOIN batches ON batches.id = jobs.batch_id
WHERE jobs.state = 'Ready' AND batches.closed
LIMIT 50;
''')

        should_wait = True
        async for record in records:
            batch_id = record['batch_id']
            job_id = record['job_id']
            id = (batch_id, job_id)

            if record['cancel']:
                log.info(f'cancelling job {id}')
                await mark_job_complete(self.app, batch_id, job_id, 'Cancelled', None)
                should_wait = False
                continue

            i = self.inst_pool.healthy_instances_by_free_cores.bisect_key_left(record['cores_mcpu'])
            if i < len(self.inst_pool.healthy_instances_by_free_cores):
                instance = self.inst_pool.healthy_instances_by_free_cores[i]
                assert record['cores_mcpu'] <= instance.free_cores_mcpu
                log.info(f'scheduling job {id} on {instance}')
                try:
                    await schedule_job(self.app, record, instance)
                except Exception:
                    log.exception(f'while scheduling job {id} on {instance}')
                should_wait = False

        return should_wait
