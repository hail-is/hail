import asyncio
import logging
import aiohttp
from hailtop.utils import request_retry_transient_errors

from ..batch import mark_job_complete
from ..database import check_call_procedure

log = logging.getLogger('driver')


class Scheduler:
    def __init__(self, scheduler_state_changed, cancel_state_changed, db, inst_pool):
        self.scheduler_state_changed = scheduler_state_changed
        self.cancel_state_changed = cancel_state_changed
        self.db = db
        self.inst_pool = inst_pool

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

    # FIXME move to InstancePool.unschedule_job?
    async def unschedule_job(self, record):
        batch_id = record['batch_id']
        job_id = record['job_id']

        instance_id = record['instance_id']
        instance = self.inst_pool.id_instance.get(instance_id)
        # FIXME what to do if instance missing?
        if not instance:
            return

        async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            url = (f'http://{instance.ip_address}:5000'
                   f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}/delete')
            try:
                await request_retry_transient_errors(session, 'DELETE', url)
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    pass
                else:
                    raise

        await check_call_procedure(
            self.db.pool,
            'CALL unschedule_job(%s, %s, %s);',
            (batch_id, job_id, instance_id))

        self.inst_pool.adjust_for_remove_instance(instance)
        instance.free_cores_mcpu -= record['cores_mcpu']
        self.inst_pool.adjust_for_add_instance(instance)

    async def cancel_1(self):
        async with self.db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # FIXME could be an expensive query
                sql = '''
SELECT job_id, batch_id, instance_id
FROM jobs
INNER JOIN batch ON batch.id = jobs.batch_id
WHERE jobs.state = 'Running' AND NOT jobs.always_run AND batch.cancelled
LIMIT 50;
'''
                await cursor.execute(sql)
                records = await cursor.fetchall()

        log.info(f'{len(records)} records to cancel')

        for record in records:
            await self.unschedule_job(record)

        should_wait = len(records) == 0
        return should_wait

    async def schedule_1(self):
        async with self.db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = '''
SELECT job_id, batch_id, directory, spec, cores_mcpu,
  always_run,
  (cancel OR batch.cancelled) as cancel,
  batch.user as user
FROM jobs
INNER JOIN batch ON batch.id = jobs.batch_id
WHERE jobs.state = 'Ready'
LIMIT 50;
'''
                await cursor.execute(sql)
                records = await cursor.fetchall()

        log.info(f'{len(records)} records ready')

        should_wait = True
        for record in records:
            batch_id = record['batch_id']
            job_id = record['job_id']
            id = (batch_id, job_id)

            log.info(f'scheduling job {id}')

            if record['cancel'] and not record['always_run']:
                log.info(f'cancelling job {id}')
                await mark_job_complete(
                    self.db, self.scheduler_state_changed, self.inst_pool,
                    batch_id, job_id, 'Cancelled', None)
                should_wait = False
                continue

            i = self.inst_pool.active_instances_by_free_cores.bisect_key_left(record['cores_mcpu'])
            if i < len(self.inst_pool.active_instances_by_free_cores):
                instance = self.inst_pool.active_instances_by_free_cores[i]
                assert record['cores_mcpu'] <= instance.free_cores_mcpu
                log.info(f'scheduling job {id} on {instance}')
                await self.inst_pool.schedule_job(record, instance)
                should_wait = False

        return should_wait
