import asyncio
import logging

log = logging.getLogger('driver')


class Scheduler:
    def __init__(self, changed, db, inst_pool):
        self.changed = changed
        self.db = db
        self.inst_pool = inst_pool

    async def async_init(self):
        asyncio.ensure_future(self.schedule())

    async def schedule_1(self):
        self.changed.clear()
        should_wait = False
        while True:
            if should_wait:
                log.info('waiting for scheduler state to change')
                await self.changed.wait()
                self.changed.clear()

            # FIXME kill running cancelled

            should_wait = True
            async with self.db.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = '''
SELECT job_id, batch_id, spec, cores_mcpu, directory, user
FROM jobs
INNER JOIN batch ON batch.id = jobs.batch_id
WHERE jobs.state = 'Ready'
LIMIT 50;
'''
                    await cursor.execute(sql)
                    records = await cursor.fetchall()

            log.info(f'{len(records)} records ready')

            for record in records:
                # FIXME cancel directly jobs that can be cancelled
                i = self.inst_pool.active_instances_by_free_cores.bisect_key_left(record['cores_mcpu'])
                if i < len(self.inst_pool.active_instances_by_free_cores):
                    instance = self.inst_pool.active_instances_by_free_cores[i]
                    assert record['cores_mcpu'] <= instance.free_cores_mcpu
                    log.info(f'scheduling ({record["batch_id"]}, {record["job_id"]}) on {instance}')
                    await self.inst_pool.schedule_job(record, instance)
                    should_wait = False

    async def schedule(self):
        log.info('scheduler started')

        while True:
            try:
                await self.schedule_1()
            except Exception:
                log.exception('while scheduling')
