import random
import logging
import asyncio
from hailtop.utils import time_msecs

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

    async def compute_fair_share(self):
        free_cores_mcpu = sum([worker.free_cores_mcpu for worker in self.inst_pool.healthy_instances_by_free_cores])
        total_cores_mcpu = len(self.inst_pool.healthy_instances_by_free_cores) * self.inst_pool.worker_cores * 1000
        allocated_cores = {}

        records = list(self.db.execute_and_fetchall(
            '''
SELECT user, ready_cores_mcpu, running_cores_mcpu
FROM user_resources
WHERE ready_cores_mcpu > 0 OR running_cores_mcpu > 0
ORDER BY running_cores_mcpu ASC;
'''))

        n_users = len(records)
        for record in records:
            cores_mcpu = max(0, min(record['ready_cores_mcpu'],
                                    free_cores_mcpu,
                                    (total_cores_mcpu / n_users) - record['running_cores_mcpu']))
            free_cores_mcpu -= cores_mcpu
            allocated_cores[record['user']] = cores_mcpu

        free_cores_mcpu_per_user = free_cores_mcpu / n_users
        donated_cores_mcpu = 0
        n_users_left = n_users
        for record in sorted(records, key=lambda x: x['ready_cores_mcpu']):
            cores_mcpu = max(0, min(record['ready_cores_mcpu'] - allocated_cores[record['user']],
                                    free_cores_mcpu_per_user + donated_cores_mcpu / n_users_left))

            allocated_cores[record['user']] += cores_mcpu
            donated_cores_mcpu += free_cores_mcpu_per_user - cores_mcpu
            n_users_left -= 1

        return {user: int(cores + 0.5) for user, cores in allocated_cores.items()}

    async def bump_loop(self):
        while True:
            self.scheduler_state_changed.set()
            self.cancel_state_changed.set()
            await asyncio.sleep(60)

    async def loop(self, name, changed, body):
        delay_secs = 0.1
        changed.clear()
        while True:
            should_wait = False
            try:
                start_time = time_msecs()
                should_wait = await body()
            except Exception:
                end_time = time_msecs()

                log.exception(f'in {name}')

                t = delay_secs * random.uniform(0.7, 1.3)
                await asyncio.sleep(t)

                ran_for_secs = (end_time - start_time) * 1000
                delay_secs = min(
                    max(0.1, 2 * delay_secs - min(0, (ran_for_secs - t) / 2)),
                    30.0)
            if should_wait:
                await changed.wait()
                changed.clear()

    async def cancel_1(self):
        user_records = self.db.execute_and_fetchall(
            '''
SELECT user
FROM user_resources
WHERE running_cores_mcpu > 0;
''')

        should_wait = True
        async for user_record in user_records:
            records = self.db.execute_and_fetchall(
                '''
SELECT jobs.job_id, jobs.batch_id, cores_mcpu, instance_name
FROM jobs
STRAIGHT_JOIN batches ON batches.id = jobs.batch_id
STRAIGHT_JOIN attempts ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id AND jobs.attempt_id = attempts.attempt_id
WHERE jobs.state = 'Running' AND (NOT jobs.always_run) AND batches.closed AND batches.cancelled AND batches.user = %s
LIMIT 50;
''',
                (user_record['user'],))
            async for record in records:
                should_wait = False
                await unschedule_job(self.app, record)

        return should_wait

    async def schedule_1(self):
        fair_share = await self.compute_fair_share()

        should_wait = True

        for user, allocated_cores_mcpu in fair_share.items():
            scheduled_cores_mcpu = 0

            records = self.db.execute_and_fetchall(
                '''
SELECT job_id, batch_id, spec, cores_mcpu,
  ((jobs.cancelled OR batches.cancelled) AND NOT always_run) AS cancel,
  userdata, user
FROM jobs
STRAIGHT_JOIN batches ON batches.id = jobs.batch_id
WHERE jobs.state = 'Ready' AND batches.closed AND batches.user = %s
LIMIT 50;
''',
                (user, ))

            async for record in records:
                batch_id = record['batch_id']
                job_id = record['job_id']
                id = (batch_id, job_id)

                if record['cancel']:
                    log.info(f'cancelling job {id}')
                    await mark_job_complete(self.app, batch_id, job_id, None,
                                            'Cancelled', None, None, None, 'cancelled')
                    should_wait = False
                    continue

                if scheduled_cores_mcpu + record['cores_mcpu'] > allocated_cores_mcpu:
                    log.info(f'user {user} exceeded fair share limits')
                    break

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
                    scheduled_cores_mcpu += record['cores_mcpu']

        return should_wait
