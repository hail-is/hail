import logging
import asyncio
import secrets
import sortedcontainers
import functools

from hailtop.utils import bounded_gather, retry_long_running, run_if_changed

from ..batch import schedule_job, unschedule_job, mark_job_complete

log = logging.getLogger('driver')

OVERSCHEDULE_CORES_MCPU = 2000


class Box:
    def __init__(self, value):
        self.value = value


class Scheduler:
    def __init__(self, app):
        self.app = app
        self.scheduler_state_changed = app['scheduler_state_changed']
        self.cancel_ready_state_changed = app['cancel_ready_state_changed']
        self.cancel_running_state_changed = app['cancel_running_state_changed']
        self.db = app['db']
        self.inst_pool = app['inst_pool']

    async def async_init(self):
        asyncio.ensure_future(retry_long_running(
            'schedule_loop',
            run_if_changed, self.scheduler_state_changed, self.schedule_loop_body))
        asyncio.ensure_future(retry_long_running(
            'cancel_cancelled_ready_jobs_loop',
            run_if_changed, self.cancel_ready_state_changed, self.cancel_cancelled_ready_jobs_loop_body))
        asyncio.ensure_future(retry_long_running(
            'cancel_cancelled_running_jobs_loop',
            run_if_changed, self.cancel_running_state_changed, self.cancel_cancelled_running_jobs_loop_body))
        asyncio.ensure_future(retry_long_running(
            'bump_loop',
            self.bump_loop))

    async def compute_fair_share(self):
        free_cores_mcpu = sum([
            worker.free_cores_mcpu + OVERSCHEDULE_CORES_MCPU
            for worker in self.inst_pool.healthy_instances_by_free_cores
        ])

        user_running_cores_mcpu = {}
        user_total_cores_mcpu = {}
        result = {}

        pending_users_by_running_cores = sortedcontainers.SortedSet(
            key=lambda user: user_running_cores_mcpu[user])
        allocating_users_by_total_cores = sortedcontainers.SortedSet(
            key=lambda user: user_total_cores_mcpu[user])

        records = self.db.execute_and_fetchall(
            '''
SELECT user,
  CAST(COALESCE(SUM(n_ready_jobs), 0) AS SIGNED) AS n_ready_jobs,
  CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu,
  CAST(COALESCE(SUM(n_running_jobs), 0) AS SIGNED) AS n_running_jobs,
  CAST(COALESCE(SUM(running_cores_mcpu), 0) AS SIGNED) AS running_cores_mcpu
FROM user_resources
GROUP BY user;
''')

        async for record in records:
            user = record['user']
            user_running_cores_mcpu[user] = record['running_cores_mcpu']
            user_total_cores_mcpu[user] = record['running_cores_mcpu'] + record['ready_cores_mcpu']
            pending_users_by_running_cores.add(user)
            record['allocated_cores_mcpu'] = 0
            result[user] = record

        def allocate_cores(user, mark):
            result[user]['allocated_cores_mcpu'] = int(mark - user_running_cores_mcpu[user] + 0.5)

        mark = 0
        while free_cores_mcpu > 0 and (pending_users_by_running_cores or allocating_users_by_total_cores):
            lowest_running = None
            lowest_total = None

            if pending_users_by_running_cores:
                lowest_running_user = pending_users_by_running_cores[0]
                lowest_running = user_running_cores_mcpu[lowest_running_user]
                if lowest_running == mark:
                    pending_users_by_running_cores.remove(lowest_running_user)
                    allocating_users_by_total_cores.add(lowest_running_user)
                    continue

            if allocating_users_by_total_cores:
                lowest_total_user = allocating_users_by_total_cores[0]
                lowest_total = user_total_cores_mcpu[lowest_total_user]
                if lowest_total == mark:
                    allocating_users_by_total_cores.remove(lowest_total_user)
                    allocate_cores(lowest_total_user, mark)
                    continue

            allocation = min([c for c in [lowest_running, lowest_total] if c is not None])

            n_allocating_users = len(allocating_users_by_total_cores)
            cores_to_allocate = n_allocating_users * (allocation - mark)

            if cores_to_allocate > free_cores_mcpu:
                mark += int(free_cores_mcpu / n_allocating_users + 0.5)
                free_cores_mcpu = 0
                break

            mark = allocation
            free_cores_mcpu -= cores_to_allocate

        for user in allocating_users_by_total_cores:
            allocate_cores(user, mark)

        return result

    async def bump_loop(self):
        while True:
            self.scheduler_state_changed.set()
            self.cancel_ready_state_changed.set()
            self.cancel_running_state_changed.set()
            await asyncio.sleep(60)

    async def cancel_cancelled_ready_jobs_loop_body(self):
        records = self.db.select_and_fetchall(
            '''
SELECT user, n_cancelled_ready_jobs
FROM (SELECT user,
    CAST(COALESCE(SUM(n_cancelled_ready_jobs), 0) AS SIGNED) AS n_cancelled_ready_jobs
  FROM user_resources
  GROUP BY user) AS t
WHERE n_cancelled_ready_jobs > 0;
''')
        user_n_cancelled_ready_jobs = {
            record['user']: record['n_cancelled_ready_jobs'] async for record in records
        }

        total = sum(user_n_cancelled_ready_jobs.values())
        user_share = {
            user: max(int(1000 * user_n_jobs / total), 20)
            for user, user_n_jobs in user_n_cancelled_ready_jobs.items()
        }

        async def user_cancelled_ready_jobs(user, remaining):
            async for batch in self.db.select_and_fetchall(
                    '''
SELECT id, cancelled
FROM batches
WHERE user = %s AND `state` = 'running';
''',
                    (user,)):
                if batch['cancelled']:
                    async for record in self.db.select_and_fetchall(
                            '''
SELECT jobs.batch_id, jobs.job_id
FROM jobs
WHERE batch_id = %s AND always_run = 0 AND state = 'Ready'
LIMIT %s;
''',
                            (batch['id'], remaining.value)):
                        yield record
                else:
                    async for record in self.db.select_and_fetchall(
                            '''
SELECT jobs.batch_id, jobs.job_id
FROM jobs
WHERE batch_id = %s AND always_run = 0 AND state = 'Ready' AND cancelled = 1
LIMIT %s;
''',
                            (batch['id'], remaining.value)):
                        yield record

        should_wait = True
        async_work = []
        for user, share in user_share.items():
            remaining = Box(share)
            async for record in user_cancelled_ready_jobs(user, remaining):
                batch_id = record['batch_id']
                job_id = record['job_id']
                id = (batch_id, job_id)
                log.info(f'cancelling job {id}')

                async def cancel_with_error_handling(id, f):
                    try:
                        await f()
                    except Exception:
                        log.info(f'error while cancelling job {id}', exc_info=True)
                async_work.append(
                    functools.partial(
                        cancel_with_error_handling,
                        id,
                        functools.partial(mark_job_complete,
                                          self.app, batch_id, job_id, None, None,
                                          'Cancelled', None, None, None, 'cancelled')))

                remaining.value -= 1
                if remaining.value == 0:
                    should_wait = False
                    break

        await bounded_gather(*[x for x in async_work], parallelism=100)
        return should_wait

    async def cancel_cancelled_running_jobs_loop_body(self):
        records = self.db.select_and_fetchall(
            '''
SELECT user, n_cancelled_running_jobs
FROM (SELECT user,
    CAST(COALESCE(SUM(n_cancelled_running_jobs), 0) AS SIGNED) AS n_cancelled_running_jobs
  FROM user_resources
  GROUP BY user) AS t
WHERE n_cancelled_running_jobs > 0;
''')
        user_n_cancelled_running_jobs = {
            record['user']: record['n_cancelled_running_jobs'] async for record in records
        }

        total = sum(user_n_cancelled_running_jobs.values())
        user_share = {
            user: max(int(1000 * user_n_jobs / total), 20)
            for user, user_n_jobs in user_n_cancelled_running_jobs.items()
        }

        async def user_cancelled_running_jobs(user, remaining):
            async for batch in self.db.select_and_fetchall(
                    '''
SELECT id
FROM batches
WHERE user = %s AND `state` = 'running' AND cancelled = 1;
''',
                    (user,)):
                async for record in self.db.select_and_fetchall(
                        '''
SELECT jobs.batch_id, jobs.job_id, jobs.attempt_id, attempts.instance_name,
FROM jobs
STRAIGHT_JOIN attempts
ON attempts.batch_id = jobs.batch_id AND attempts.job_id = jobs.job_id
AND attempts.attempt_id = jobs.attempt_id
WHERE batch_id = %s AND always_run = 0 AND state = 'Running' AND cancelled = 0
AND jobs.attempt_id IS NOT NULL
LIMIT %s;
''',
                        (batch['id'], remaining.value)):
                    yield record

        async_work = []
        should_wait = True
        for user, share in user_share.items():
            remaining = Box(share)
            async for record in user_cancelled_running_jobs(user, remaining):
                batch_id = record['batch_id']
                job_id = record['job_id']
                id = (batch_id, job_id)

                async def unschedule_with_error_handling(id, instance, f):
                    try:
                        await f()
                    except Exception:
                        log.info(f'unscheduling job {id} on instance {instance}', exc_info=True)
                async_work.append(
                    functools.partial(
                        unschedule_with_error_handling,
                        id, record['instance_name'],
                        functools.partial(
                            unschedule_job,
                            self.app, record)))

                remaining.value -= 1
                if remaining.value == 0:
                    should_wait = False
                    break

        await bounded_gather(*[x for x in async_work], parallelism=100)
        return should_wait

    async def schedule_loop_body(self):
        user_resources = await self.compute_fair_share()

        async def user_runnable_jobs(user, remaining):
            async for batch in self.db.select_and_fetchall(
                    '''
SELECT id, cancelled
FROM batches
WHERE user = %s AND `state` = 'running';
            ''',
                    (user,)):
                async for record in self.db.select_and_fetchall(
                        '''
SELECT job_id, spec, cores_mcpu, userdata, user
FROM jobs
WHERE batch_id = %s AND always_run = 1 AND state = 'Ready'
LIMIT %s;
''',
                        (batch['id'], remaining.value)):
                    yield record
                if not batch['cancelled']:
                    async for record in self.db.select_and_fetchall(
                            '''
SELECT job_id, spec, cores_mcpu, userdata, user
FROM jobs
WHERE batch_id = %s AND always_run = 0 AND state = 'Ready' AND cancelled = 0
LIMIT %s;
''',
                            (batch['id'], remaining.value)):
                        yield record

        total = sum(resources['allocated_cores_mcpu']
                    for resources in user_resources.values())
        user_share = {
            user: max(1000 * resources['allocated_cores_mcpu'] / total, 20)
            for user, resources in user_resources.items()
        }

        async_work = []
        should_wait = True
        for user, resources in user_resources.items():
            allocated_cores_mcpu = resources['allocated_cores_mcpu']
            if allocated_cores_mcpu == 0:
                continue

            scheduled_cores_mcpu = 0
            share = user_share[user]

            remaining = Box(share)
            async for record in user_runnable_jobs(user, remaining):
                batch_id = record['batch_id']
                job_id = record['job_id']
                id = (batch_id, job_id)
                attempt_id = ''.join([secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(6)])
                record['attempt_id'] = attempt_id

                i = self.inst_pool.healthy_instances_by_free_cores.bisect_key_left(record['cores_mcpu'])
                if i < len(self.inst_pool.healthy_instances_by_free_cores):
                    instance = self.inst_pool.healthy_instances_by_free_cores[i]
                else:
                    instance = self.inst_pool.healthy_instances_by_free_cores[-1]
                if (record['cores_mcpu'] <= instance.free_cores_mcpu + OVERSCHEDULE_CORES_MCPU or
                        instance.free_cores_mcpu == 0):
                    instance.adjust_free_cores_in_memory(-record['cores_mcpu'])
                    scheduled_cores_mcpu += record['cores_mcpu']

                    async def schedule_with_error_handling(id, instance, f):
                        try:
                            await f()
                        except Exception:
                            log.info(f'scheduling job {id} on {instance}', exc_info=True)
                    async_work.append(
                        functools.partial(
                            schedule_with_error_handling,
                            id, instance,
                            functools.partial(
                                schedule_job,
                                self.app, record, instance)))

                remaining.value -= 1
                if remaining.value == 0:
                    should_wait = False
                    break
                if scheduled_cores_mcpu + record['cores_mcpu'] > allocated_cores_mcpu:
                    should_wait = False
                    break

        await bounded_gather(*[x for x in async_work], parallelism=100)
        return should_wait
