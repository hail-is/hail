import logging
import asyncio
import sortedcontainers

from hailtop.utils import (
    AsyncWorkerPool, WaitableSharedPool, retry_long_running, run_if_changed,
    time_msecs, secret_alnum_string)

from ..batch import schedule_job, unschedule_job, mark_job_complete

log = logging.getLogger('driver')


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
        self.async_worker_pool = AsyncWorkerPool(parallelism=100, queue_size=100)

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
            worker.free_cores_mcpu
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
''',
            timer_description='in compute_fair_share: aggregate user_resources')

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
            log.info('bump loop')
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
''',
            timer_description='in cancel_cancelled_ready_jobs: aggregate n_cancelled_ready_jobs')
        user_n_cancelled_ready_jobs = {
            record['user']: record['n_cancelled_ready_jobs'] async for record in records
        }

        total = sum(user_n_cancelled_ready_jobs.values())
        if not total:
            should_wait = True
            return should_wait
        user_share = {
            user: max(int(300 * user_n_jobs / total + 0.5), 20)
            for user, user_n_jobs in user_n_cancelled_ready_jobs.items()
        }

        async def user_cancelled_ready_jobs(user, remaining):
            async for batch in self.db.select_and_fetchall(
                    '''
SELECT id, cancelled
FROM batches
WHERE user = %s AND `state` = 'running';
''',
                    (user,),
                    timer_description=f'in cancel_cancelled_ready_jobs: get {user} running batches'):
                if batch['cancelled']:
                    async for record in self.db.select_and_fetchall(
                            '''
SELECT jobs.job_id
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
WHERE batch_id = %s AND state = 'Ready' AND always_run = 0
LIMIT %s;
''',
                            (batch['id'], remaining.value),
                            timer_description=f'in cancel_cancelled_ready_jobs: get {user} batch {batch["id"]} ready cancelled jobs (1)'):
                        record['batch_id'] = batch['id']
                        yield record
                else:
                    async for record in self.db.select_and_fetchall(
                            '''
SELECT jobs.job_id
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
WHERE batch_id = %s AND state = 'Ready' AND always_run = 0 AND cancelled = 1
LIMIT %s;
''',
                            (batch['id'], remaining.value),
                            timer_description=f'in cancel_cancelled_ready_jobs: get {user} batch {batch["id"]} ready cancelled jobs (2)'):
                        record['batch_id'] = batch['id']
                        yield record

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        should_wait = True
        for user, share in user_share.items():
            remaining = Box(share)
            async for record in user_cancelled_ready_jobs(user, remaining):
                batch_id = record['batch_id']
                job_id = record['job_id']
                id = (batch_id, job_id)
                log.info(f'cancelling job {id}')

                async def cancel_with_error_handling(app, batch_id, job_id, id):
                    try:
                        resources = []
                        await mark_job_complete(
                            app, batch_id, job_id, None, None,
                            'Cancelled', None, None, None, 'cancelled', resources)
                    except Exception:
                        log.info(f'error while cancelling job {id}', exc_info=True)
                await waitable_pool.call(
                    cancel_with_error_handling,
                    self.app, batch_id, job_id, id)

                remaining.value -= 1
                if remaining.value <= 0:
                    should_wait = False
                    break

        await waitable_pool.wait()

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
''',
            timer_description='in cancel_cancelled_running_jobs: aggregate n_cancelled_running_jobs')
        user_n_cancelled_running_jobs = {
            record['user']: record['n_cancelled_running_jobs'] async for record in records
        }

        total = sum(user_n_cancelled_running_jobs.values())
        if not total:
            should_wait = True
            return should_wait
        user_share = {
            user: max(int(300 * user_n_jobs / total + 0.5), 20)
            for user, user_n_jobs in user_n_cancelled_running_jobs.items()
        }

        async def user_cancelled_running_jobs(user, remaining):
            async for batch in self.db.select_and_fetchall(
                    '''
SELECT id
FROM batches
WHERE user = %s AND `state` = 'running' AND cancelled = 1;
''',
                    (user,),
                    timer_description=f'in cancel_cancelled_running_jobs: get {user} cancelled batches'):
                async for record in self.db.select_and_fetchall(
                        '''
SELECT jobs.job_id, attempts.attempt_id, attempts.instance_name
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
STRAIGHT_JOIN attempts
  ON attempts.batch_id = jobs.batch_id AND attempts.job_id = jobs.job_id
WHERE jobs.batch_id = %s AND state = 'Running' AND always_run = 0 AND cancelled = 0
LIMIT %s;
''',
                        (batch['id'], remaining.value),
                        timer_description=f'in cancel_cancelled_running_jobs: get {user} batch {batch["id"]} running cancelled jobs'):
                    record['batch_id'] = batch['id']
                    yield record

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        should_wait = True
        for user, share in user_share.items():
            remaining = Box(share)
            async for record in user_cancelled_running_jobs(user, remaining):
                batch_id = record['batch_id']
                job_id = record['job_id']
                id = (batch_id, job_id)

                async def unschedule_with_error_handling(app, record, instance_name, id):
                    try:
                        await unschedule_job(app, record)
                    except Exception:
                        log.info(f'unscheduling job {id} on instance {instance_name}', exc_info=True)
                await waitable_pool.call(
                    unschedule_with_error_handling, self.app, record, record['instance_name'], id)

                remaining.value -= 1
                if remaining.value <= 0:
                    should_wait = False
                    break

        await waitable_pool.wait()

        return should_wait

    async def schedule_loop_body(self):
        log.info('schedule: starting')
        start = time_msecs()
        n_scheduled = 0

        user_resources = await self.compute_fair_share()

        total = sum(resources['allocated_cores_mcpu']
                    for resources in user_resources.values())
        if not total:
            log.info('schedule: no allocated cores')
            should_wait = True
            return should_wait
        user_share = {
            user: max(int(300 * resources['allocated_cores_mcpu'] / total + 0.5), 20)
            for user, resources in user_resources.items()
        }

        async def user_runnable_jobs(user, remaining):
            async for batch in self.db.select_and_fetchall(
                    '''
SELECT id, cancelled, userdata, user, format_version
FROM batches
WHERE user = %s AND `state` = 'running';
''',
                    (user,),
                    timer_description=f'in schedule: get {user} running batches'):
                async for record in self.db.select_and_fetchall(
                        '''
SELECT job_id, spec, cores_mcpu
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
WHERE batch_id = %s AND state = 'Ready' AND always_run = 1
LIMIT %s;
''',
                        (batch['id'], remaining.value),
                        timer_description=f'in schedule: get {user} batch {batch["id"]} runnable jobs (1)'):
                    record['batch_id'] = batch['id']
                    record['userdata'] = batch['userdata']
                    record['user'] = batch['user']
                    record['format_version'] = batch['format_version']
                    yield record
                if not batch['cancelled']:
                    async for record in self.db.select_and_fetchall(
                            '''
SELECT job_id, spec, cores_mcpu
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
WHERE batch_id = %s AND state = 'Ready' AND always_run = 0 AND cancelled = 0
LIMIT %s;
''',
                            (batch['id'], remaining.value),
                            timer_description=f'in schedule: get {user} batch {batch["id"]} runnable jobs (2)'):
                        record['batch_id'] = batch['id']
                        record['userdata'] = batch['userdata']
                        record['user'] = batch['user']
                        record['format_version'] = batch['format_version']
                        yield record

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        def get_instance(user, cores_mcpu):
            i = self.inst_pool.healthy_instances_by_free_cores.bisect_key_left(cores_mcpu)
            while i < len(self.inst_pool.healthy_instances_by_free_cores):
                instance = self.inst_pool.healthy_instances_by_free_cores[i]
                assert cores_mcpu <= instance.free_cores_mcpu
                if user != 'ci' or (user == 'ci' and instance.zone.startswith('us-central1')):
                    return instance
                i += 1
            return None

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
                attempt_id = secret_alnum_string(6)
                record['attempt_id'] = attempt_id

                if scheduled_cores_mcpu + record['cores_mcpu'] > allocated_cores_mcpu:
                    break

                instance = get_instance(user, record['cores_mcpu'])
                if instance:
                    instance.adjust_free_cores_in_memory(-record['cores_mcpu'])
                    scheduled_cores_mcpu += record['cores_mcpu']
                    n_scheduled += 1
                    should_wait = False

                    async def schedule_with_error_handling(app, record, id, instance):
                        try:
                            await schedule_job(app, record, instance)
                        except Exception:
                            log.info(f'scheduling job {id} on {instance}', exc_info=True)
                    await waitable_pool.call(
                        schedule_with_error_handling, self.app, record, id, instance)

                remaining.value -= 1
                if remaining.value <= 0:
                    break

        await waitable_pool.wait()

        end = time_msecs()
        log.info(f'schedule: scheduled {n_scheduled} jobs in {end - start}ms')

        return should_wait
