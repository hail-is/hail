import random
import logging
import asyncio
import sortedcontainers

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

        user_running_cores_mcpu = {}
        user_total_cores_mcpu = {}
        user_allocated_cores = {}

        pending_users_by_running_cores = sortedcontainers.SortedSet(
            key=lambda user: user_running_cores_mcpu[user])
        allocating_users_by_total_cores = sortedcontainers.SortedSet(
            key=lambda user: user_total_cores_mcpu[user])

        records = self.db.execute_and_fetchall(
            '''
SELECT user, ready_cores_mcpu, running_cores_mcpu
FROM user_resources
WHERE ready_cores_mcpu > 0;
''')

        async for record in records:
            user = record['user']
            user_running_cores_mcpu[user] = record['running_cores_mcpu']
            user_total_cores_mcpu[user] = record['running_cores_mcpu'] + record['ready_cores_mcpu']
            pending_users_by_running_cores.add(user)

        log.info(user_running_cores_mcpu)
        log.info(user_total_cores_mcpu)
        log.info(pending_users_by_running_cores)
        log.info(allocating_users_by_total_cores)

        mark = 0
        log.info(f'free cores mcpu = {free_cores_mcpu}')
        while free_cores_mcpu > 0 and (pending_users_by_running_cores or allocating_users_by_total_cores):
            while True:
                lowest_running = None
                lowest_total = None
                lowest_running_user = None
                lowest_total_user = None

                if pending_users_by_running_cores:
                    lowest_running_user = pending_users_by_running_cores[0]
                    lowest_running = user_running_cores_mcpu[lowest_running_user]
                    log.info(f'lowest_running_user: {lowest_running_user}')
                    log.info(f'lowest_running: {lowest_running}')

                if allocating_users_by_total_cores:
                    lowest_total_user = allocating_users_by_total_cores[0]
                    lowest_total = user_total_cores_mcpu[lowest_total_user]
                    log.info(f'lowest_total_user: {lowest_total_user}')
                    log.info(f'lowest_total: {lowest_total}')

                if lowest_total is not None and lowest_total == mark:
                    log.info(f'have lowest total and it equals the mark')
                    allocating_users_by_total_cores.remove(lowest_total_user)
                elif lowest_running is not None and lowest_running == mark:
                    log.info(f'have lowest running and it equals the mark; add it to allocating users')
                    pending_users_by_running_cores.remove(lowest_running_user)
                    allocating_users_by_total_cores.add(lowest_running_user)
                    log.info(allocating_users_by_total_cores)
                else:
                    break

            log.info(pending_users_by_running_cores)
            log.info(allocating_users_by_total_cores)

            if lowest_running is not None and lowest_total is not None:
                allocation = min(lowest_running, lowest_total)
                log.info(f'allocation = {allocation}; min(running, totla)')
            elif lowest_running is not None:
                allocation = lowest_running
                log.info(f'allocation = {allocation}, lowest_running')
            elif lowest_total is not None:
                allocation = lowest_total
                log.info(f'allocation = {allocation}, lowest_total')
            else:
                log.info(f'not lowest running or lowest total. exiting')
                break

            n_allocating_users = len(allocating_users_by_total_cores)
            cores_to_allocate = n_allocating_users * (allocation - mark)
            log.info(f'cores to allocate: {cores_to_allocate}')

            if cores_to_allocate > free_cores_mcpu:
                mark += free_cores_mcpu / n_allocating_users
                log.info(f'mark = {mark}; ran out of free cores')
                break

            mark = allocation
            log.info(f'mark = {mark}; still have cores to allocate')
            free_cores_mcpu -= cores_to_allocate
            log.info(f'free cores = {free_cores_mcpu}')

        for user in allocating_users_by_total_cores:
            user_allocated_cores[user] = int(mark - user_running_cores_mcpu[user] + 0.5)

        log.info(f'allocated_cores: {user_allocated_cores}')
        return user_allocated_cores

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
