import logging
import asyncio

from hailtop.utils import (
    WaitableSharedPool,
    retry_long_running,
    run_if_changed,
    AsyncWorkerPool,
    time_msecs,
    periodically_call,
)
from hailtop import aiotools, aiogoogle
from gear import Database

from .job import unschedule_job, mark_job_complete
from .instance_collection_manager import InstanceCollectionManager
from ..utils import Box

log = logging.getLogger('canceller')


class Canceller:
    def __init__(self, app):
        self.app = app
        self.cancel_ready_state_changed: asyncio.Event = app['cancel_ready_state_changed']
        self.cancel_creating_state_changed: asyncio.Event = app['cancel_creating_state_changed']
        self.cancel_running_state_changed: asyncio.Event = app['cancel_running_state_changed']
        self.db: Database = app['db']
        self.async_worker_pool: AsyncWorkerPool = self.app['async_worker_pool']
        self.compute_client: aiogoogle.ComputeClient = self.app['compute_client']
        self.inst_coll_manager: InstanceCollectionManager = app['inst_coll_manager']

        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(
            retry_long_running(
                'cancel_cancelled_ready_jobs_loop',
                run_if_changed,
                self.cancel_ready_state_changed,
                self.cancel_cancelled_ready_jobs_loop_body,
            )
        )
        self.task_manager.ensure_future(
            retry_long_running(
                'cancel_cancelled_creating_jobs_loop',
                run_if_changed,
                self.cancel_creating_state_changed,
                self.cancel_cancelled_creating_jobs_loop_body,
            )
        )
        self.task_manager.ensure_future(
            retry_long_running(
                'cancel_cancelled_running_jobs_loop',
                run_if_changed,
                self.cancel_running_state_changed,
                self.cancel_cancelled_running_jobs_loop_body,
            )
        )

        self.task_manager.ensure_future(periodically_call(60, self.cancel_orphaned_attempts_loop_body))

    def shutdown(self):
        try:
            self.task_manager.shutdown()
        finally:
            self.async_worker_pool.shutdown()

    async def cancel_cancelled_ready_jobs_loop_body(self):
        records = self.db.select_and_fetchall(
            '''
SELECT user, CAST(COALESCE(SUM(n_cancelled_ready_jobs), 0) AS SIGNED) AS n_cancelled_ready_jobs
FROM user_inst_coll_resources
GROUP BY user
HAVING n_cancelled_ready_jobs > 0;
''',
            timer_description='in cancel_cancelled_ready_jobs: aggregate n_cancelled_ready_jobs',
        )
        user_n_cancelled_ready_jobs = {record['user']: record['n_cancelled_ready_jobs'] async for record in records}

        total = sum(user_n_cancelled_ready_jobs.values())
        if total == 0:
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
                timer_description=f'in cancel_cancelled_ready_jobs: get {user} running batches',
            ):
                if batch['cancelled']:
                    async for record in self.db.select_and_fetchall(
                        '''
SELECT jobs.job_id
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
WHERE batch_id = %s AND state = 'Ready' AND always_run = 0
LIMIT %s;
''',
                        (batch['id'], remaining.value),
                        timer_description=f'in cancel_cancelled_ready_jobs: get {user} batch {batch["id"]} ready cancelled jobs (1)',
                    ):
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
                        timer_description=f'in cancel_cancelled_ready_jobs: get {user} batch {batch["id"]} ready cancelled jobs (2)',
                    ):
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
                            app, batch_id, job_id, None, None, 'Cancelled', None, None, None, 'cancelled', resources
                        )
                    except Exception:
                        log.info(f'error while cancelling job {id}', exc_info=True)

                await waitable_pool.call(cancel_with_error_handling, self.app, batch_id, job_id, id)

                remaining.value -= 1
                if remaining.value <= 0:
                    should_wait = False
                    break

        await waitable_pool.wait()

        return should_wait

    async def cancel_cancelled_creating_jobs_loop_body(self):
        records = self.db.select_and_fetchall(
            '''
SELECT user, CAST(COALESCE(SUM(n_cancelled_creating_jobs), 0) AS SIGNED) AS n_cancelled_creating_jobs
FROM user_inst_coll_resources
GROUP BY user
HAVING n_cancelled_creating_jobs > 0;
''',
            timer_description='in cancel_cancelled_creating_jobs: aggregate n_cancelled_creating_jobs',
        )
        user_n_cancelled_creating_jobs = {
            record['user']: record['n_cancelled_creating_jobs'] async for record in records
        }

        total = sum(user_n_cancelled_creating_jobs.values())
        if total == 0:
            should_wait = True
            return should_wait
        user_share = {
            user: max(int(300 * user_n_jobs / total + 0.5), 20)
            for user, user_n_jobs in user_n_cancelled_creating_jobs.items()
        }

        async def user_cancelled_creating_jobs(user, remaining):
            async for batch in self.db.select_and_fetchall(
                '''
SELECT id
FROM batches
WHERE user = %s AND `state` = 'running' AND cancelled = 1;
''',
                (user,),
                timer_description=f'in cancel_cancelled_creating_jobs: get {user} cancelled batches',
            ):
                async for record in self.db.select_and_fetchall(
                    '''
SELECT jobs.job_id, attempts.attempt_id, attempts.instance_name
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
STRAIGHT_JOIN attempts
  ON attempts.batch_id = jobs.batch_id AND attempts.job_id = jobs.job_id
WHERE jobs.batch_id = %s AND state = 'Creating' AND always_run = 0 AND cancelled = 0
LIMIT %s;
''',
                    (batch['id'], remaining.value),
                    timer_description=f'in cancel_cancelled_creating_jobs: get {user} batch {batch["id"]} creating cancelled jobs',
                ):
                    record['batch_id'] = batch['id']
                    yield record

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        should_wait = True
        for user, share in user_share.items():
            remaining = Box(share)
            async for record in user_cancelled_creating_jobs(user, remaining):
                batch_id = record['batch_id']
                job_id = record['job_id']
                attempt_id = record['attempt_id']
                instance_name = record['instance_name']
                id = (batch_id, job_id)

                async def cancel_with_error_handling(app, batch_id, job_id, attempt_id, instance_name, id):
                    try:
                        resources = []
                        end_time = time_msecs()
                        await mark_job_complete(
                            app,
                            batch_id,
                            job_id,
                            attempt_id,
                            instance_name,
                            'Cancelled',
                            None,
                            None,
                            end_time,
                            'cancelled',
                            resources,
                        )

                        instance = self.inst_coll_manager.get_instance(instance_name)
                        if instance is None:
                            log.warning(f'in cancel_cancelled_creating_jobs: unknown instance {instance_name}')
                            return

                        await instance.inst_coll.call_delete_instance(instance, 'cancelled')

                    except Exception:
                        log.info(f'cancelling creating job {id} on instance {instance_name}', exc_info=True)

                await waitable_pool.call(
                    cancel_with_error_handling, self.app, batch_id, job_id, attempt_id, instance_name, id
                )

                remaining.value -= 1
                if remaining.value <= 0:
                    should_wait = False
                    break

        await waitable_pool.wait()

        return should_wait

    async def cancel_cancelled_running_jobs_loop_body(self):
        records = self.db.select_and_fetchall(
            '''
SELECT user, CAST(COALESCE(SUM(n_cancelled_running_jobs), 0) AS SIGNED) AS n_cancelled_running_jobs
FROM user_inst_coll_resources
GROUP BY user
HAVING n_cancelled_running_jobs > 0;
''',
            timer_description='in cancel_cancelled_running_jobs: aggregate n_cancelled_running_jobs',
        )
        user_n_cancelled_running_jobs = {record['user']: record['n_cancelled_running_jobs'] async for record in records}

        total = sum(user_n_cancelled_running_jobs.values())
        if total == 0:
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
                timer_description=f'in cancel_cancelled_running_jobs: get {user} cancelled batches',
            ):
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
                    timer_description=f'in cancel_cancelled_running_jobs: get {user} batch {batch["id"]} running cancelled jobs',
                ):
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

                await waitable_pool.call(unschedule_with_error_handling, self.app, record, record['instance_name'], id)

                remaining.value -= 1
                if remaining.value <= 0:
                    should_wait = False
                    break

        await waitable_pool.wait()

        return should_wait

    async def cancel_orphaned_attempts_loop_body(self):
        log.info('cancelling orphaned attempts')
        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        n_unscheduled = 0

        async for record in self.db.select_and_fetchall(
            '''
SELECT attempts.*
FROM attempts
INNER JOIN jobs ON attempts.batch_id = jobs.batch_id AND attempts.job_id = jobs.job_id
LEFT JOIN instances ON attempts.instance_name = instances.name
WHERE attempts.start_time IS NOT NULL
  AND attempts.end_time IS NULL
  AND (jobs.state != 'Running' OR jobs.attempt_id != attempts.attempt_id)
  AND instances.`state` = 'active'
ORDER BY attempts.start_time ASC
LIMIT 300;
''',
            timer_description='in cancel_orphaned_attempts',
        ):
            batch_id = record['batch_id']
            job_id = record['job_id']
            attempt_id = record['attempt_id']
            instance_name = record['instance_name']
            id = (batch_id, job_id)

            n_unscheduled += 1

            async def unschedule_with_error_handling(app, record, instance_name, id, attempt_id):
                try:
                    await unschedule_job(app, record)
                except Exception:
                    log.info(
                        f'unscheduling job {id} with orphaned attempt {attempt_id} on instance {instance_name}',
                        exc_info=True,
                    )

            await waitable_pool.call(unschedule_with_error_handling, self.app, record, instance_name, id, attempt_id)

        await waitable_pool.wait()

        log.info(f'cancelled {n_unscheduled} orphaned attempts')
