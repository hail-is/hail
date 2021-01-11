import logging
import asyncio

from hailtop.utils import (WaitableSharedPool, retry_long_running, run_if_changed,
                           AsyncWorkerPool)
from hailtop import aiotools
from gear import Database

from ..batch import unschedule_job, mark_job_complete

log = logging.getLogger('canceller')


class Canceller:
    def __init__(self, app):
        self.app = app
        self.cancel_ready_state_changed: asyncio.Event = app['cancel_ready_state_changed']
        self.cancel_running_state_changed: asyncio.Event = app['cancel_running_state_changed']
        self.db: Database = app['db']
        self.async_worker_pool: AsyncWorkerPool = self.app['async_worker_pool']
        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(retry_long_running(
            'cancel_cancelled_ready_jobs_loop',
            run_if_changed, self.cancel_ready_state_changed, self.cancel_cancelled_ready_jobs_loop_body))
        self.task_manager.ensure_future(retry_long_running(
            'cancel_cancelled_running_jobs_loop',
            run_if_changed, self.cancel_running_state_changed, self.cancel_cancelled_running_jobs_loop_body))

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
            timer_description='in cancel_cancelled_ready_jobs: aggregate n_cancelled_ready_jobs')
        user_n_cancelled_ready_jobs = {
            record['user']: record['n_cancelled_ready_jobs'] async for record in records
        }

        total = sum(user_n_cancelled_ready_jobs.values())
        if total == 0:
            should_wait = True
            return should_wait
        user_share = {
            user: max(int(300 * user_n_jobs / total + 0.5), 20)
            for user, user_n_jobs in user_n_cancelled_ready_jobs.items()
        }

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        should_wait = True
        for user, share in user_share.items():
            n_cancelled = 0
            async for record in self.db.select_and_fetchall(
                    '''
SELECT batch_id, job_id
FROM batches
INNER JOIN jobs ON batches.id = jobs.batch_id
WHERE batches.user = %s
  AND batches.state = 'running'
  AND jobs.state = 'Ready'
  AND NOT jobs.always_run
  AND (jobs.cancelled OR batches.cancelled)
LIMIT %s;
''',
                    (user, share + 1),
                    timer_description='in cancel_cancelled_ready_jobs: get cancelled ready jobs'):
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

                if n_cancelled == share:
                    should_wait = False
                    break

                n_cancelled += 1

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
            timer_description='in cancel_cancelled_running_jobs: aggregate n_cancelled_running_jobs')
        user_n_cancelled_running_jobs = {
            record['user']: record['n_cancelled_running_jobs'] async for record in records
        }

        total = sum(user_n_cancelled_running_jobs.values())
        if total == 0:
            should_wait = True
            return should_wait
        user_share = {
            user: max(int(300 * user_n_jobs / total + 0.5), 20)
            for user, user_n_jobs in user_n_cancelled_running_jobs.items()
        }

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        should_wait = True
        for user, share in user_share.items():
            n_cancelled = 0
            async for record in self.db.select_and_fetchall(
                    '''
SELECT attempts.batch_id, attempts.job_id, attempts.attempt_id,
       attempts.instance_name
FROM batches
INNER JOIN jobs ON batches.id = jobs.batch_id
INNER JOIN attempts ON attempts.batch_id = jobs.batch_id AND
                       attempts.job_id = jobs.job_id
WHERE batches.user = %s
  AND batches.state = 'running'
  AND jobs.state = 'Running'
  AND NOT jobs.always_run
  AND NOT jobs.cancelled
LIMIT %s;
''',
                    (user, share + 1),
                    timer_description='in cancel_cancelled_running_jobs: get cancelled running jobs'):
                batch_id = record['batch_id']
                job_id = record['job_id']
                id = (batch_id, job_id)
                log.info(f'cancelling job {id}')

                async def unschedule_with_error_handling(app, record, instance_name, id):
                    try:
                        await unschedule_job(app, record)
                    except Exception:
                        log.info(f'unscheduling job {id} on instance {instance_name}', exc_info=True)

                await waitable_pool.call(
                    unschedule_with_error_handling, self.app, record, record['instance_name'], id)

                if n_cancelled == share:
                    should_wait = False
                    break

                n_cancelled += 1

        await waitable_pool.wait()

        return should_wait
