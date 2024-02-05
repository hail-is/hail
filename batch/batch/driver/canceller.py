import asyncio
import logging
from typing import Any, AsyncIterator, Dict

from gear import Database
from hailtop import aiotools
from hailtop.utils import (
    AsyncWorkerPool,
    WaitableSharedPool,
    periodically_call,
    retry_long_running,
    run_if_changed,
    time_msecs,
)

from ..utils import Box
from .instance_collection import InstanceCollectionManager
from .job import mark_job_complete, unschedule_job

log = logging.getLogger('canceller')


class Canceller:
    @staticmethod
    async def create(app):
        c = Canceller(app)

        c.task_manager.ensure_future(
            retry_long_running(
                'cancel_cancelled_ready_jobs_loop',
                run_if_changed,
                c.cancel_ready_state_changed,
                c.cancel_cancelled_ready_jobs_loop_body,
            )
        )

        c.task_manager.ensure_future(
            retry_long_running(
                'cancel_cancelled_creating_jobs_loop',
                run_if_changed,
                c.cancel_creating_state_changed,
                c.cancel_cancelled_creating_jobs_loop_body,
            )
        )

        c.task_manager.ensure_future(
            retry_long_running(
                'cancel_cancelled_running_jobs_loop',
                run_if_changed,
                c.cancel_running_state_changed,
                c.cancel_cancelled_running_jobs_loop_body,
            )
        )

        c.task_manager.ensure_future(periodically_call(60, c.cancel_orphaned_attempts_loop_body))

        return c

    def __init__(self, app):
        self.app = app
        self.cancel_ready_state_changed: asyncio.Event = app['cancel_ready_state_changed']
        self.cancel_creating_state_changed: asyncio.Event = app['cancel_creating_state_changed']
        self.cancel_running_state_changed: asyncio.Event = app['cancel_running_state_changed']
        self.db: Database = app['db']
        self.async_worker_pool: AsyncWorkerPool = self.app['async_worker_pool']
        self.inst_coll_manager: InstanceCollectionManager = app['driver'].inst_coll_manager

        self.task_manager = aiotools.BackgroundTaskManager()

    async def shutdown_and_wait(self):
        try:
            await self.task_manager.shutdown_and_wait()
        finally:
            await self.async_worker_pool.shutdown_and_wait()

    async def cancel_cancelled_ready_jobs_loop_body(self):
        records = self.db.select_and_fetchall(
            """
SELECT user, CAST(COALESCE(SUM(n_cancelled_ready_jobs), 0) AS SIGNED) AS n_cancelled_ready_jobs
FROM user_inst_coll_resources
GROUP BY user
HAVING n_cancelled_ready_jobs > 0;
""",
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

        async def user_cancelled_ready_jobs(user, remaining) -> AsyncIterator[Dict[str, Any]]:
            async for job_group in self.db.select_and_fetchall(
                """
SELECT job_groups.batch_id, job_groups.job_group_id, job_groups_cancelled.id IS NOT NULL AS cancelled
FROM job_groups
LEFT JOIN job_groups_cancelled
       ON job_groups.batch_id = job_groups_cancelled.id AND
          job_groups.job_group_id = job_groups_cancelled.job_group_id
WHERE user = %s AND `state` = 'running';
""",
                (user,),
            ):
                if job_group['cancelled']:
                    async for record in self.db.select_and_fetchall(
                        """
SELECT jobs.job_id
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
WHERE batch_id = %s AND job_group_id = %s AND state = 'Ready' AND always_run = 0
LIMIT %s;
""",
                        (job_group['batch_id'], job_group['job_group_id'], remaining.value),
                    ):
                        record['batch_id'] = job_group['batch_id']
                        yield record
                else:
                    async for record in self.db.select_and_fetchall(
                        """
SELECT jobs.job_id
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
WHERE batch_id = %s AND job_group_id = %s AND state = 'Ready' AND always_run = 0 AND cancelled = 1
LIMIT %s;
""",
                        (job_group['batch_id'], job_group['job_group_id'], remaining.value),
                    ):
                        record['batch_id'] = job_group['batch_id']
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
                        await mark_job_complete(
                            app, batch_id, job_id, None, None, 'Cancelled', None, None, None, 'cancelled', []
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
            """
SELECT user, CAST(COALESCE(SUM(n_cancelled_creating_jobs), 0) AS SIGNED) AS n_cancelled_creating_jobs
FROM user_inst_coll_resources
GROUP BY user
HAVING n_cancelled_creating_jobs > 0;
""",
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

        async def user_cancelled_creating_jobs(user, remaining) -> AsyncIterator[Dict[str, Any]]:
            async for job_group in self.db.select_and_fetchall(
                """
SELECT job_groups.batch_id, job_groups.job_group_id
FROM job_groups
INNER JOIN job_groups_cancelled
  ON job_groups.batch_id = job_groups_cancelled.id AND
     job_groups.job_group_id = job_groups_cancelled.job_group_id
WHERE user = %s AND `state` = 'running';
""",
                (user,),
            ):
                async for record in self.db.select_and_fetchall(
                    """
SELECT jobs.job_id, attempts.attempt_id, attempts.instance_name
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
STRAIGHT_JOIN attempts
  ON attempts.batch_id = jobs.batch_id AND attempts.job_id = jobs.job_id
WHERE jobs.batch_id = %s AND jobs.job_group_id = %s AND state = 'Creating' AND always_run = 0 AND cancelled = 0
LIMIT %s;
""",
                    (job_group['batch_id'], job_group['job_group_id'], remaining.value),
                ):
                    record['batch_id'] = job_group['batch_id']
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
                            [],
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
            """
SELECT user, CAST(COALESCE(SUM(n_cancelled_running_jobs), 0) AS SIGNED) AS n_cancelled_running_jobs
FROM user_inst_coll_resources
GROUP BY user
HAVING n_cancelled_running_jobs > 0;
""",
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

        async def user_cancelled_running_jobs(user, remaining) -> AsyncIterator[Dict[str, Any]]:
            async for job_group in self.db.select_and_fetchall(
                """
SELECT job_groups.batch_id, job_groups.job_group_id, job_groups_cancelled.id IS NOT NULL AS cancelled
FROM job_groups
INNER JOIN job_groups_cancelled
  ON job_groups.batch_id = job_groups_cancelled.id AND
     job_groups.job_group_id = job_groups_cancelled.job_group_id
WHERE user = %s AND `state` = 'running';
""",
                (user,),
            ):
                async for record in self.db.select_and_fetchall(
                    """
SELECT jobs.job_id, attempts.attempt_id, attempts.instance_name
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
STRAIGHT_JOIN attempts
  ON attempts.batch_id = jobs.batch_id AND attempts.job_id = jobs.job_id
WHERE jobs.batch_id = %s AND jobs.job_group_id = %s AND state = 'Running' AND always_run = 0 AND cancelled = 0
LIMIT %s;
""",
                    (job_group['batch_id'], job_group['job_group_id'], remaining.value),
                ):
                    record['batch_id'] = job_group['batch_id']
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
        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        n_unscheduled = 0

        async for record in self.db.select_and_fetchall(
            """
SELECT attempts.*
FROM attempts
INNER JOIN jobs ON attempts.batch_id = jobs.batch_id AND attempts.job_id = jobs.job_id
LEFT JOIN instances ON attempts.instance_name = instances.name
WHERE attempts.start_time IS NOT NULL
  AND attempts.end_time IS NULL
  AND ((jobs.state != 'Running' AND jobs.state != 'Creating') OR jobs.attempt_id != attempts.attempt_id)
  AND instances.`state` = 'active'
ORDER BY attempts.start_time ASC
LIMIT 300;
""",
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

        if n_unscheduled > 0:
            log.info(f'cancelled {n_unscheduled} orphaned attempts')
