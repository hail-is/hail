import asyncio
import json
import logging
import random
import traceback
from typing import Any, AsyncIterator, Dict, List, Tuple

import sortedcontainers

from gear import Database
from hailtop import aiotools
from hailtop.utils import (
    AsyncWorkerPool,
    Notice,
    WaitableSharedPool,
    periodically_call,
    retry_long_running,
    run_if_changed,
    secret_alnum_string,
    time_msecs,
)

from ...batch_format_version import BatchFormatVersion
from ...inst_coll_config import JobPrivateInstanceManagerConfig
from ...instance_config import QuantifiedResource
from ...utils import Box, ExceededSharesCounter, regions_bits_rep_to_regions
from ..exceptions import RegionsNotSupportedError
from ..instance import Instance
from ..job import mark_job_creating, mark_job_errored, schedule_job
from ..resource_manager import CloudResourceManager
from .base import InstanceCollection, InstanceCollectionManager

log = logging.getLogger('job_private_inst_coll')


class JobPrivateInstanceManager(InstanceCollection):
    @staticmethod
    async def create(
        app,
        db: Database,  # BORROWED
        inst_coll_manager: InstanceCollectionManager,
        resource_manager: CloudResourceManager,
        machine_name_prefix: str,
        config: JobPrivateInstanceManagerConfig,
        task_manager: aiotools.BackgroundTaskManager,
    ):
        jpim = JobPrivateInstanceManager(
            app, db, inst_coll_manager, resource_manager, machine_name_prefix, config, task_manager
        )

        log.info(f'initializing {jpim}')

        async for record in db.select_and_fetchall(
            '''
SELECT instances.*, instances_free_cores_mcpu.free_cores_mcpu
FROM instances
INNER JOIN instances_free_cores_mcpu
ON instances.name = instances_free_cores_mcpu.name
WHERE removed = 0 AND inst_coll = %s;
''',
            (jpim.name,),
        ):
            jpim.add_instance(Instance.from_record(app, jpim, record))

        task_manager.ensure_future(
            retry_long_running(
                'create_instances_loop',
                run_if_changed,
                jpim.create_instances_state_changed,
                jpim.create_instances_loop_body,
            )
        )
        task_manager.ensure_future(
            retry_long_running(
                'schedule_jobs_loop', run_if_changed, jpim.scheduler_state_changed, jpim.schedule_jobs_loop_body
            )
        )
        task_manager.ensure_future(periodically_call(15, jpim.bump_scheduler))
        return jpim

    def __init__(
        self,
        app,
        db: Database,  # BORROWED
        inst_coll_manager: InstanceCollectionManager,
        resource_manager: CloudResourceManager,
        machine_name_prefix: str,
        config: JobPrivateInstanceManagerConfig,
        task_manager: aiotools.BackgroundTaskManager,
    ):
        super().__init__(
            db,
            inst_coll_manager,
            resource_manager,
            config.cloud,
            config.name,
            machine_name_prefix,
            is_pool=False,
            max_instances=config.max_instances,
            max_live_instances=config.max_live_instances,
            task_manager=task_manager,
        )
        self.app = app
        global_scheduler_state_changed: Notice = self.app['scheduler_state_changed']
        self.create_instances_state_changed = global_scheduler_state_changed.subscribe()
        self.scheduler_state_changed = asyncio.Event()

        self.async_worker_pool: AsyncWorkerPool = app['async_worker_pool']
        self.exceeded_shares_counter = ExceededSharesCounter()

        self.boot_disk_size_gb = config.boot_disk_size_gb
        self.max_new_instances_per_autoscaler_loop = config.max_new_instances_per_autoscaler_loop
        self.autoscaler_loop_period_secs = config.autoscaler_loop_period_secs
        self.worker_max_idle_time_secs = config.worker_max_idle_time_secs

    def config(self):
        return {
            'name': self.name,
            'worker_disk_size_gb': self.boot_disk_size_gb,
            'max_instances': self.max_instances,
            'max_live_instances': self.max_live_instances,
            'max_new_instances_per_autoscaler_loop': self.max_new_instances_per_autoscaler_loop,
            'autoscaler_loop_period_secs': self.autoscaler_loop_period_secs,
            'worker_max_idle_time_secs': self.worker_max_idle_time_secs,
        }

    async def configure(
        self,
        *,
        boot_disk_size_gb,
        max_instances,
        max_live_instances,
        max_new_instances_per_autoscaler_loop,
        autoscaler_loop_period_secs,
        worker_max_idle_time_secs,
    ):
        await self.db.just_execute(
            '''
UPDATE inst_colls
SET boot_disk_size_gb = %s,
    max_instances = %s,
    max_live_instances = %s,
    max_new_instances_per_autoscaler_loop = %s,
    autoscaler_loop_period_secs = %s,
    worker_max_idle_time_secs = %s
WHERE name = %s;
''',
            (
                boot_disk_size_gb,
                max_instances,
                max_live_instances,
                max_new_instances_per_autoscaler_loop,
                autoscaler_loop_period_secs,
                worker_max_idle_time_secs,
                self.name,
            ),
        )

        self.boot_disk_size_gb = boot_disk_size_gb
        self.max_instances = max_instances
        self.max_live_instances = max_live_instances
        self.max_new_instances_per_autoscaler_loop = max_new_instances_per_autoscaler_loop
        self.autoscaler_loop_period_secs = autoscaler_loop_period_secs
        self.worker_max_idle_time_secs = worker_max_idle_time_secs

    async def bump_scheduler(self):
        self.scheduler_state_changed.set()

    async def schedule_jobs_loop_body(self):
        if self.app['frozen']:
            log.info(f'not scheduling any jobs for {self}; batch is frozen')
            return True

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        n_records_seen = 0
        max_records = 300

        async for record in self.db.select_and_fetchall(
            '''
SELECT jobs.*, batches.format_version, batches.userdata, batches.user, attempts.instance_name, time_ready
FROM job_groups
LEFT JOIN batches ON batches.id = job_groups.batch_id
INNER JOIN jobs ON job_groups.batch_id = jobs.batch_id AND job_groups.job_group_id = jobs.job_group_id
LEFT JOIN jobs_telemetry ON jobs.batch_id = jobs_telemetry.batch_id AND jobs.job_id = jobs_telemetry.job_id
LEFT JOIN attempts ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id
LEFT JOIN instances ON attempts.instance_name = instances.name
WHERE job_groups.state = 'running'
  AND jobs.state = 'Creating'
  AND (jobs.always_run OR NOT jobs.cancelled)
  AND jobs.inst_coll = %s
  AND instances.`state` = 'active'
ORDER BY instances.time_activated ASC
LIMIT %s;
''',
            (self.name, max_records),
        ):
            batch_id = record['batch_id']
            job_id = record['job_id']
            instance_name = record['instance_name']
            id = (batch_id, job_id)
            log.info(f'scheduling job {id}')

            instance = self.name_instance[instance_name]
            n_records_seen += 1

            async def schedule_with_error_handling(app, record, id, instance):
                try:
                    await schedule_job(app, record, instance)
                except Exception:
                    log.info(f'scheduling job {id} on {instance} for {self}', exc_info=True)

            await waitable_pool.call(schedule_with_error_handling, self.app, record, id, instance)

        await waitable_pool.wait()

        if n_records_seen > 0:
            log.info(f'attempted to schedule {n_records_seen} jobs for {self}')

        should_wait = n_records_seen < max_records
        return should_wait

    def max_instances_to_create(self):
        pool_stats = self.current_worker_version_stats
        n_live_instances = pool_stats.n_instances_by_state['pending'] + pool_stats.n_instances_by_state['active']

        return min(
            self.max_live_instances - n_live_instances,
            self.max_instances - self.n_instances,
            # 20 queries/s; our GCE long-run quota
            300,
        )

    async def compute_fair_share(self):
        n_jobs_to_allocate = self.max_instances_to_create()

        user_live_jobs: Dict[str, int] = {}
        user_total_jobs: Dict[str, int] = {}
        result = {}

        pending_users_by_live_jobs = sortedcontainers.SortedSet(key=lambda user: user_live_jobs[user])
        allocating_users_by_total_jobs = sortedcontainers.SortedSet(key=lambda user: user_total_jobs[user])

        records = self.db.execute_and_fetchall(
            '''
SELECT user,
  CAST(COALESCE(SUM(n_ready_jobs), 0) AS SIGNED) AS n_ready_jobs,
  CAST(COALESCE(SUM(n_creating_jobs), 0) AS SIGNED) AS n_creating_jobs,
  CAST(COALESCE(SUM(n_running_jobs), 0) AS SIGNED) AS n_running_jobs
FROM user_inst_coll_resources
WHERE inst_coll = %s
GROUP BY user
HAVING n_ready_jobs + n_creating_jobs + n_running_jobs > 0;
''',
            (self.name,),
        )

        async for record in records:
            user = record['user']
            user_live_jobs[user] = record['n_creating_jobs'] + record['n_running_jobs']
            user_total_jobs[user] = record['n_ready_jobs'] + record['n_creating_jobs'] + record['n_running_jobs']
            pending_users_by_live_jobs.add(user)
            record['n_allocated_jobs'] = 0
            result[user] = record

        def allocate_jobs(user, mark):
            result[user]['n_allocated_jobs'] = mark - user_live_jobs[user]

        mark = 0
        while n_jobs_to_allocate > 0 and (pending_users_by_live_jobs or allocating_users_by_total_jobs):
            lowest_running = None
            lowest_total = None

            if pending_users_by_live_jobs:
                lowest_running_user: str = pending_users_by_live_jobs[0]  # type: ignore
                lowest_running = user_live_jobs[lowest_running_user]
                if lowest_running == mark:
                    pending_users_by_live_jobs.remove(lowest_running_user)
                    allocating_users_by_total_jobs.add(lowest_running_user)
                    continue

            if allocating_users_by_total_jobs:
                lowest_total_user: str = allocating_users_by_total_jobs[0]  # type: ignore
                lowest_total = user_total_jobs[lowest_total_user]
                if lowest_total == mark:
                    allocating_users_by_total_jobs.remove(lowest_total_user)
                    allocate_jobs(lowest_total_user, mark)
                    continue

            allocation = min(c for c in [lowest_running, lowest_total] if c is not None)

            n_allocating_users = len(allocating_users_by_total_jobs)
            jobs_to_allocate = n_allocating_users * (allocation - mark)

            if jobs_to_allocate > n_jobs_to_allocate:
                mark += int(n_jobs_to_allocate / n_allocating_users + 0.5)
                n_jobs_to_allocate = 0
                break

            mark = allocation
            n_jobs_to_allocate -= jobs_to_allocate

        for user in allocating_users_by_total_jobs:
            allocate_jobs(user, mark)

        return result

    async def create_instance(
        self, machine_spec: dict, regions: List[str]
    ) -> Tuple[Instance, List[QuantifiedResource]]:
        machine_type = machine_spec['machine_type']
        preemptible = machine_spec['preemptible']
        storage_gb = machine_spec['storage_gib']
        _, cores = self.resource_manager.worker_type_and_cores(machine_type)
        instance, total_resources_on_instance = await self._create_instance(
            app=self.app,
            cores=cores,
            machine_type=machine_type,
            job_private=True,
            regions=regions,
            preemptible=preemptible,
            max_idle_time_msecs=self.worker_max_idle_time_secs * 1000,
            local_ssd_data_disk=False,
            data_disk_size_gb=storage_gb,
            boot_disk_size_gb=self.boot_disk_size_gb,
        )

        return (instance, total_resources_on_instance)

    async def create_instances_loop_body(self):
        if self.app['frozen']:
            log.info(f'not creating instances for {self}; batch is frozen')
            return True

        start = time_msecs()
        n_instances_created = 0

        user_resources = await self.compute_fair_share()

        total = sum(resources['n_allocated_jobs'] for resources in user_resources.values())
        if not total:
            should_wait = True
            return should_wait
        user_share = {
            user: min(
                int(self.max_new_instances_per_autoscaler_loop * (resources['n_allocated_jobs'] / total) + 0.5),
                resources['n_allocated_jobs'],
            )
            for user, resources in user_resources.items()
        }

        async def user_runnable_jobs(user, remaining) -> AsyncIterator[Dict[str, Any]]:
            async for job_group in self.db.select_and_fetchall(
                '''
SELECT job_groups.batch_id, job_groups.job_group_id, job_groups_cancelled.id IS NOT NULL AS cancelled, userdata, job_groups.user, format_version
FROM job_groups
LEFT JOIN batches ON batches.id = job_groups.batch_id
LEFT JOIN job_groups_cancelled
       ON job_groups.batch_id = job_groups_cancelled.id AND job_groups.job_group_id = job_groups_cancelled.job_group_id
WHERE job_groups.user = %s AND job_groups.`state` = 'running';
''',
                (user,),
            ):
                async for record in self.db.select_and_fetchall(
                    '''
SELECT jobs.batch_id, jobs.job_id, jobs.spec, jobs.cores_mcpu, regions_bits_rep, COALESCE(SUM(instances.state IS NOT NULL AND
  (instances.state = 'pending' OR instances.state = 'active')), 0) as live_attempts, jobs.job_group_id
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_inst_coll_cancelled)
LEFT JOIN attempts ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id
LEFT JOIN instances ON attempts.instance_name = instances.name
WHERE jobs.batch_id = %s AND jobs.job_group_id = %s AND jobs.state = 'Ready' AND always_run = 1 AND jobs.inst_coll = %s
GROUP BY jobs.job_id, jobs.spec, jobs.cores_mcpu
HAVING live_attempts = 0
LIMIT %s;
''',
                    (job_group['batch_id'], job_group['job_group_id'], self.name, remaining.value),
                ):
                    record['batch_id'] = job_group['batch_id']
                    record['userdata'] = job_group['userdata']
                    record['user'] = job_group['user']
                    record['format_version'] = job_group['format_version']
                    yield record
                if not job_group['cancelled']:
                    async for record in self.db.select_and_fetchall(
                        '''
SELECT jobs.batch_id, jobs.job_id, jobs.spec, jobs.cores_mcpu, regions_bits_rep, COALESCE(SUM(instances.state IS NOT NULL AND
  (instances.state = 'pending' OR instances.state = 'active')), 0) as live_attempts, job_group_id
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
LEFT JOIN attempts ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id
LEFT JOIN instances ON attempts.instance_name = instances.name
WHERE jobs.batch_id = %s AND jobs.job_group_id = %s AND jobs.state = 'Ready' AND always_run = 0 AND jobs.inst_coll = %s AND cancelled = 0
GROUP BY jobs.job_id, jobs.spec, jobs.cores_mcpu
HAVING live_attempts = 0
LIMIT %s
''',
                        (job_group['batch_id'], job_group['job_group_id'], self.name, remaining.value),
                    ):
                        record['batch_id'] = job_group['batch_id']
                        record['userdata'] = job_group['userdata']
                        record['user'] = job_group['user']
                        record['format_version'] = job_group['format_version']
                        yield record

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        should_wait = True
        for user, resources in user_resources.items():
            n_allocated_instances = resources['n_allocated_jobs']
            if n_allocated_instances == 0:
                continue

            n_user_instances_created = 0

            share = user_share[user]

            log.info(f'create_instances {self}: user-share: {user}: {share}')

            remaining = Box(share)
            async for record in user_runnable_jobs(user, remaining):
                batch_id = record['batch_id']
                job_id = record['job_id']
                id = (batch_id, job_id)
                attempt_id = secret_alnum_string(6)
                record['attempt_id'] = attempt_id
                job_group_id = record['job_group_id']

                if n_user_instances_created >= n_allocated_instances:
                    if random.random() > self.exceeded_shares_counter.rate():
                        self.exceeded_shares_counter.push(True)
                        self.scheduler_state_changed.set()
                        break
                    self.exceeded_shares_counter.push(False)

                n_instances_created += 1
                n_user_instances_created += 1
                should_wait = False

                log.info(f'creating job private instance for job {id}')

                async def create_instance_with_error_handling(
                    batch_id: int, job_id: int, attempt_id: str, job_group_id: int, record: dict, id: Tuple[int, int]
                ):
                    try:
                        batch_format_version = BatchFormatVersion(record['format_version'])
                        spec = json.loads(record['spec'])
                        machine_spec = batch_format_version.get_spec_machine_spec(spec)

                        regions_bits_rep = record['regions_bits_rep']
                        if regions_bits_rep is None:
                            regions = self.inst_coll_manager.regions
                        else:
                            regions = regions_bits_rep_to_regions(regions_bits_rep, self.app['regions'])

                        assert machine_spec
                        instance, total_resources_on_instance = await self.create_instance(machine_spec, regions)
                        log.info(f'created {instance} for {(batch_id, job_id)}')
                        await mark_job_creating(
                            self.app, batch_id, job_id, attempt_id, instance, time_msecs(), total_resources_on_instance
                        )
                    except RegionsNotSupportedError:
                        await mark_job_errored(
                            self.app,
                            batch_id,
                            job_id,
                            attempt_id,
                            job_group_id,
                            record['user'],
                            record['format_version'],
                            traceback.format_exc(),
                        )
                    except Exception:
                        log.exception(f'while creating job private instance for job {id}', exc_info=True)

                await waitable_pool.call(
                    create_instance_with_error_handling, batch_id, job_id, attempt_id, job_group_id, record, id
                )

                remaining.value -= 1
                if remaining.value <= 0:
                    break

        await waitable_pool.wait()

        end = time_msecs()
        log.info(f'create_instances: created instances for {n_instances_created} jobs in {end - start}ms for {self}')

        await asyncio.sleep(self.autoscaler_loop_period_secs)  # ensure we don't create more instances than GCE limit

        return should_wait

    def __str__(self):
        return f'jpim {self.name}'
