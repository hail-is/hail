import asyncio
import logging
import random
from typing import Optional

import sortedcontainers

import prometheus_client as pc

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

from ...batch_configuration import STANDING_WORKER_MAX_IDLE_TIME_MSECS
from ...inst_coll_config import PoolConfig
from ...utils import Box, ExceededSharesCounter
from ..instance import Instance
from ..job import schedule_job
from ..resource_manager import CloudResourceManager
from .base import InstanceCollection, InstanceCollectionManager

log = logging.getLogger('pool')

SCHEDULING_LOOP_RUNS = pc.Counter('scheduling_loop_runs', '')


class Pool(InstanceCollection):
    @staticmethod
    async def create(
        app,
        db: Database,  # BORROWED
        inst_coll_manager: InstanceCollectionManager,
        resource_manager: CloudResourceManager,
        machine_name_prefix: str,
        config: PoolConfig,
        async_worker_pool: AsyncWorkerPool,  # BORROWED
        task_manager: aiotools.BackgroundTaskManager,
    ) -> 'Pool':
        pool = Pool(
            app, db, inst_coll_manager, resource_manager, machine_name_prefix, config, async_worker_pool, task_manager
        )
        log.info(f'initializing {pool}')

        async for record in db.select_and_fetchall(
            '''
SELECT instances.*, instances_free_cores_mcpu.free_cores_mcpu
FROM instances
INNER JOIN instances_free_cores_mcpu
ON instances.name = instances_free_cores_mcpu.name
WHERE removed = 0 AND inst_coll = %s;
''',
            (pool.name,),
        ):
            pool.add_instance(Instance.from_record(app, pool, record))

        return pool

    def __init__(
        self,
        app,
        db: Database,  # BORROWED
        inst_coll_manager: InstanceCollectionManager,
        resource_manager: CloudResourceManager,
        machine_name_prefix: str,
        config: PoolConfig,
        async_worker_pool: AsyncWorkerPool,  # BORROWED
        task_manager: aiotools.BackgroundTaskManager,  # BORROWED
    ):
        super().__init__(
            db,
            inst_coll_manager,
            resource_manager,
            config.cloud,
            config.name,
            machine_name_prefix,
            is_pool=True,
            max_instances=config.max_instances,
            max_live_instances=config.max_live_instances,
            task_manager=task_manager,
        )
        self.app = app
        self.inst_coll_manager = inst_coll_manager
        global_scheduler_state_changed: Notice = self.app['scheduler_state_changed']
        self.scheduler_state_changed = global_scheduler_state_changed.subscribe()
        self.scheduler = PoolScheduler(self.app, self, async_worker_pool, task_manager)

        self.healthy_instances_by_free_cores = sortedcontainers.SortedSet(key=lambda instance: instance.free_cores_mcpu)

        self.worker_type = config.worker_type
        self.worker_cores = config.worker_cores
        self.worker_local_ssd_data_disk = config.worker_local_ssd_data_disk
        self.worker_external_ssd_data_disk_size_gb = config.worker_external_ssd_data_disk_size_gb
        self.enable_standing_worker = config.enable_standing_worker
        self.standing_worker_cores = config.standing_worker_cores
        self.boot_disk_size_gb = config.boot_disk_size_gb
        self.data_disk_size_gb = config.data_disk_size_gb
        self.data_disk_size_standing_gb = config.data_disk_size_standing_gb
        self.preemptible = config.preemptible

        task_manager.ensure_future(self.control_loop())

    @property
    def local_ssd_data_disk(self) -> bool:
        return self.worker_local_ssd_data_disk

    def _default_location(self) -> str:
        return self.inst_coll_manager.location_monitor.default_location()

    def config(self):
        return {
            'name': self.name,
            'worker_type': self.worker_type,
            'worker_cores': self.worker_cores,
            'boot_disk_size_gb': self.boot_disk_size_gb,
            'worker_local_ssd_data_disk': self.worker_local_ssd_data_disk,
            'worker_external_ssd_data_disk_size_gb': self.worker_external_ssd_data_disk_size_gb,
            'enable_standing_worker': self.enable_standing_worker,
            'standing_worker_cores': self.standing_worker_cores,
            'max_instances': self.max_instances,
            'max_live_instances': self.max_live_instances,
            'preemptible': self.preemptible,
        }

    def configure(self, pool_config: PoolConfig):
        assert self.name == pool_config.name
        assert self.cloud == pool_config.cloud
        assert self.worker_type == pool_config.worker_type

        self.worker_cores = pool_config.worker_cores
        self.worker_local_ssd_data_disk = pool_config.worker_local_ssd_data_disk
        self.worker_external_ssd_data_disk_size_gb = pool_config.worker_external_ssd_data_disk_size_gb
        self.enable_standing_worker = pool_config.enable_standing_worker
        self.standing_worker_cores = pool_config.standing_worker_cores
        self.boot_disk_size_gb = pool_config.boot_disk_size_gb
        self.data_disk_size_gb = pool_config.data_disk_size_gb
        self.data_disk_size_standing_gb = pool_config.data_disk_size_standing_gb
        self.max_instances = pool_config.max_instances
        self.max_live_instances = pool_config.max_live_instances
        self.preemptible = pool_config.preemptible

    def adjust_for_remove_instance(self, instance):
        super().adjust_for_remove_instance(instance)
        if instance in self.healthy_instances_by_free_cores:
            self.healthy_instances_by_free_cores.remove(instance)

    def adjust_for_add_instance(self, instance):
        super().adjust_for_add_instance(instance)
        if instance.state == 'active' and instance.failed_request_count <= 1:
            self.healthy_instances_by_free_cores.add(instance)

    def get_instance(self, user, cores_mcpu):
        i = self.healthy_instances_by_free_cores.bisect_key_left(cores_mcpu)
        while i < len(self.healthy_instances_by_free_cores):
            instance = self.healthy_instances_by_free_cores[i]
            assert cores_mcpu <= instance.free_cores_mcpu
            if user != 'ci' or (user == 'ci' and instance.location == self._default_location()):
                return instance
            i += 1
        return None

    async def create_instance(
        self,
        cores: int,
        data_disk_size_gb: int,
        max_idle_time_msecs: Optional[int] = None,
        location: Optional[str] = None,
    ):
        machine_type = self.resource_manager.machine_type(cores, self.worker_type, self.worker_local_ssd_data_disk)
        _, _ = await self._create_instance(
            app=self.app,
            cores=cores,
            machine_type=machine_type,
            job_private=False,
            location=location,
            preemptible=self.preemptible,
            max_idle_time_msecs=max_idle_time_msecs,
            local_ssd_data_disk=self.worker_local_ssd_data_disk,
            data_disk_size_gb=data_disk_size_gb,
            boot_disk_size_gb=self.boot_disk_size_gb,
        )

    async def create_instances_from_ready_cores(self, ready_cores_mcpu, location=None):
        n_live_instances = self.n_instances_by_state['pending'] + self.n_instances_by_state['active']

        if location is None:
            live_free_cores_mcpu = self.live_free_cores_mcpu
        else:
            live_free_cores_mcpu = self.live_free_cores_mcpu_by_location[location]

        instances_needed = (ready_cores_mcpu - live_free_cores_mcpu + (self.worker_cores * 1000) - 1) // (
            self.worker_cores * 1000
        )
        instances_needed = min(
            instances_needed,
            self.max_live_instances - n_live_instances,
            self.max_instances - self.n_instances,
            # 20 queries/s; our GCE long-run quota
            300,
            # n * 16 cores / 15s = excess_scheduling_rate/s = 10/s => n ~= 10
            10,
        )

        if instances_needed > 0:
            log.info(f'creating {instances_needed} new instances')
            # parallelism will be bounded by thread pool
            await asyncio.gather(
                *[
                    self.create_instance(
                        cores=self.worker_cores, data_disk_size_gb=self.data_disk_size_gb, location=location
                    )
                    for _ in range(instances_needed)
                ]
            )

    async def create_instances(self):
        if self.app['frozen']:
            log.info(f'not creating instances for {self}; batch is frozen')
            return

        ready_cores_mcpu_per_user = self.db.select_and_fetchall(
            '''
SELECT user,
  CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu
FROM user_inst_coll_resources
WHERE inst_coll = %s
GROUP BY user;
''',
            (self.name,),
        )

        if ready_cores_mcpu_per_user is None:
            ready_cores_mcpu_per_user = {}
        else:
            ready_cores_mcpu_per_user = {r['user']: r['ready_cores_mcpu'] async for r in ready_cores_mcpu_per_user}

        ready_cores_mcpu = sum(ready_cores_mcpu_per_user.values())

        free_cores_mcpu = sum([worker.free_cores_mcpu for worker in self.healthy_instances_by_free_cores])
        free_cores = free_cores_mcpu / 1000

        log.info(
            f'{self} n_instances {self.n_instances} {self.n_instances_by_state}'
            f' free_cores {free_cores} live_free_cores {self.live_free_cores_mcpu / 1000}'
            f' ready_cores {ready_cores_mcpu / 1000}'
        )

        if ready_cores_mcpu > 0 and free_cores < 500:
            await self.create_instances_from_ready_cores(ready_cores_mcpu)

        default_location = self._default_location()
        ci_ready_cores_mcpu = ready_cores_mcpu_per_user.get('ci', 0)
        if ci_ready_cores_mcpu > 0 and self.live_free_cores_mcpu_by_location[default_location] == 0:
            await self.create_instances_from_ready_cores(ci_ready_cores_mcpu, location=default_location)

        n_live_instances = self.n_instances_by_state['pending'] + self.n_instances_by_state['active']
        if self.enable_standing_worker and n_live_instances == 0 and self.max_instances > 0:
            await self.create_instance(
                cores=self.standing_worker_cores,
                data_disk_size_gb=self.data_disk_size_standing_gb,
                max_idle_time_msecs=STANDING_WORKER_MAX_IDLE_TIME_MSECS,
            )

    async def control_loop(self):
        await periodically_call(15, self.create_instances)

    def __str__(self):
        return f'pool {self.name}'


class PoolScheduler:
    def __init__(
        self,
        app,
        pool: Pool,
        async_worker_pool: AsyncWorkerPool,  # BORROWED
        task_manager: aiotools.BackgroundTaskManager,  # BORROWED
    ):
        self.app = app
        self.scheduler_state_changed = pool.scheduler_state_changed
        self.db: Database = app['db']
        self.pool = pool
        self.async_worker_pool = async_worker_pool
        self.exceeded_shares_counter = ExceededSharesCounter()
        task_manager.ensure_future(
            retry_long_running('schedule_loop', run_if_changed, self.scheduler_state_changed, self.schedule_loop_body)
        )

    async def compute_fair_share(self):
        free_cores_mcpu = sum([worker.free_cores_mcpu for worker in self.pool.healthy_instances_by_free_cores])

        user_running_cores_mcpu = {}
        user_total_cores_mcpu = {}
        result = {}

        pending_users_by_running_cores = sortedcontainers.SortedSet(key=lambda user: user_running_cores_mcpu[user])
        allocating_users_by_total_cores = sortedcontainers.SortedSet(key=lambda user: user_total_cores_mcpu[user])

        records = self.db.execute_and_fetchall(
            '''
SELECT user,
  CAST(COALESCE(SUM(n_ready_jobs), 0) AS SIGNED) AS n_ready_jobs,
  CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu,
  CAST(COALESCE(SUM(n_running_jobs), 0) AS SIGNED) AS n_running_jobs,
  CAST(COALESCE(SUM(running_cores_mcpu), 0) AS SIGNED) AS running_cores_mcpu
FROM user_inst_coll_resources
WHERE inst_coll = %s
GROUP BY user
HAVING n_ready_jobs + n_running_jobs > 0;
''',
            (self.pool.name,),
            "compute_fair_share",
        )

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

    async def schedule_loop_body(self):
        if self.app['frozen']:
            log.info(f'not scheduling any jobs for {self.pool}; batch is frozen')
            return True

        log.info(f'schedule {self.pool}: starting')
        start = time_msecs()
        SCHEDULING_LOOP_RUNS.inc()
        n_scheduled = 0

        user_resources = await self.compute_fair_share()

        total = sum(resources['allocated_cores_mcpu'] for resources in user_resources.values())
        if not total:
            log.info(f'schedule {self.pool}: no allocated cores')
            should_wait = True
            return should_wait
        user_share = {
            user: max(int(300 * resources['allocated_cores_mcpu'] / total + 0.5), 20)
            for user, resources in user_resources.items()
        }

        async def user_runnable_jobs(user, remaining):
            async for batch in self.db.select_and_fetchall(
                '''
SELECT batches.id, batches_cancelled.id IS NOT NULL AS cancelled, userdata, user, format_version
FROM batches
LEFT JOIN batches_cancelled
       ON batches.id = batches_cancelled.id
WHERE user = %s AND `state` = 'running';
''',
                (user,),
                "user_runnable_jobs__select_running_batches",
            ):
                async for record in self.db.select_and_fetchall(
                    '''
SELECT job_id, spec, cores_mcpu
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_inst_coll_cancelled)
WHERE batch_id = %s AND state = 'Ready' AND always_run = 1 AND inst_coll = %s
LIMIT %s;
''',
                    (batch['id'], self.pool.name, remaining.value),
                    "user_runnable_jobs__select_ready_always_run_jobs",
                ):
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
WHERE batch_id = %s AND state = 'Ready' AND always_run = 0 AND inst_coll = %s AND cancelled = 0
LIMIT %s;
''',
                        (batch['id'], self.pool.name, remaining.value),
                        "user_runnable_jobs__select_ready_jobs_batch_not_cancelled",
                    ):
                        record['batch_id'] = batch['id']
                        record['userdata'] = batch['userdata']
                        record['user'] = batch['user']
                        record['format_version'] = batch['format_version']
                        yield record

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

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
                    if random.random() > self.exceeded_shares_counter.rate():
                        self.exceeded_shares_counter.push(True)
                        self.scheduler_state_changed.set()
                        break
                    self.exceeded_shares_counter.push(False)

                instance = self.pool.get_instance(user, record['cores_mcpu'])
                if instance:
                    instance.adjust_free_cores_in_memory(-record['cores_mcpu'])
                    scheduled_cores_mcpu += record['cores_mcpu']
                    n_scheduled += 1

                    async def schedule_with_error_handling(app, record, id, instance):
                        try:
                            await schedule_job(app, record, instance)
                        except Exception:
                            log.info(f'scheduling job {id} on {instance} for {self.pool}', exc_info=True)

                    await waitable_pool.call(schedule_with_error_handling, self.app, record, id, instance)

                remaining.value -= 1
                if remaining.value <= 0:
                    should_wait = False
                    break

        await waitable_pool.wait()

        end = time_msecs()
        log.info(f'schedule: attempted to schedule {n_scheduled} jobs in {end - start}ms for {self.pool}')

        return should_wait
