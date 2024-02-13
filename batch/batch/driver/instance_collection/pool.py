import asyncio
import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import prometheus_client as pc
import sortedcontainers

from gear import Database
from hailtop import aiotools
from hailtop.utils import (
    AsyncWorkerPool,
    Notice,
    WaitableSharedPool,
    periodically_call_with_dynamic_sleep,
    retry_long_running,
    run_if_changed,
    secret_alnum_string,
    time_msecs,
)

from ...batch_format_version import BatchFormatVersion
from ...globals import INSTANCE_VERSION
from ...inst_coll_config import PoolConfig
from ...utils import ExceededSharesCounter, regions_bits_rep_to_regions
from ..instance import Instance
from ..job import mark_job_errored, schedule_job
from ..resource_manager import CloudResourceManager
from .base import InstanceCollection, InstanceCollectionManager

log = logging.getLogger('pool')

SCHEDULING_LOOP_RUNS = pc.Counter(
    'scheduling_loop_runs',
    'Number of scheduling loop executions per pool',
    ['pool_name'],
)

AUTOSCALER_LOOP_RUNS = pc.Counter(
    'autoscaler_loop_runs',
    'Number of control loop executions per pool',
    ['pool_name'],
)

SCHEDULING_LOOP_CORES = pc.Gauge(
    'scheduling_loop_cores',
    'Number of cores scheduled or unscheduled in one scheduling execution loop',
    ['pool_name', 'region', 'scheduled'],
)
SCHEDULING_LOOP_JOBS = pc.Gauge(
    'scheduling_loop_jobs',
    'Number of jobs scheduled or unscheduled in one scheduling execution loop',
    ['pool_name', 'region', 'scheduled'],
)

AUTOSCALER_HEAD_JOB_QUEUE_READY_CORES = pc.Gauge(
    'autoscaler_head_job_queue_ready_cores',
    'Number of ready cores per control loop execution calculated from the head of the job queue',
    ['pool_name', 'region'],
)


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
            """
SELECT instances.*, instances_free_cores_mcpu.free_cores_mcpu
FROM instances
INNER JOIN instances_free_cores_mcpu
ON instances.name = instances_free_cores_mcpu.name
WHERE removed = 0 AND inst_coll = %s;
""",
            (pool.name,),
        ):
            pool.add_instance(Instance.from_record(app, pool, record))

        task_manager.ensure_future(pool.control_loop())
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
        self.standing_worker_cores = config.standing_worker_cores
        self.boot_disk_size_gb = config.boot_disk_size_gb
        self.data_disk_size_gb = config.data_disk_size_gb
        self.data_disk_size_standing_gb = config.data_disk_size_standing_gb
        self.preemptible = config.preemptible
        self.max_new_instances_per_autoscaler_loop = config.max_new_instances_per_autoscaler_loop
        self.autoscaler_loop_period_secs = config.autoscaler_loop_period_secs
        self.standing_worker_max_idle_time_secs = config.standing_worker_max_idle_time_secs
        self.worker_max_idle_time_secs = config.worker_max_idle_time_secs
        self.job_queue_scheduling_window_secs = config.job_queue_scheduling_window_secs
        self.min_instances = config.min_instances

        self.all_supported_regions = self.inst_coll_manager.regions

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
            'standing_worker_cores': self.standing_worker_cores,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'max_live_instances': self.max_live_instances,
            'preemptible': self.preemptible,
            'max_new_instances_per_autoscaler_loop': self.max_new_instances_per_autoscaler_loop,
            'autoscaler_loop_period_secs': self.autoscaler_loop_period_secs,
            'standing_worker_max_idle_time_secs': self.standing_worker_max_idle_time_secs,
            'worker_max_idle_time_secs': self.worker_max_idle_time_secs,
            'job_queue_scheduling_window_secs': self.job_queue_scheduling_window_secs,
        }

    def configure(self, pool_config: PoolConfig):
        assert self.name == pool_config.name
        assert self.cloud == pool_config.cloud
        assert self.worker_type == pool_config.worker_type

        self.worker_cores = pool_config.worker_cores
        self.worker_local_ssd_data_disk = pool_config.worker_local_ssd_data_disk
        self.worker_external_ssd_data_disk_size_gb = pool_config.worker_external_ssd_data_disk_size_gb
        self.standing_worker_cores = pool_config.standing_worker_cores
        self.boot_disk_size_gb = pool_config.boot_disk_size_gb
        self.data_disk_size_gb = pool_config.data_disk_size_gb
        self.data_disk_size_standing_gb = pool_config.data_disk_size_standing_gb
        self.min_instances = pool_config.min_instances
        self.max_instances = pool_config.max_instances
        self.max_live_instances = pool_config.max_live_instances
        self.preemptible = pool_config.preemptible
        self.max_new_instances_per_autoscaler_loop = pool_config.max_new_instances_per_autoscaler_loop
        self.autoscaler_loop_period_secs = pool_config.autoscaler_loop_period_secs
        self.standing_worker_max_idle_time_secs = pool_config.standing_worker_max_idle_time_secs
        self.worker_max_idle_time_secs = pool_config.worker_max_idle_time_secs
        self.job_queue_scheduling_window_secs = pool_config.job_queue_scheduling_window_secs

    def adjust_for_remove_instance(self, instance):
        super().adjust_for_remove_instance(instance)
        if instance in self.healthy_instances_by_free_cores:
            self.healthy_instances_by_free_cores.remove(instance)

    def adjust_for_add_instance(self, instance):
        super().adjust_for_add_instance(instance)
        if instance.state == 'active' and instance.failed_request_count <= 1:
            self.healthy_instances_by_free_cores.add(instance)

    def get_instance(self, cores_mcpu: int, regions: List[str]) -> Optional[Instance]:
        i = self.healthy_instances_by_free_cores.bisect_key_left(cores_mcpu)
        while i < len(self.healthy_instances_by_free_cores):
            instance: Instance = self.healthy_instances_by_free_cores[i]  # type: ignore
            assert cores_mcpu <= instance.free_cores_mcpu
            if instance.region in regions and instance.version == INSTANCE_VERSION:
                return instance
            i += 1
        return None

    async def create_instance(
        self,
        cores: int,
        data_disk_size_gb: int,
        regions: List[str],
        max_idle_time_msecs: int,
    ):
        machine_type = self.resource_manager.machine_type(cores, self.worker_type, self.worker_local_ssd_data_disk)
        _, _ = await self._create_instance(
            app=self.app,
            cores=cores,
            machine_type=machine_type,
            job_private=False,
            regions=regions,
            preemptible=self.preemptible,
            max_idle_time_msecs=max_idle_time_msecs,
            local_ssd_data_disk=self.worker_local_ssd_data_disk,
            data_disk_size_gb=data_disk_size_gb,
            boot_disk_size_gb=self.boot_disk_size_gb,
        )

    def compute_n_instances_needed(
        self,
        ready_cores_mcpu: int,
        regions: List[str],
        remaining_max_new_instances_per_autoscaler_loop: int,
    ):
        pool_stats = self.current_worker_version_stats
        n_live_instances = pool_stats.n_instances_by_state['pending'] + pool_stats.n_instances_by_state['active']
        live_free_cores_mcpu = sum(pool_stats.live_free_cores_mcpu_by_region[region] for region in regions)

        instances_needed = (ready_cores_mcpu - live_free_cores_mcpu + (self.worker_cores * 1000) - 1) // (
            self.worker_cores * 1000
        )
        instances_needed = min(
            instances_needed,
            self.max_live_instances - n_live_instances,
            self.max_instances - self.n_instances,
            # 20 queries/s; our GCE long-run quota
            300,
            remaining_max_new_instances_per_autoscaler_loop,
        )
        return max(0, instances_needed)

    async def _create_instances(
        self,
        n_instances: int,
        cores: int,
        data_disk_size_gb: int,
        regions: List[str],
        max_idle_time_msecs: int,
    ):
        if n_instances > 0:
            log.info(f'creating {n_instances} new instances')
            # parallelism will be bounded by thread pool
            await asyncio.gather(*[
                self.create_instance(
                    cores=cores,
                    data_disk_size_gb=data_disk_size_gb,
                    regions=regions,
                    max_idle_time_msecs=max_idle_time_msecs,
                )
                for _ in range(n_instances)
            ])

    async def create_instances_from_ready_cores(
        self, ready_cores_mcpu: int, regions: List[str], remaining_max_new_instances_per_autoscaler_loop: int
    ):
        instances_needed = self.compute_n_instances_needed(
            ready_cores_mcpu,
            regions,
            remaining_max_new_instances_per_autoscaler_loop,
        )

        await self._create_instances(
            instances_needed,
            self.worker_cores,
            self.data_disk_size_gb,
            regions,
            max_idle_time_msecs=self.worker_max_idle_time_secs * 1000,
        )
        return instances_needed

    async def regions_to_ready_cores_mcpu_from_estimated_job_queue(self) -> List[Tuple[List[str], int]]:
        autoscaler_runs_per_minute = 60 / self.autoscaler_loop_period_secs
        max_new_instances_in_two_and_a_half_minutes = int(
            2.5 * self.max_new_instances_per_autoscaler_loop * autoscaler_runs_per_minute
        )
        max_possible_future_cores = self.worker_cores * max_new_instances_in_two_and_a_half_minutes

        user_resources = await self.scheduler._compute_fair_share(max_possible_future_cores)

        total = sum(resources['allocated_cores_mcpu'] for resources in user_resources.values())

        if total == 0:
            return []

        # estimate of number of jobs scheduled per scheduling loop approximately every second
        user_share = {
            user: max(int(300 * resources['allocated_cores_mcpu'] / total + 0.5), 20)
            for user, resources in user_resources.items()
        }

        jobs_query = []
        jobs_query_args = []

        for user_idx, (user, share) in enumerate(user_share.items(), start=1):
            user_job_query = f"""
(
  SELECT scheduling_iteration, user_idx, n_regions, regions_bits_rep, CAST(COALESCE(SUM(cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu
  FROM (
    SELECT {user_idx} AS user_idx, batch_id, job_id, cores_mcpu, always_run, n_regions, regions_bits_rep,
      ROW_NUMBER() OVER (ORDER BY batch_id, job_group_id, always_run DESC, -n_regions DESC, regions_bits_rep, job_id ASC) DIV {share} AS scheduling_iteration
    FROM (
      (
        SELECT jobs.batch_id, jobs.job_id, jobs.job_group_id, cores_mcpu, always_run, n_regions, regions_bits_rep
        FROM jobs FORCE INDEX(jobs_batch_id_ic_state_ar_n_regions_bits_rep_job_group_id)
        LEFT JOIN batches ON jobs.batch_id = batches.id
        WHERE user = %s AND batches.`state` = 'running' AND jobs.state = 'Ready' AND always_run AND inst_coll = %s
        ORDER BY jobs.batch_id ASC, jobs.job_group_id ASC, jobs.job_id ASC
        LIMIT {share * self.job_queue_scheduling_window_secs}
      )
      UNION
      (
        SELECT jobs.batch_id, jobs.job_id, jobs.job_group_id, cores_mcpu, always_run, n_regions, regions_bits_rep
        FROM jobs FORCE INDEX(jobs_batch_id_ic_state_ar_n_regions_bits_rep_job_group_id)
        LEFT JOIN batches ON jobs.batch_id = batches.id
        WHERE user = %s AND batches.`state` = 'running' AND jobs.state = 'Ready' AND NOT always_run AND NOT is_job_group_cancelled(jobs.batch_id, jobs.job_group_id) AND inst_coll = %s
        ORDER BY jobs.batch_id ASC, jobs.job_group_id ASC, jobs.job_id ASC
        LIMIT {share * self.job_queue_scheduling_window_secs}
      )
    ) AS t1
    ORDER BY batch_id, job_group_id, always_run DESC, -n_regions DESC, regions_bits_rep, job_id ASC
    LIMIT {share * self.job_queue_scheduling_window_secs}
  ) AS t2
  GROUP BY scheduling_iteration, user_idx, regions_bits_rep, n_regions
  HAVING ready_cores_mcpu > 0
  LIMIT {self.max_new_instances_per_autoscaler_loop * self.worker_cores}
)
"""

            jobs_query.append(user_job_query)
            jobs_query_args += [user, self.name, user, self.name]

        result = self.db.select_and_fetchall(
            f"""
WITH ready_cores_by_scheduling_iteration_regions AS (
    {" UNION ".join(jobs_query)}
)
SELECT regions_bits_rep, ready_cores_mcpu
FROM ready_cores_by_scheduling_iteration_regions
ORDER BY scheduling_iteration, user_idx, -n_regions DESC, regions_bits_rep
LIMIT {self.max_new_instances_per_autoscaler_loop * self.worker_cores};
""",
            jobs_query_args,
            query_name='get_job_queue_head',
        )

        def extract_regions(regions_bits_rep: int):
            if regions_bits_rep is None:
                return self.all_supported_regions
            return regions_bits_rep_to_regions(regions_bits_rep, self.app['regions'])

        return [(extract_regions(record['regions_bits_rep']), record['ready_cores_mcpu']) async for record in result]

    async def ready_cores_mcpu_per_user(self):
        ready_cores_mcpu_per_user = self.db.select_and_fetchall(
            """
SELECT user,
  CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu
FROM user_inst_coll_resources
WHERE inst_coll = %s
GROUP BY user;
""",
            (self.name,),
        )

        if ready_cores_mcpu_per_user is None:
            ready_cores_mcpu_per_user = {}
        else:
            ready_cores_mcpu_per_user = {r['user']: r['ready_cores_mcpu'] async for r in ready_cores_mcpu_per_user}

        return ready_cores_mcpu_per_user

    async def create_instances(self):
        if self.app['frozen']:
            log.info(f'not creating instances for {self}; batch is frozen')
            return

        AUTOSCALER_LOOP_RUNS.labels(pool_name=self.name).inc()

        ready_cores_mcpu_per_user = await self.ready_cores_mcpu_per_user()

        free_cores_mcpu = sum(worker.free_cores_mcpu for worker in self.healthy_instances_by_free_cores)
        free_cores = free_cores_mcpu / 1000

        head_job_queue_regions_ready_cores_mcpu_ordered = (
            await self.regions_to_ready_cores_mcpu_from_estimated_job_queue()
        )

        head_job_queue_ready_cores_mcpu: Dict[str, float] = defaultdict(float)

        remaining_instances_per_autoscaler_loop = self.max_new_instances_per_autoscaler_loop
        if head_job_queue_regions_ready_cores_mcpu_ordered and free_cores < 500:
            for regions, ready_cores_mcpu in head_job_queue_regions_ready_cores_mcpu_ordered:
                n_instances_created = await self.create_instances_from_ready_cores(
                    ready_cores_mcpu,
                    regions,
                    remaining_instances_per_autoscaler_loop,
                )

                n_regions = len(regions)
                for region in regions:
                    head_job_queue_ready_cores_mcpu[region] += ready_cores_mcpu / n_regions

                remaining_instances_per_autoscaler_loop -= n_instances_created
                if remaining_instances_per_autoscaler_loop <= 0:
                    break

        pool_stats = self.current_worker_version_stats
        n_live_instances = pool_stats.n_instances_by_state['pending'] + pool_stats.n_instances_by_state['active']

        capacity_for_live_instances = max(0, self.max_live_instances - n_live_instances)
        capacity_for_any_instances = max(0, self.max_instances - self.n_instances)

        capacity_for_new_instances = min(capacity_for_live_instances, capacity_for_any_instances)
        n_instances_to_meet_shortfall = max(0, self.min_instances - self.n_instances)

        if capacity_for_new_instances > 0 and n_instances_to_meet_shortfall > 0:
            n_standing_instances_to_provision = min(
                capacity_for_new_instances,
                n_instances_to_meet_shortfall,
                remaining_instances_per_autoscaler_loop,
                300,  # 20 queries/s; our GCE long-run quota
            )
            if n_standing_instances_to_provision > 0:
                await self._create_instances(
                    n_instances=n_standing_instances_to_provision,
                    cores=self.standing_worker_cores,
                    data_disk_size_gb=self.data_disk_size_standing_gb,
                    regions=self.all_supported_regions,
                    max_idle_time_msecs=self.standing_worker_max_idle_time_secs * 1000,
                )

        log.info(
            f'{self} n_instances {self.n_instances} {pool_stats.n_instances_by_state}'
            f' active_schedulable_free_cores {pool_stats.active_schedulable_free_cores_mcpu / 1000}'
            f' full_job_queue_ready_cores {sum(ready_cores_mcpu_per_user.values()) / 1000}'
            f' head_job_queue_ready_cores {sum(head_job_queue_ready_cores_mcpu.values()) / 1000}'
        )

        for region in self.all_supported_regions:
            ready_cores_mcpu = int(head_job_queue_ready_cores_mcpu.get(region, 0.0))
            AUTOSCALER_HEAD_JOB_QUEUE_READY_CORES.labels(pool_name=self.name, region=region).set(ready_cores_mcpu)

    async def control_loop(self):
        await periodically_call_with_dynamic_sleep(lambda: self.autoscaler_loop_period_secs, self.create_instances)

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
        free_cores_mcpu = sum(worker.free_cores_mcpu for worker in self.pool.healthy_instances_by_free_cores)
        return await self._compute_fair_share(free_cores_mcpu)

    async def _compute_fair_share(self, free_cores_mcpu):
        user_running_cores_mcpu: Dict[str, int] = {}
        user_total_cores_mcpu: Dict[str, int] = {}
        result = {}

        pending_users_by_running_cores = sortedcontainers.SortedSet(key=lambda user: user_running_cores_mcpu[user])
        allocating_users_by_total_cores = sortedcontainers.SortedSet(key=lambda user: user_total_cores_mcpu[user])

        records = self.db.execute_and_fetchall(
            """
SELECT user,
  CAST(COALESCE(SUM(n_ready_jobs), 0) AS SIGNED) AS n_ready_jobs,
  CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu,
  CAST(COALESCE(SUM(n_running_jobs), 0) AS SIGNED) AS n_running_jobs,
  CAST(COALESCE(SUM(running_cores_mcpu), 0) AS SIGNED) AS running_cores_mcpu
FROM user_inst_coll_resources
WHERE inst_coll = %s
GROUP BY user
HAVING n_ready_jobs + n_running_jobs > 0;
""",
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
                lowest_running_user: str = pending_users_by_running_cores[0]  # type: ignore
                lowest_running = user_running_cores_mcpu[lowest_running_user]
                if lowest_running == mark:
                    pending_users_by_running_cores.remove(lowest_running_user)
                    allocating_users_by_total_cores.add(lowest_running_user)
                    continue

            if allocating_users_by_total_cores:
                lowest_total_user: str = allocating_users_by_total_cores[0]  # type: ignore
                lowest_total = user_total_cores_mcpu[lowest_total_user]
                if lowest_total == mark:
                    allocating_users_by_total_cores.remove(lowest_total_user)
                    allocate_cores(lowest_total_user, mark)
                    continue

            allocation = min(c for c in [lowest_running, lowest_total] if c is not None)

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

        result = dict(sorted(result.items(), key=lambda item: item[1]['allocated_cores_mcpu'], reverse=True))
        return result

    async def schedule_loop_body(self):
        if self.app['frozen']:
            log.info(f'not scheduling any jobs for {self.pool}; batch is frozen')
            return True

        start = time_msecs()
        SCHEDULING_LOOP_RUNS.labels(pool_name=self.pool.name).inc()

        n_scheduled = 0

        scheduled_jobs_per_region: Dict[str, int] = defaultdict(int)
        scheduled_cores_mcpu_per_region: Dict[str, int] = defaultdict(int)
        unscheduled_jobs_per_region: Dict[str, float] = defaultdict(float)
        unscheduled_cores_mcpu_per_region: Dict[str, int] = defaultdict(int)

        user_resources = await self.compute_fair_share()

        total = sum(resources['allocated_cores_mcpu'] for resources in user_resources.values())
        if not total:
            should_wait = True
            for region in self.pool.all_supported_regions:
                SCHEDULING_LOOP_CORES.labels(pool_name=self.pool.name, region=region, scheduled=True).set(0)
                SCHEDULING_LOOP_CORES.labels(pool_name=self.pool.name, region=region, scheduled=False).set(0)
                SCHEDULING_LOOP_JOBS.labels(pool_name=self.pool.name, region=region, scheduled=True).set(0)
                SCHEDULING_LOOP_JOBS.labels(pool_name=self.pool.name, region=region, scheduled=False).set(0)
            return should_wait
        user_share = {
            user: max(int(300 * resources['allocated_cores_mcpu'] / total + 0.5), 20)
            for user, resources in user_resources.items()
        }

        async def user_runnable_jobs(user):
            async for job_group in self.db.select_and_fetchall(
                """
SELECT job_groups.batch_id, job_groups.job_group_id, is_job_group_cancelled(job_groups.batch_id, job_groups.job_group_id) AS cancelled,
  userdata, job_groups.user, format_version
FROM job_groups
LEFT JOIN batches ON job_groups.batch_id = batches.id
WHERE job_groups.user = %s AND job_groups.`state` = 'running'
ORDER BY job_groups.batch_id, job_groups.job_group_id;
""",
                (user,),
                "user_runnable_jobs__select_running_batches",
            ):
                async for record in self.db.select_and_fetchall(
                    """
SELECT jobs.job_id, spec, cores_mcpu, regions_bits_rep, time_ready, job_group_id
FROM jobs FORCE INDEX(jobs_batch_id_ic_state_ar_n_regions_bits_rep_job_group_id)
LEFT JOIN jobs_telemetry ON jobs.batch_id = jobs_telemetry.batch_id AND jobs.job_id = jobs_telemetry.job_id
WHERE jobs.batch_id = %s AND job_group_id = %s AND inst_coll = %s AND jobs.state = 'Ready' AND always_run = 1
ORDER BY jobs.batch_id, jobs.job_group_id, inst_coll, state, always_run, -n_regions DESC, regions_bits_rep, jobs.job_id
LIMIT 300;
""",
                    (job_group['batch_id'], job_group['job_group_id'], self.pool.name),
                    "user_runnable_jobs__select_ready_always_run_jobs",
                ):
                    record['batch_id'] = job_group['batch_id']
                    record['job_group_id'] = job_group['job_group_id']
                    record['userdata'] = job_group['userdata']
                    record['user'] = job_group['user']
                    record['format_version'] = job_group['format_version']
                    yield record
                if not job_group['cancelled']:
                    async for record in self.db.select_and_fetchall(
                        """
SELECT jobs.job_id, spec, cores_mcpu, regions_bits_rep, time_ready, job_group_id
FROM jobs FORCE INDEX(jobs_batch_id_ic_state_ar_n_regions_bits_rep_job_group_id)
LEFT JOIN jobs_telemetry ON jobs.batch_id = jobs_telemetry.batch_id AND jobs.job_id = jobs_telemetry.job_id
WHERE jobs.batch_id = %s AND job_group_id = %s AND inst_coll = %s AND jobs.state = 'Ready' AND always_run = 0 AND cancelled = 0
ORDER BY jobs.batch_id, jobs.job_group_id, inst_coll, state, always_run, -n_regions DESC, regions_bits_rep, jobs.job_id
LIMIT 300;
""",
                        (job_group['batch_id'], job_group['job_group_id'], self.pool.name),
                        "user_runnable_jobs__select_ready_jobs_batch_not_cancelled",
                    ):
                        record['batch_id'] = job_group['batch_id']
                        record['job_group_id'] = job_group['job_group_id']
                        record['userdata'] = job_group['userdata']
                        record['user'] = job_group['user']
                        record['format_version'] = job_group['format_version']
                        yield record

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        should_wait = True
        for user, resources in user_resources.items():
            allocated_cores_mcpu = resources['allocated_cores_mcpu']
            if allocated_cores_mcpu == 0:
                continue

            scheduled_cores_mcpu = 0
            share = user_share[user]

            async for record in user_runnable_jobs(user):
                attempt_id = secret_alnum_string(6)
                record['attempt_id'] = attempt_id

                supported_regions = self.pool.all_supported_regions
                regions_bits_rep = record['regions_bits_rep']

                if regions_bits_rep is None:
                    regions = supported_regions
                else:
                    regions = regions_bits_rep_to_regions(regions_bits_rep, self.app['regions'])

                if len(set(regions).intersection(supported_regions)) == 0:
                    await mark_job_errored(
                        self.app,
                        record['batch_id'],
                        record['job_group_id'],
                        record['job_id'],
                        attempt_id,
                        record['user'],
                        BatchFormatVersion(record['format_version']),
                        f'no regions given in {regions} are supported. choose from a region in {supported_regions}',
                    )
                    continue

                if scheduled_cores_mcpu + record['cores_mcpu'] > allocated_cores_mcpu:
                    if random.random() > self.exceeded_shares_counter.rate():
                        self.exceeded_shares_counter.push(True)
                        self.scheduler_state_changed.set()
                        break
                    self.exceeded_shares_counter.push(False)

                instance = self.pool.get_instance(record['cores_mcpu'], regions)
                if instance:
                    instance.adjust_free_cores_in_memory(-record['cores_mcpu'])
                    scheduled_cores_mcpu += record['cores_mcpu']

                    scheduled_cores_mcpu_per_region[instance.region] += record['cores_mcpu']
                    scheduled_jobs_per_region[instance.region] += 1

                    n_scheduled += 1

                    async def schedule_with_error_handling(app, record, instance):
                        try:
                            await schedule_job(app, record, instance)
                        except Exception:
                            if instance.state == 'active':
                                instance.adjust_free_cores_in_memory(record['cores_mcpu'])

                    await waitable_pool.call(schedule_with_error_handling, self.app, record, instance)
                else:
                    n_regions = len(regions)
                    for region in regions:
                        unscheduled_cores_mcpu_per_region[region] += record['cores_mcpu'] / n_regions
                        unscheduled_jobs_per_region[region] += 1 / n_regions

                if n_scheduled >= share:
                    should_wait = False
                    break

        await waitable_pool.wait()

        end = time_msecs()

        if n_scheduled > 0:
            log.info(f'schedule: attempted to schedule {n_scheduled} jobs in {end - start}ms for {self.pool}')

        for region in self.pool.inst_coll_manager.regions:
            n_cores_mcpu_scheduled = scheduled_cores_mcpu_per_region.get(region, 0)
            n_cores_mcpu_unscheduled = unscheduled_cores_mcpu_per_region.get(region, 0)

            n_jobs_scheduled = scheduled_jobs_per_region.get(region, 0)
            n_jobs_unscheduled = unscheduled_jobs_per_region.get(region, 0)

            SCHEDULING_LOOP_CORES.labels(pool_name=self.pool.name, region=region, scheduled=True).set(
                n_cores_mcpu_scheduled / 1000
            )
            SCHEDULING_LOOP_CORES.labels(pool_name=self.pool.name, region=region, scheduled=False).set(
                n_cores_mcpu_unscheduled / 1000
            )

            SCHEDULING_LOOP_JOBS.labels(pool_name=self.pool.name, region=region, scheduled=True).set(n_jobs_scheduled)
            SCHEDULING_LOOP_JOBS.labels(pool_name=self.pool.name, region=region, scheduled=False).set(
                n_jobs_unscheduled
            )

        return should_wait
