import asyncio
import logging
import random
from collections import defaultdict
from typing import List, Optional, Tuple

import prometheus_client as pc
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

from ...batch_configuration import STANDING_WORKER_MAX_IDLE_TIME_MSECS
from ...batch_format_version import BatchFormatVersion
from ...inst_coll_config import PoolConfig
from ...utils import ExceededSharesCounter, json_to_value
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
AUTOSCALER_HEAD_JOB_QUEUE_N_INSTANCES = pc.Gauge(
    'autoscaler_head_job_queue_n_instances',
    'Number of instances estimated per control loop calculated from the head of the job queue',
    ['pool_name', 'region'],
)


AUTOSCALER_LOOP_PERIOD_SECONDS = 15
MAX_INSTANCES_PER_AUTOSCALER_LOOP = 10  # n * 16 cores / 15s = excess_scheduling_rate/s = 10/s => n ~= 10
JOB_QUEUE_SCHEDULING_WINDOW_SECONDS = 150  # 2.5 minutes is approximately worker start up time


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
        self.enable_standing_worker = config.enable_standing_worker
        self.standing_worker_cores = config.standing_worker_cores
        self.boot_disk_size_gb = config.boot_disk_size_gb
        self.data_disk_size_gb = config.data_disk_size_gb
        self.data_disk_size_standing_gb = config.data_disk_size_standing_gb
        self.preemptible = config.preemptible

        # FIXME: CI needs to submit jobs to a specific region
        # instead of batch making this decicion on behalf of CI
        self._ci_region = self.inst_coll_manager._default_region

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

    def get_instance(self, user: str, cores_mcpu: int, regions: List[str]):
        i = self.healthy_instances_by_free_cores.bisect_key_left(cores_mcpu)
        while i < len(self.healthy_instances_by_free_cores):
            instance = self.healthy_instances_by_free_cores[i]
            assert cores_mcpu <= instance.free_cores_mcpu
            if user == 'ci' and instance.region == self._ci_region:
                return instance
            if user != 'ci' and instance.region in regions:
                return instance
            i += 1
        return None

    async def create_instance(
        self,
        cores: int,
        data_disk_size_gb: int,
        regions: List[str],
        max_idle_time_msecs: Optional[int] = None,
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
    ):
        n_live_instances = self.n_instances_by_state['pending'] + self.n_instances_by_state['active']
        n_instances = sum(count for count in self.n_instances_by_state.values())

        live_free_cores_mcpu = sum(self.live_free_cores_mcpu_by_region[region] for region in regions)

        instances_needed = (ready_cores_mcpu - live_free_cores_mcpu + (self.worker_cores * 1000) - 1) // (
            self.worker_cores * 1000
        )
        instances_needed = min(
            instances_needed,
            self.max_live_instances - n_live_instances,
            self.max_instances - n_instances,
            # 20 queries/s; our GCE long-run quota
            300,
            MAX_INSTANCES_PER_AUTOSCALER_LOOP,
        )
        return max(0, instances_needed)

    async def _create_instances(self, n_instances: int, regions: List[str]):
        if n_instances > 0:
            log.info(f'creating {n_instances} new instances')
            # parallelism will be bounded by thread pool
            await asyncio.gather(
                *[
                    self.create_instance(
                        cores=self.worker_cores,
                        data_disk_size_gb=self.data_disk_size_gb,
                        regions=regions,
                    )
                    for _ in range(n_instances)
                ]
            )

    async def create_instances_from_ready_cores(self, ready_cores_mcpu: int, regions: List[str]):
        instances_needed = self.compute_n_instances_needed(
            ready_cores_mcpu,
            regions,
        )

        await self._create_instances(instances_needed, regions)

    async def regions_to_ready_cores_mcpu_from_estimated_job_queue(self) -> List[Tuple[List[str], int]]:
        autoscaler_runs_per_minute = 60 / AUTOSCALER_LOOP_PERIOD_SECONDS
        max_new_instances_in_two_and_a_half_minutes = int(
            2.5 * MAX_INSTANCES_PER_AUTOSCALER_LOOP * autoscaler_runs_per_minute
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

        for user, share in user_share.items():
            user_job_query = f'''
(
  SELECT user, jobs.batch_id, jobs.job_id, cores_mcpu,
    JSON_ARRAYAGG(region) AS regions,
    COUNT(region) AS n_regions,
    ROW_NUMBER() OVER (ORDER BY batch_id ASC, job_id ASC) DIV {share} AS scheduling_iteration
  FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
  LEFT JOIN batches ON jobs.batch_id = batches.id
  LEFT JOIN batches_cancelled ON batches.id = batches_cancelled.id
  LEFT JOIN job_regions ON jobs.batch_id = job_regions.batch_id AND jobs.job_id = job_regions.job_id
  LEFT JOIN region_ids ON job_regions.region_id = region_ids.region_id
  WHERE user = %s AND batches.`state` = 'running' AND jobs.state = 'Ready' AND (always_run = 1 OR (always_run = 0 AND batches_cancelled.id IS NULL)) AND inst_coll = %s
  GROUP BY jobs.batch_id, jobs.job_id
  ORDER BY jobs.batch_id ASC, jobs.job_id ASC
  LIMIT {share * JOB_QUEUE_SCHEDULING_WINDOW_SECONDS}
)
'''

            jobs_query.append(user_job_query)
            jobs_query_args += [user, self.name]

        result = self.db.select_and_fetchall(
            # We use the string representation of regions JSON_UNQUOTE for order by instead of the json array
            f'''
WITH ready_jobs AS (
    {" UNION ".join(jobs_query)}
)
SELECT regions, CAST(COALESCE(SUM(cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu
FROM (
  SELECT regions, cores_mcpu,
    CAST(ROW_NUMBER() OVER (ORDER BY scheduling_iteration, user, -n_regions DESC, JSON_UNQUOTE(regions), ready_jobs.batch_id, ready_jobs.job_id) AS SIGNED) AS row_num_overall,
    CAST(ROW_NUMBER() OVER (PARTITION BY JSON_UNQUOTE(regions) ORDER BY scheduling_iteration, user, -n_regions DESC, JSON_UNQUOTE(regions), ready_jobs.batch_id, ready_jobs.job_id) AS SIGNED) AS row_num_by_regions
  FROM ready_jobs
) AS ordered_jobs
GROUP BY regions, row_num_overall - row_num_by_regions
HAVING ready_cores_mcpu > 0
ORDER BY row_num_overall
LIMIT {MAX_INSTANCES_PER_AUTOSCALER_LOOP * self.worker_cores};
''',
            jobs_query_args,
            query_name='get_job_queue_head',
        )

        def extract_regions(regions_str):
            regions = json_to_value(regions_str)
            # Left join with JSON_ARRAYAGG returns [null]; treat this case as no regions selected
            if regions == [None]:
                return self.all_supported_regions
            return regions

        return [(extract_regions(record['regions']), record['ready_cores_mcpu']) async for record in result]

    async def ready_cores_mcpu_per_user(self):
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

        return ready_cores_mcpu_per_user

    async def create_instances(self):
        if self.app['frozen']:
            log.info(f'not creating instances for {self}; batch is frozen')
            return

        AUTOSCALER_LOOP_RUNS.labels(pool_name=self.name).inc()

        ready_cores_mcpu_per_user = await self.ready_cores_mcpu_per_user()

        free_cores_mcpu = sum([worker.free_cores_mcpu for worker in self.healthy_instances_by_free_cores])
        free_cores = free_cores_mcpu / 1000

        head_job_queue_regions_ready_cores_mcpu_ordered = (
            await self.regions_to_ready_cores_mcpu_from_estimated_job_queue()
        )

        head_job_queue_n_instances = defaultdict(int)
        head_job_queue_ready_cores_mcpu = defaultdict(int)

        if head_job_queue_regions_ready_cores_mcpu_ordered and free_cores < 500:
            for regions, ready_cores_mcpu in head_job_queue_regions_ready_cores_mcpu_ordered:
                n_regions = len(regions)
                for region in regions:
                    n_instances = self.compute_n_instances_needed(ready_cores_mcpu, regions)
                    head_job_queue_ready_cores_mcpu[region] += ready_cores_mcpu / n_regions
                    head_job_queue_n_instances[region] += n_instances / n_regions

                await self.create_instances_from_ready_cores(ready_cores_mcpu, regions=regions)

        ci_ready_cores_mcpu = ready_cores_mcpu_per_user.get('ci', 0)
        if ci_ready_cores_mcpu > 0 and self.live_free_cores_mcpu_by_region[self._ci_region] == 0:
            await self.create_instances_from_ready_cores(ci_ready_cores_mcpu, regions=[self._ci_region])

        n_live_instances = self.n_instances_by_state['pending'] + self.n_instances_by_state['active']
        if self.enable_standing_worker and n_live_instances == 0 and self.max_instances > 0:
            await self.create_instance(
                cores=self.standing_worker_cores,
                data_disk_size_gb=self.data_disk_size_standing_gb,
                regions=self.all_supported_regions,
                max_idle_time_msecs=STANDING_WORKER_MAX_IDLE_TIME_MSECS,
            )

        log.info(
            f'{self} n_instances {self.n_instances} {self.n_instances_by_state}'
            f' free_cores {free_cores} live_free_cores {self.live_free_cores_mcpu / 1000}'
            f' full_job_queue_ready_cores {sum(ready_cores_mcpu_per_user.values()) / 1000}'
            f' head_job_queue_ready_cores {sum(head_job_queue_ready_cores_mcpu.values()) / 1000}'
            f' head_job_queue_n_instances {sum(head_job_queue_n_instances.values())}'
        )

    async def control_loop(self):
        await periodically_call(AUTOSCALER_LOOP_PERIOD_SECONDS, self.create_instances)

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
        return await self._compute_fair_share(free_cores_mcpu)

    async def _compute_fair_share(self, free_cores_mcpu):
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

        start = time_msecs()
        SCHEDULING_LOOP_RUNS.labels(pool_name=self.pool.name).inc()

        n_scheduled = 0

        scheduled_jobs_per_region = defaultdict(int)
        scheduled_cores_mcpu_per_region = defaultdict(int)
        unscheduled_jobs_per_region = defaultdict(int)
        unscheduled_cores_mcpu_per_region = defaultdict(int)

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

        async def user_runnable_jobs(user, share):
            async for record in self.db.select_and_fetchall(
                f'''
SELECT user, jobs.batch_id, jobs.job_id, cores_mcpu, spec, userdata, format_version,
  JSON_ARRAYAGG(region) AS regions,
  COUNT(region) AS n_regions,
  ROW_NUMBER() OVER (ORDER BY batch_id ASC, job_id ASC) DIV {share} AS scheduling_iteration
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_cancelled)
LEFT JOIN batches ON jobs.batch_id = batches.id
LEFT JOIN batches_cancelled ON batches.id = batches_cancelled.id
LEFT JOIN job_regions ON jobs.batch_id = job_regions.batch_id AND jobs.job_id = job_regions.job_id
LEFT JOIN region_ids ON job_regions.region_id = region_ids.region_id
WHERE user = %s AND batches.`state` = 'running' AND jobs.state = 'Ready' AND (always_run = 1 OR (always_run = 0 AND batches_cancelled.id IS NULL)) AND inst_coll = %s
GROUP BY jobs.batch_id, jobs.job_id
ORDER BY scheduling_iteration, user, -n_regions DESC, JSON_UNQUOTE(regions), jobs.batch_id, jobs.job_id
LIMIT {share * JOB_QUEUE_SCHEDULING_WINDOW_SECONDS};
''',
                (user, self.pool.name),
                'select_user_ready_jobs',
            ):
                yield record

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        should_wait = True
        for user, resources in user_resources.items():
            allocated_cores_mcpu = resources['allocated_cores_mcpu']
            if allocated_cores_mcpu == 0:
                continue

            scheduled_cores_mcpu = 0
            share = user_share[user]

            async for record in user_runnable_jobs(user, share):
                attempt_id = secret_alnum_string(6)
                record['attempt_id'] = attempt_id

                regions = json_to_value(record['regions'])
                supported_regions = self.pool.all_supported_regions
                # Left join with JSON_ARRAYAGG returns [null]; treat this case as no regions selected
                if regions == [None]:
                    regions = supported_regions
                if len(set(regions).intersection(supported_regions)) == 0:
                    await mark_job_errored(
                        self.app,
                        record['batch_id'],
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

                instance = self.pool.get_instance(user, record['cores_mcpu'], regions)
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

                share -= 1
                if share <= 0:
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
