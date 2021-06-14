import sortedcontainers
import logging
import asyncio
import secrets
import random
import collections

from gear import Database, transaction
from hailtop import aiotools
from hailtop.utils import (
    secret_alnum_string,
    retry_long_running,
    run_if_changed,
    time_msecs,
    WaitableSharedPool,
    AsyncWorkerPool,
    Notice,
    periodically_call,
)

from ..batch_configuration import STANDING_WORKER_MAX_IDLE_TIME_MSECS, WORKER_MAX_IDLE_TIME_MSECS
from ..inst_coll_config import PoolConfig
from ..utils import (
    Box,
    ExceededSharesCounter,
    adjust_cores_for_memory_request,
    adjust_cores_for_packability,
    adjust_cores_for_storage_request,
)
from .create_instance import create_instance
from .instance import Instance
from .instance_collection import InstanceCollection
from .job import schedule_job

log = logging.getLogger('pool')


class Pool(InstanceCollection):
    def __init__(self, app, machine_name_prefix: str, config: PoolConfig):
        super().__init__(app, config.name, machine_name_prefix, is_pool=True)

        global_scheduler_state_changed: Notice = app['scheduler_state_changed']
        self.scheduler_state_changed = global_scheduler_state_changed.subscribe()
        self.scheduler = PoolScheduler(self.app, self)

        self.healthy_instances_by_free_cores = sortedcontainers.SortedSet(key=lambda instance: instance.free_cores_mcpu)

        self.worker_type = config.worker_type
        self.worker_cores = config.worker_cores
        self.worker_local_ssd_data_disk = config.worker_local_ssd_data_disk
        self.worker_pd_ssd_data_disk_size_gb = config.worker_pd_ssd_data_disk_size_gb
        self.enable_standing_worker = config.enable_standing_worker
        self.standing_worker_cores = config.standing_worker_cores
        self.boot_disk_size_gb = config.boot_disk_size_gb
        self.max_instances = config.max_instances
        self.max_live_instances = config.max_live_instances

    async def async_init(self):
        log.info(f'initializing {self}')

        await super().async_init()

        async for record in self.db.select_and_fetchall(
            'SELECT * FROM instances WHERE removed = 0 AND inst_coll = %s;', (self.name,)
        ):
            instance = Instance.from_record(self.app, self, record)
            self.add_instance(instance)

        self.task_manager.ensure_future(self.control_loop())

        await self.scheduler.async_init()

    def shutdown(self):
        try:
            super().shutdown()
        finally:
            self.scheduler.shutdown()

    def config(self):
        return {
            'name': self.name,
            'worker_type': self.worker_type,
            'worker_cores': self.worker_cores,
            'boot_disk_size_gb': self.boot_disk_size_gb,
            'worker_local_ssd_data_disk': self.worker_local_ssd_data_disk,
            'worker_pd_ssd_data_disk_size_gb': self.worker_pd_ssd_data_disk_size_gb,
            'enable_standing_worker': self.enable_standing_worker,
            'standing_worker_cores': self.standing_worker_cores,
            'max_instances': self.max_instances,
            'max_live_instances': self.max_live_instances,
        }

    async def configure(
        self,
        worker_cores,
        boot_disk_size_gb,
        worker_local_ssd_data_disk,
        worker_pd_ssd_data_disk_size_gb,
        enable_standing_worker,
        standing_worker_cores,
        max_instances,
        max_live_instances,
    ):
        @transaction(self.db)
        async def update(tx):
            await tx.just_execute(
                '''
UPDATE pools
SET worker_cores = %s, worker_local_ssd_data_disk = %s, worker_pd_ssd_data_disk_size_gb = %s,
  enable_standing_worker = %s, standing_worker_cores = %s
WHERE name = %s;
''',
                (
                    worker_cores,
                    worker_local_ssd_data_disk,
                    worker_pd_ssd_data_disk_size_gb,
                    enable_standing_worker,
                    standing_worker_cores,
                    self.name,
                ),
            )

            await tx.just_execute(
                '''
UPDATE inst_colls
SET boot_disk_size_gb = %s, max_instances = %s, max_live_instances = %s
WHERE name = %s;
''',
                (boot_disk_size_gb, max_instances, max_live_instances, self.name),
            )

        await update()  # pylint: disable=no-value-for-parameter

        self.worker_cores = worker_cores
        self.boot_disk_size_gb = boot_disk_size_gb
        self.worker_local_ssd_data_disk = worker_local_ssd_data_disk
        self.worker_pd_ssd_data_disk_size_gb = worker_pd_ssd_data_disk_size_gb
        self.enable_standing_worker = enable_standing_worker
        self.standing_worker_cores = standing_worker_cores
        self.max_instances = max_instances
        self.max_live_instances = max_live_instances

    def resources_to_cores_mcpu(self, cores_mcpu, memory_bytes, storage_bytes):
        cores_mcpu = adjust_cores_for_memory_request(cores_mcpu, memory_bytes, self.worker_type)
        cores_mcpu = adjust_cores_for_storage_request(
            cores_mcpu,
            storage_bytes,
            self.worker_cores,
            self.worker_local_ssd_data_disk,
            self.worker_pd_ssd_data_disk_size_gb,
        )
        cores_mcpu = adjust_cores_for_packability(cores_mcpu)

        if cores_mcpu < self.worker_cores * 1000:
            return cores_mcpu
        return None

    def adjust_for_remove_instance(self, instance):
        super().adjust_for_remove_instance(instance)
        if instance in self.healthy_instances_by_free_cores:
            self.healthy_instances_by_free_cores.remove(instance)

    def adjust_for_add_instance(self, instance):
        super().adjust_for_add_instance(instance)
        if instance.state == 'active' and instance.failed_request_count <= 1:
            self.healthy_instances_by_free_cores.add(instance)

    async def create_instance(self, cores=None, max_idle_time_msecs=None):
        if cores is None:
            cores = self.worker_cores

        if max_idle_time_msecs is None:
            max_idle_time_msecs = WORKER_MAX_IDLE_TIME_MSECS

        machine_name = self.generate_machine_name()

        zone = self.zone_monitor.get_zone(cores, self.worker_local_ssd_data_disk, self.worker_pd_ssd_data_disk_size_gb)
        if zone is None:
            return

        machine_type = f'n1-{self.worker_type}-{cores}'

        activation_token = secrets.token_urlsafe(32)

        instance = await Instance.create(
            app=self.app,
            inst_coll=self,
            name=machine_name,
            activation_token=activation_token,
            worker_cores_mcpu=cores * 1000,
            zone=zone,
            machine_type=machine_type,
            preemptible=True,
        )
        self.add_instance(instance)
        log.info(f'created {instance}')

        await create_instance(
            app=self.app,
            zone=zone,
            machine_name=machine_name,
            machine_type=machine_type,
            activation_token=activation_token,
            max_idle_time_msecs=max_idle_time_msecs,
            worker_local_ssd_data_disk=self.worker_local_ssd_data_disk,
            worker_pd_ssd_data_disk_size_gb=self.worker_pd_ssd_data_disk_size_gb,
            boot_disk_size_gb=self.boot_disk_size_gb,
            preemptible=True,
            job_private=False,
        )

    async def create_instances(self):
        ready_cores = await self.db.select_and_fetchone(
            '''
SELECT CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu
FROM user_inst_coll_resources
WHERE inst_coll = %s
LOCK IN SHARE MODE;
''',
            (self.name,),
        )

        ready_cores_mcpu = ready_cores['ready_cores_mcpu']

        free_cores_mcpu = sum([worker.free_cores_mcpu for worker in self.healthy_instances_by_free_cores])
        free_cores = free_cores_mcpu / 1000

        log.info(
            f'{self} n_instances {self.n_instances} {self.n_instances_by_state}'
            f' free_cores {free_cores} live_free_cores {self.live_free_cores_mcpu / 1000}'
            f' ready_cores {ready_cores_mcpu / 1000}'
        )

        if ready_cores_mcpu > 0 and free_cores < 500:
            n_live_instances = self.n_instances_by_state['pending'] + self.n_instances_by_state['active']

            instances_needed = (ready_cores_mcpu - self.live_free_cores_mcpu + (self.worker_cores * 1000) - 1) // (
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
                await asyncio.gather(*[self.create_instance() for _ in range(instances_needed)])

        n_live_instances = self.n_instances_by_state['pending'] + self.n_instances_by_state['active']
        if self.enable_standing_worker and n_live_instances == 0 and self.max_instances > 0:
            await self.create_instance(
                cores=self.standing_worker_cores, max_idle_time_msecs=STANDING_WORKER_MAX_IDLE_TIME_MSECS
            )

    async def control_loop(self):
        await periodically_call(15, self.create_instances)

    def __str__(self):
        return f'pool {self.name}'


class PoolScheduler:
    def __init__(self, app, pool):
        self.app = app
        self.scheduler_state_changed = pool.scheduler_state_changed
        self.db: Database = app['db']
        self.pool = pool
        self.async_worker_pool: AsyncWorkerPool = self.app['async_worker_pool']
        self.exceeded_shares_counter = ExceededSharesCounter()
        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(
            retry_long_running('schedule_loop', run_if_changed, self.scheduler_state_changed, self.schedule_loop_body)
        )

    def shutdown(self):
        try:
            self.task_manager.shutdown()
        finally:
            self.async_worker_pool.shutdown()

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
            timer_description=f'in compute_fair_share for {self.pool.name}: aggregate user_inst_coll_resources',
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
        log.info(f'schedule {self.pool}: starting')
        start = time_msecs()
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
SELECT id, cancelled, userdata, user, format_version
FROM batches
WHERE user = %s AND `state` = 'running';
''',
                (user,),
                timer_description=f'in schedule {self.pool}: get {user} running batches',
            ):
                async for record in self.db.select_and_fetchall(
                    '''
SELECT job_id, spec, cores_mcpu
FROM jobs FORCE INDEX(jobs_batch_id_state_always_run_inst_coll_cancelled)
WHERE batch_id = %s AND state = 'Ready' AND always_run = 1 AND inst_coll = %s
LIMIT %s;
''',
                    (batch['id'], self.pool.name, remaining.value),
                    timer_description=f'in schedule {self.pool}: get {user} batch {batch["id"]} runnable jobs (1)',
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
                        timer_description=f'in schedule {self.pool}: get {user} batch {batch["id"]} runnable jobs (2)',
                    ):
                        record['batch_id'] = batch['id']
                        record['userdata'] = batch['userdata']
                        record['user'] = batch['user']
                        record['format_version'] = batch['format_version']
                        yield record

        waitable_pool = WaitableSharedPool(self.async_worker_pool)

        def get_instance(user, cores_mcpu):
            i = self.pool.healthy_instances_by_free_cores.bisect_key_left(cores_mcpu)
            while i < len(self.pool.healthy_instances_by_free_cores):
                instance = self.pool.healthy_instances_by_free_cores[i]
                assert cores_mcpu <= instance.free_cores_mcpu
                if user != 'ci' or (user == 'ci' and instance.zone.startswith('us-central1')):
                    return instance
                i += 1
            histogram = collections.defaultdict(int)
            for instance in self.pool.healthy_instances_by_free_cores:
                histogram[instance.free_cores_mcpu] += 1
            log.info(f'schedule {self.pool}: no viable instances for {cores_mcpu}: {histogram}')
            return None

        should_wait = True
        for user, resources in user_resources.items():
            allocated_cores_mcpu = resources['allocated_cores_mcpu']
            if allocated_cores_mcpu == 0:
                continue

            scheduled_cores_mcpu = 0
            share = user_share[user]

            log.info(f'schedule {self.pool}: user-share: {user}: {allocated_cores_mcpu} {share}')

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
                            log.info(f'scheduling job {id} on {instance} for {self.pool}', exc_info=True)

                    await waitable_pool.call(schedule_with_error_handling, self.app, record, id, instance)

                remaining.value -= 1
                if remaining.value <= 0:
                    break

        await waitable_pool.wait()

        end = time_msecs()
        log.info(f'schedule: scheduled {n_scheduled} jobs in {end - start}ms for {self.pool}')

        return should_wait
