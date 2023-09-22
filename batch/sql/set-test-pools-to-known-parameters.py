import os
import asyncio
from gear import Database


async def main():
    if os.environ['HAIL_SCOPE'] in ('deploy', 'dev'):
        return

    db = Database()
    await db.async_init()

    async def set_pool(
        name,
        max_instances = 16,
        max_live_instances = 16,
        max_new_instances_per_autoscaler_loop = 10,
        autoscaler_loop_period_secs = 15,
        worker_max_idle_time_secs = 120,
        worker_cores = 16,
        enable_standing_worker = False,
        standing_worker_cores = 16,
        standing_worker_max_idle_time_secs = 5 * 60,
        job_queue_scheduling_window_secs = 150,
        min_instances = 0,
    ):
        assert not enable_standing_worker, 'use min_instances'

        await db.execute_update(
            '''
UPDATE inst_colls
SET
  max_instances = %s,
  max_live_instances = %s,
  max_new_instances_per_autoscaler_loop = %s,
  autoscaler_loop_period_secs = %s,
  worker_max_idle_time_secs = %s
WHERE name = %s
            ''',
            (
                max_instances,
                max_live_instances,
                max_new_instances_per_autoscaler_loop,
                autoscaler_loop_period_secs,
                worker_max_idle_time_secs,
                name,
            )
        )
        await db.execute_update(
            '''
UPDATE pools
SET
  worker_cores = %s,
  enable_standing_worker = %s,
  standing_worker_cores = %s,
  standing_worker_max_idle_time_secs = %s,
  job_queue_scheduling_window_secs = %s,
  min_instances = %s
WHERE name = %s
            ''',
            (
                worker_cores,
                enable_standing_worker,
                standing_worker_cores,
                standing_worker_max_idle_time_secs,
                job_queue_scheduling_window_secs,
                min_instances,
                name,
            )
        )

    await set_pool(
        'highcpu',
        min_instances=0,  # min_instances means n_standing_workers
    )
    await set_pool(
        'highcpu-np',
        min_instances=0,  # min_instances means n_standing_workers
    )
    await set_pool(
        'highmem',
        min_instances=0,  # min_instances means n_standing_workers
    )
    await set_pool(
        'highmem-np',
        min_instances=0,  # min_instances means n_standing_workers
    )
    await set_pool(
        'standard',
        min_instances=1,  # min_instances means n_standing_workers
    )
    await set_pool(
        'standard-np',
        min_instances=1,  # min_instances means n_standing_workers
    )
    await db.async_close()


asyncio.run(main())
