import os
import asyncio
from gear import Database, transaction


async def main():
    scope = os.environ['HAIL_SCOPE']

    if scope in ('deploy', 'dev'):
        standing_worker_max_idle_time_secs = 7200  # 2 hours
    else:
        standing_worker_max_idle_time_secs = 300  # 5 minutes

    db = Database()
    await db.async_init()

    columns = [
        ('inst_colls', 'max_new_instances_per_autoscaler_loop', 'INT', 10),  # n * 16 cores / 15s = excess_scheduling_rate/s = 10/s => n ~= 10
        ('inst_colls', 'autoscaler_loop_period_secs', 'INT', 15),
        ('inst_colls', 'worker_max_idle_time_secs', 'INT', 30),
        ('pools', 'standing_worker_max_idle_time_secs', 'INT', standing_worker_max_idle_time_secs),
        ('pools', 'job_queue_scheduling_window_secs', 'INT', 150),  # 2.5 minutes is approximately worker start up time
    ]

    @transaction(db)
    async def insert_and_update(tx):
        for table, col, typ, value in columns:
            await tx.just_execute(f'ALTER TABLE {table} ADD COLUMN {col} {typ}, ALGORITHM=INSTANT;')
            await tx.execute_update(f'UPDATE {table} SET {col} = {value};')
            await tx.just_execute(f'ALTER TABLE {table} MODIFY COLUMN {col} {typ} NOT NULL;')

    await insert_and_update()  # pylint: disable=no-value-for-parameter

    await db.async_close()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
