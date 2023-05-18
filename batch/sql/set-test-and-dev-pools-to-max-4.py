import os
import asyncio
from gear import Database


async def main():
    if os.environ['HAIL_SCOPE'] == 'deploy':
        return

    max_instances = 8
    max_live_instances = 8
    worker_max_idle_time_secs = 3 * 60
    standing_worker_cores = 16

    db = Database()
    await db.async_init()

    await db.execute_update(
        '''
UPDATE inst_colls
SET max_instances = %s, max_live_instances = %s, worker_max_idle_time_secs = %s
''', (max_instances, max_live_instances, worker_max_idle_time_secs))

    if os.environ['HAIL_SCOPE'] == 'dev':
        return

    await db.execute_update(
        '''
UPDATE pools
SET standing_worker_cores = %s
''', (standing_worker_cores,))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
