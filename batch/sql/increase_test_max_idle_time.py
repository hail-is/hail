import os
import asyncio
from gear import Database


async def main():
    if os.environ['HAIL_SCOPE'] == 'deploy':
        return

    db = Database()
    await db.async_init()

    await db.execute_update(
        '''
UPDATE inst_colls
SET worker_max_idle_time_secs = 120
''')

    await db.execute_update(
        '''
UPDATE pools
SET standing_worker_cores = 16, min_instances = 4
WHERE name = "standard-np" OR name = "standard";
''')

    await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
