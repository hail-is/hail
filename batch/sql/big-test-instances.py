import os
import asyncio
from gear import Database


async def main():
    scope = os.environ['HAIL_SCOPE']

    if scope == 'deploy':
        return

    db = Database()
    await db.async_init()

    await db.execute_update(
        '''
UPDATE inst_colls
SET max_instances = 8, max_live_instances = 8
WHERE `name` = 'standard';
''')

    await db.execute_update(
        '''
UPDATE pools
SET worker_cores = 16, standing_worker_cores = 16, enable_standing_worker = 1
WHERE `name` = 'standard' AND `worker_type` = 'standard';
''')

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
