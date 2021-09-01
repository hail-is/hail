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
WHERE `name` = 'standard'
SET max_instances = 8, pool_size = 8;
''')

    await db.execute_update(
        '''
UPDATE pools
WHERE `name` = 'standard' AND `worker_type` = 'standard'
SET worker_cores = 8;
''')

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

