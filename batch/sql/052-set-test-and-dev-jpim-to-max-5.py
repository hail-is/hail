import os
import asyncio
from gear import Database


async def main():
    if os.environ['HAIL_SCOPE'] in ('dev', 'deploy'):
        return

    max_instances = 5
    max_live_instances = 5

    db = Database()
    await db.async_init()

    await db.execute_update(
        '''
UPDATE inst_colls
SET max_instances = %s, max_live_instances = %s
WHERE not is_pool
''', (max_instances, max_live_instances))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
