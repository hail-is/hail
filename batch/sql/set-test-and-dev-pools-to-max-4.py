import os
import asyncio
from gear import Database


async def main():
    if os.environ['HAIL_SCOPE'] == 'deploy':
        return

    max_instances = 4
    max_live_instances = 4

    db = Database()
    await db.async_init()

    await db.execute_update(
        '''
UPDATE inst_colls
SET max_instances = %s, max_live_instances = %s
''', (max_instances, max_live_instances))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
