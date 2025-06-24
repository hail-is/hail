import os
import asyncio
from gear import Database


async def main():
    scope = os.environ['HAIL_SCOPE']

    if scope == 'deploy':
        return
    else:
        max_instances = 4
        pool_size = 3

    db = Database()
    await db.async_init()

    await db.execute_update(
        '''
UPDATE globals SET max_instances = %s, pool_size = %s;
''',
        (max_instances, pool_size))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
