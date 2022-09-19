import os
import asyncio
from gear import Database


async def main():
    if os.environ['HAIL_SCOPE'] != 'dev':
        return

    db = Database()
    await db.async_init()

    await db.execute_update(
        '''
UPDATE pools
SET enable_standing_worker = %s
''', (False,))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
