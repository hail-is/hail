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
SET enable_standing_worker = 0
'''
    )


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
