import os
import asyncio
from gear import Database


async def main():
    if os.environ['HAIL_SCOPE'] != 'test':
        return
    db = Database()
    await db.async_init()
    await db.execute_update(
        '''
UPDATE pools SET min_instances = 1;
''')
    await db.async_close()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
