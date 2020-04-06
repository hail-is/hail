import os
import asyncio
from gear import Database


async def main():
    scope = os.environ['HAIL_SCOPE']

    if scope == 'deploy':
        return
    else:
        worker_disk_size_gb = 20

    db = Database()
    await db.async_init()

    await db.execute_insertone(
        '''
INSERT INTO globals (worker_disk_size_gb) VALUES (%s);
''',
        (worker_disk_size_gb,))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
