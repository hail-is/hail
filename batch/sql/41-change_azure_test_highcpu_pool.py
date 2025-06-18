import asyncio
import os

from gear import Database


async def main():
    db = Database()
    try:
        await db.async_init()

        cloud = os.environ['HAIL_CLOUD']
        if cloud != 'azure':
            return

        await db.just_execute('''
UPDATE `pools` SET worker_external_ssd_data_disk_size_gb = %s WHERE name = %s
''',
                              (375, 'highcpu'))
    finally:
        await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
