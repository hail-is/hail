import asyncio
import os

from gear import Database
from hailtop.utils import rate_gib_month_to_mib_msec


async def main():
    # https://cloud.google.com/compute/disks-image-pricing#localssdpricing
    # https://cloud.google.com/compute/disks-image-pricing#persistentdisk
    rates = [
        ('disk/local-ssd/1', rate_gib_month_to_mib_msec(0.048)),
        ('disk/pd-ssd/1', rate_gib_month_to_mib_msec(0.17))
    ]

    db = Database()
    await db.async_init()

    await db.execute_many('''
INSERT INTO `resources` (resource, rate)
VALUES (%s, %s)
''',
                          rates)

    scope = os.environ['HAIL_SCOPE']

    if scope == 'deploy':
        worker_disk_size_gb = 10
    else:
        return

    db = Database()
    await db.async_init()

    await db.execute_insertone(
        '''
UPDATE globals SET worker_disk_size_gb = %s;
''',
        (worker_disk_size_gb,))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())