import asyncio
import os

from gear import Database
from hailtop.utils import rate_gib_month_to_mib_msec


async def main():
    db = Database()
    await db.async_init()

    # https://cloud.google.com/compute/disks-image-pricing#localssdpricing
    # https://cloud.google.com/compute/disks-image-pricing#persistentdisk
    rates = [
        ('disk/local-ssd/preemptible/1', rate_gib_month_to_mib_msec(0.048)),
        ('disk/local-ssd/nonpreemptible/1', rate_gib_month_to_mib_msec(0.08)),
    ]
    
    await db.execute_many(
        '''
INSERT INTO `resources` (resource, rate)
VALUES (%s, %s)
''',
        rates)

    product_versions = [
        ('disk/local-ssd/preemptible', '1'),
        ('disk/local-ssd/nonpreemptible', '1'),
    ]

    await db.execute_many(
        '''
INSERT INTO `latest_product_versions` (product, version)
VALUES (%s, %s)
''',
        product_versions)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
