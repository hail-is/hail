import asyncio
import os

from gear import Database
from hailtop.utils import rate_gib_month_to_mib_msec, rate_cpu_hour_to_mcpu_msec, rate_gib_hour_to_mib_msec


async def main():
    db = Database()
    await db.async_init()

    # https://cloud.google.com/compute/disks-image-pricing#localssdpricing
    # https://cloud.google.com/compute/disks-image-pricing#persistentdisk
    rates = [
        ('disk/local-ssd/preemptible/1', rate_gib_month_to_mib_msec(0.048)),
        ('disk/local-ssd/nonpreemptible/1', rate_gib_month_to_mib_msec(0.08)),
        ('compute/n1-preemptible/2', rate_cpu_hour_to_mcpu_msec(0.00698)),
        ('memory/n1-preemptible/2', rate_gib_hour_to_mib_msec(0.00094)),
        ('compute/n1-nonpreemptible/2', rate_cpu_hour_to_mcpu_msec(0.033174)),
        ('memory/n1-nonpreemptible/2', rate_gib_hour_to_mib_msec(0.004446)),
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
        ('compute/n1-preemptible', '2'),
        ('memory/n1-preemptible', '2'),
        ('compute/n1-nonpreemptible', '2'),
        ('memory/n1-nonpreemptible', '2'),
    ]

    await db.execute_many(
        '''
INSERT INTO `latest_product_versions` (product, version)
VALUES (%s, %s)
''',
        product_versions)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
