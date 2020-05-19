import asyncio

from gear import Database
from hailtop.utils import (
    rate_gib_hour_to_mib_msec,
    rate_gib_month_to_mib_msec,
    rate_cpu_hour_to_mcpu_msec,
    rate_instance_hour_to_fraction_msec
)


async def main():
    # https://cloud.google.com/compute/all-pricing
    rates = [
        ('compute/n1-preemptible/1', rate_cpu_hour_to_mcpu_msec(0.006655)),
        ('memory/n1-preemptible/1', rate_gib_hour_to_mib_msec(0.000892)),
        ('boot-disk/pd-ssd/1', rate_gib_month_to_mib_msec(0.17)),
        ('ip-fee/1024/1', rate_instance_hour_to_fraction_msec(0.004, 1024)),
        ('service-fee/1', rate_cpu_hour_to_mcpu_msec(0.01))
    ]

    db = Database()
    await db.async_init()

    await db.execute_many('''
INSERT INTO `resources` (resource, rate)
VALUES (%s, %s)
''',
                          rates)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
