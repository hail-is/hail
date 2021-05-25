import asyncio

from gear import Database
from hailtop.utils import (
    rate_gib_hour_to_mib_msec,
    rate_cpu_hour_to_mcpu_msec
)


async def main():
    # https://cloud.google.com/compute/all-pricing
    rates = [
        ('compute/n2-preemptible/1', rate_cpu_hour_to_mcpu_msec(0.00765)),
        ('memory/n2-preemptible/1', rate_gib_hour_to_mib_msec(0.00103)),
        ('compute/n2-nonpreemptible/1', rate_cpu_hour_to_mcpu_msec(0.031611)),
        ('memory/n2-nonpreemptible/1', rate_gib_hour_to_mib_msec(0.004237)),
    ]

    db = Database()
    await db.async_init()

    await db.execute_many('''
INSERT INTO `resources` (resource, rate)
VALUES (%s, %s)
''',
                          rates)

    await db.execute_update('''
UPDATE pools SET worker_cores = 8
''')


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
