import asyncio
import os

from hailtop.utils.rates import rate_instance_hour_to_fraction_msec, rate_cpu_hour_to_mcpu_msec
from gear import Database


async def main():
    cloud = os.environ['HAIL_CLOUD']
    if cloud != 'azure':
        return

    db = Database()
    await db.async_init()

    # https://azure.microsoft.com/en-us/pricing/details/ip-addresses/
    azure_ip_fee = 0.004

    resources = [
        ('az/ip-fee/1024/2021-11-01', rate_instance_hour_to_fraction_msec(azure_ip_fee, 1024)),
        ('az/service-fee/2021-11-01', rate_cpu_hour_to_mcpu_msec(0.01)),
    ]

    await db.execute_many('''
INSERT INTO `resources` (resource, rate)
VALUES (%s, %s)
''',
                          resources)

    resource_versions = [
        ('az/ip-fee/1024', '2021-11-01'),
        ('az/service-fee', '2021-11-01'),
    ]

    await db.execute_many('''
INSERT INTO `latest_resource_versions` (prefix, version)
VALUES (%s, %s)
''',
                          resource_versions)

    await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
