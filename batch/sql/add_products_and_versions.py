import asyncio
import os

from hailtop.utils.rates import rate_instance_hour_to_fraction_msec, rate_cpu_hour_to_mcpu_msec
from gear import Database, transaction


async def main():
    cloud = os.environ['HAIL_CLOUD']

    db = Database()
    await db.async_init()

    @transaction(db)
    async def populate(tx):
        resources = [resource['resource'] async for resource in
                     tx.execute_and_fetchall('SELECT resource FROM resources')]

        latest_versions = []
        for resource in resources:
            prefix, version = resource.rsplit('/', maxsplit=1)
            latest_versions.append((prefix, version))

        await tx.execute_many('''
INSERT INTO `latest_product_versions` (product, version)
VALUES (%s, %s)
''',
                              latest_versions)

        if cloud == 'azure':
            # https://azure.microsoft.com/en-us/pricing/details/ip-addresses/
            azure_ip_fee = 0.004

            resources = [
                ('az/ip-fee/1024/2021-12-01', rate_instance_hour_to_fraction_msec(azure_ip_fee, 1024)),
                ('az/service-fee/2021-12-01', rate_cpu_hour_to_mcpu_msec(0.01)),
            ]

            await tx.execute_many('''
INSERT INTO `resources` (resource, rate)
VALUES (%s, %s)
''',
                                  resources)

            resource_versions = [
                ('az/ip-fee/1024', '2021-12-01'),
                ('az/service-fee', '2021-12-01'),
            ]

            await tx.execute_many('''
INSERT INTO `latest_product_versions` (product, version)
VALUES (%s, %s)
''',
                                  resource_versions)

    await populate()
    await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
