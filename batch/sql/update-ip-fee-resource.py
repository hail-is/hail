import os
import asyncio
from gear import Database, transaction, Transaction


async def main():
    cloud = os.environ['HAIL_CLOUD']
    if cloud != 'gcp':
        return

    db = Database()
    await db.async_init()
    try:
        @transaction(db)
        async def insert(tx: Transaction):
            await tx.execute_many(
                '''
INSERT INTO latest_product_versions (product, version)
VALUES (%s, %s);
''',
                [('ip-fee/preemptible/1024', '1'),
                 ('ip-fee/nonpreemptible/1024', '1')]
            )

            # https://cloud.google.com/vpc/pricing-announce-external-ips
            # from hailtop.utils import rate_instance_hour_to_fraction_msec
            # spot_ip_fee = rate_instance_hour_to_fraction_msec(0.0025, 1024)
            spot_ip_fee = 6.781684027777778e-13
            # standard_ip_fee = rate_instance_hour_to_fraction_msec(0.005, 1024)
            standard_ip_fee = 1.3563368055555557e-12

            await tx.execute_many(
                '''
INSERT INTO resources (resource, rate)
VALUES (%s, %s);
''',
                [('ip-fee/preemptible/1024/1', spot_ip_fee),
                 ('ip-fee/nonpreemptible/1024/1', standard_ip_fee)]
            )

            await tx.execute_update('''
UPDATE resources
SET deduped_resource_id = resource_id
WHERE resource = 'ip-fee/preemptible/1024/1' OR resource = 'ip-fee/nonpreemptible/1024/1';
''')

        await insert()
    finally:
        await db.async_close()


asyncio.run(main())
