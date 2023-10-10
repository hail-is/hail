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
            await tx.execute_insertone('''
INSERT INTO latest_product_versions (product, version)
VALUES ('gcp-support-logs-specs-and-firewall-fees', '1');
''')
            # 0.005 USD per core-hour
            await tx.execute_insertone('''
INSERT INTO resources (resource, rate)
VALUES ('gcp-support-logs-specs-and-firewall-fees/1', 0.000000000001388888888888889);
''')
            await tx.execute_update('''
UPDATE resources
SET deduped_resource_id = resource_id
WHERE resource = 'gcp-support-logs-specs-and-firewall-fees/1';
''')

        await insert()
    finally:
        await db.async_close()


asyncio.run(main())
