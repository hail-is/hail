import os
import asyncio
from gear import Database


async def main():
    cloud = os.environ['HAIL_CLOUD']
    if cloud != 'gcp':
        return

    db = Database()
    try:
        await db.async_init()
        await db.execute_insertone('''
INSERT INTO latest_product_versions (product, version)
VALUES ('gcp-support-logs-specs-and-firewall-fees', '1');
''')
        # 0.005 USD per core-hour
        await db.execute_insertone('''
INSERT INTO resources (resource, rate)
VALUES ('gcp-support-logs-specs-and-firewall-fees/1', 0.000000000001388888888888889);
''')
    finally:
        await db.async_close()


asyncio.run(main())
