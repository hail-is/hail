import asyncio
import os

from gear import Database


async def main():
    cloud = os.environ['HAIL_CLOUD']
    if cloud != 'gcp':
        return

    db = Database()
    await db.async_init()

    resources = [resource['resource'] async for resource in db.select_and_fetchall('SELECT resource FROM resources')]

    latest_versions = []
    for resource in resources:
        prefix, version = resource.rsplit('/', maxsplit=1)
        latest_versions.append((prefix, version))

    await db.execute_many('''
INSERT INTO `latest_resource_versions` (prefix, version)
VALUES (%s, %s)
''',
                          latest_versions)

    await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
