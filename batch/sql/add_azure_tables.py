import asyncio
import os

from gear import Database


async def main():
    db = Database()
    try:
        await db.async_init()

        cloud = os.environ['HAIL_CLOUD']
        if cloud != 'azure':
            return

        scope = os.environ['HAIL_SCOPE']

        await db.just_execute('''
DELETE FROM `pools`;    
''')
        await db.just_execute('''
DELETE FROM `inst_colls`;    
''')

        enable_standing_worker = scope != 'dev'
        boot_disk_size_gb = 30
        worker_cores = 16

        if scope == 'deploy':
            max_instances = 1000
            max_live_instances = 20
        else:
            assert scope in ('dev', 'test')
            max_instances = 8
            max_live_instances = 8

        inst_colls = [
            ('highcpu', True, boot_disk_size_gb, max_instances, max_live_instances, 'azure'),
            ('standard', True, boot_disk_size_gb, max_instances, max_live_instances, 'azure'),
            ('highmem', True, boot_disk_size_gb, max_instances, max_live_instances, 'azure'),
            ('job-private', False, boot_disk_size_gb, max_instances, max_live_instances, 'azure'),
        ]

        await db.execute_many('''
INSERT INTO `inst_colls` (name, is_pool, boot_disk_size_gb, max_instances, max_live_instances, cloud)
VALUES (%s, %s, %s, %s, %s, %s)
''',
                              inst_colls)

        pools = [
            ('highcpu', 'F', worker_cores, True, 0, enable_standing_worker, 8),
            ('standard', 'D', worker_cores, True, 0, enable_standing_worker, 4),
            ('highmem', 'E', worker_cores, True, 0, enable_standing_worker, 4),
        ]

        await db.execute_many('''
INSERT INTO `inst_colls` (name, worker_type, worker_cores, worker_local_ssd_data_disk, worker_external_ssd_data_disk_size_gb, enable_standing_worker, standing_worker_cores)
VALUES (%s, %s, %s, %s, %s, %s, %s)
''',
                              pools)
    finally:
        await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
