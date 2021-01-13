from gear import Database


class PoolConfig:
    @staticmethod
    def from_record(record):
        return PoolConfig(record['name'], record['worker_type'], record['worker_cores'],
                          record['boot_disk_size_gb'], record['worker_local_ssd_data_disk'],
                          record['worker_pd_ssd_data_disk_size_gb'])

    def __init__(self, name, worker_type, worker_cores, boot_disk_size_gb,
                 worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb):
        self.name = name
        self.worker_type = worker_type
        self.worker_cores = worker_cores
        self.worker_local_ssd_data_disk = worker_local_ssd_data_disk
        self.worker_pd_ssd_data_disk_size_gb = worker_pd_ssd_data_disk_size_gb
        self.boot_disk_size_gb = boot_disk_size_gb


class PoolSelector:
    def __init__(self, app):
        self.app = app
        self.db: Database = app['db']
        self.pool_configs = {}

    async def async_init(self):
        records = self.db.execute_and_fetchall('''
SELECT name, worker_type, worker_cores, boot_disk_size_gb,
  worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb
FROM pools
LEFT JOIN inst_colls ON pools.name = inst_colls.name;
''')
        async for record in records:
            self.pool_configs[record['name']] = PoolConfig.from_record(record)

    def select_pool(self, worker_type=None):
        if worker_type:
            pools = [pool for pool in self.pool_configs.values()
                     if pool.worker_type == worker_type]
            assert len(pools) == 1, pools
            return pools[0]
        return None
