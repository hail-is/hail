import os
import asyncio
import urllib.parse
import logging
from hailtop import aiotools
from hailtop.utils import retry_long_running

from ..utils import WindowFractionCounter
from ..batch_configuration import GCP_REGION, BATCH_GCP_REGIONS

log = logging.getLogger('zone_monitor')


class ZoneWeight:
    def __init__(self, zone, weight):
        self.zone = zone
        self.weight = weight

    def __repr__(self):
        return f'{self.zone}: {self.weight}'


class ZoneSuccessRate:
    def __init__(self):
        self._global_counter = WindowFractionCounter(10)
        self._zone_counters = {}

    def _get_zone_counter(self, zone: str):
        zone_counter = self._zone_counters.get(zone)
        if not zone_counter:
            zone_counter = WindowFractionCounter(10)
            self._zone_counters[zone] = zone_counter
        return zone_counter

    def push(self, zone: str, key: str, success: bool):
        self._global_counter.push(key, success)
        zone_counter = self._get_zone_counter(zone)
        zone_counter.push(key, success)

    def global_success_rate(self) -> float:
        return self._global_counter.fraction()

    def zone_success_rate(self, zone) -> float:
        zone_counter = self._get_zone_counter(zone)
        return zone_counter.fraction()

    def __repr__(self):
        return f'global {self._global_counter}, zones {self._zone_counters}'


class ZoneMonitor:
    def __init__(self, app):
        self.app = app
        self.compute_client = app['compute_client']

        self.zone_success_rate = ZoneSuccessRate()

        self.init_zones = None
        self.init_zone_weights = None

        self.region_info = None

        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(retry_long_running(
            'update_zones_loop',
            self.update_region_quotas_loop))

    def shutdown(self):
        self.task_manager.shutdown()

    def zone_weights(self, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb):
        if not self.region_info:
            return None

        _zone_weights = []
        for r in self.region_info.values():
            quota_remaining = {
                q['metric']: q['limit'] - q['usage']
                for q in r['quotas']
            }

            remaining = quota_remaining['PREEMPTIBLE_CPUS'] / worker_cores
            if worker_local_ssd_data_disk:
                remaining = min(remaining, quota_remaining['LOCAL_SSD_TOTAL_GB'] / 375)
            else:
                remaining = min(remaining, quota_remaining['SSD_TOTAL_GB'] / worker_pd_ssd_data_disk_size_gb)

            weight = max(remaining / len(r['zones']), 1)
            for z in r['zones']:
                zone_name = os.path.basename(urllib.parse.urlparse(z).path)
                _zone_weights.append(ZoneWeight(zone_name, weight))

        log.info(f'zone_weights {_zone_weights}')
        return _zone_weights

    async def update_region_quotas(self):
        self.region_info = {
            name: await self.compute_client.get(f'/regions/{name}')
            for name in BATCH_GCP_REGIONS
        }

        self.init_zones = [
            os.path.basename(urllib.parse.urlparse(z).path)
            for z in self.region_info[GCP_REGION]['zones']
        ]
        self.init_zone_weights = [ZoneWeight(z, 1) for z in self.init_zones]

        log.info('updated region quotas')

    async def update_region_quotas_loop(self):
        while True:
            log.info('update region quotas loop')
            await self.update_region_quotas()
            await asyncio.sleep(60)
