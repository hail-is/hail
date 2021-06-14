import logging
import random

from hailtop import aiotools, aiogoogle
from hailtop.utils import periodically_call, url_basename

from ..utils import WindowFractionCounter
from ..batch_configuration import BATCH_GCP_REGIONS, GCP_ZONE

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
        self.compute_client: aiogoogle.ComputeClient = app['compute_client']

        self.zone_success_rate = ZoneSuccessRate()

        self.region_info = None
        self.zones = []

        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(self.update_region_quotas_loop())

    def shutdown(self):
        self.task_manager.shutdown()

    def get_zone(self, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb):
        if self.app['inst_coll_manager'].global_live_total_cores_mcpu // 1000 < 1_000:
            zone = GCP_ZONE
        else:
            zone_weights = self.zone_weights(worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb)

            if not zone_weights:
                return None

            zones = [zw.zone for zw in zone_weights]

            zone_prob_weights = [
                min(zw.weight, 10) * self.zone_success_rate.zone_success_rate(zw.zone) for zw in zone_weights
            ]

            log.info(f'zone_success_rate {self.zone_success_rate}')
            log.info(f'zone_prob_weights {zone_prob_weights}')

            zone = random.choices(zones, zone_prob_weights)[0]
        return zone

    def zone_weights(self, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb):
        if not self.region_info:
            return None

        _zone_weights = []
        for r in self.region_info.values():
            quota_remaining = {q['metric']: q['limit'] - q['usage'] for q in r['quotas']}

            remaining = quota_remaining['PREEMPTIBLE_CPUS'] / worker_cores
            if worker_local_ssd_data_disk:
                remaining = min(remaining, quota_remaining['LOCAL_SSD_TOTAL_GB'] / 375)
            else:
                remaining = min(remaining, quota_remaining['SSD_TOTAL_GB'] / worker_pd_ssd_data_disk_size_gb)

            weight = max(remaining / len(r['zones']), 1)
            for z in r['zones']:
                zone_name = url_basename(z)
                _zone_weights.append(ZoneWeight(zone_name, weight))

        log.info(f'zone_weights {_zone_weights}')
        return _zone_weights

    async def update_region_quotas(self):
        self.region_info = {name: await self.compute_client.get(f'/regions/{name}') for name in BATCH_GCP_REGIONS}

        self.zones = [url_basename(z) for r in self.region_info.values() for z in r['zones']]

        log.info('updated region quotas')

    async def update_region_quotas_loop(self):
        await periodically_call(60, self.update_region_quotas)
