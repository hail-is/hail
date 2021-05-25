from typing import Tuple, List, Optional, Dict

import logging
import random
from collections import defaultdict

from hailtop import aiotools, aiogoogle
from hailtop.utils import periodically_call, url_basename

from ..utils import WindowFractionCounter
from ..batch_configuration import BATCH_GCP_REGIONS, GCP_ZONE

log = logging.getLogger('zoned_family_monitor')


class ZonedFamilyWeight:
    def __init__(self, zone: str, family: str, weight):
        self.zone = zone
        self.family = family
        self.weight = weight

    def __repr__(self):
        return f'{(self.zone, self.family)}: {self.weight}'


class ZonedFamilySuccessRate:
    def __init__(self):
        self._global_counter = WindowFractionCounter(10)
        self._zoned_family_counters: Dict[Tuple[str, str], WindowFractionCounter] = defaultdict(lambda: WindowFractionCounter(10))

    def push(self, zoned_family: Tuple[str, str], key: str, success: bool):
        self._global_counter.push(key, success)
        self._zoned_family_counters[zoned_family].push(key, success)

    def global_success_rate(self) -> float:
        return self._global_counter.fraction()

    def zoned_family_success_rate(self, zoned_family: Tuple[str, str]) -> float:
        return self._zoned_family_counters[zoned_family].fraction()

    def __repr__(self):
        return f'global {self._global_counter}, zones {self._zoned_family_counters}'


ZONES_TO_FAMILIES = {
    'us-central1-a': ['n1', 'n2'],
    'us-central1-b': ['n1', 'n2'],
    'us-central1-c': ['n1', 'n2'],
    'us-central1-f': ['n1', 'n2'],
    'us-east1-b': ['n1', 'n2'],
    'us-east1-c': ['n1', 'n2'],
    'us-east1-d': ['n1', 'n2'],
    'us-east4-a': ['n1', 'n2'],
    'us-east4-b': ['n1', 'n2'],
    'us-east4-c': ['n1', 'n2'],
    'us-west1-a': ['n1', 'n2'],
    'us-west1-b': ['n1', 'n2'],
    'us-west1-c': ['n1', 'n2'],
    'us-west2-a': ['n1'],
    'us-west2-b': ['n1'],
    'us-west2-c': ['n1'],
    'us-west3-a': ['n1'],
    'us-west3-b': ['n1'],
    'us-west3-c': ['n1'],
    'us-west4-a': ['n1', 'n2'],
    'us-west4-b': ['n1', 'n2'],
    'us-west4-c': ['n1', 'n2'],
}


class ZonedFamilyMonitor:
    def __init__(self, app):
        self.app = app
        self.compute_client: aiogoogle.ComputeClient = app['compute_client']

        self.zoned_family_success_rate = ZonedFamilySuccessRate()

        self.region_info = None
        self.zones = []

        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(self.update_region_quotas_loop())

    def shutdown(self):
        self.task_manager.shutdown()

    def get_zone(self, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb) -> Optional[str]:
        zoned_family = self.get_zoned_family(worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb)
        if zoned_family:
            return zoned_family[0]
        return None

    def get_zoned_family(self, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb) -> Optional[Tuple[str, str]]:
        if self.app['inst_coll_manager'].global_live_total_cores_mcpu // 1000 < 1_000:
            return (GCP_ZONE, 'n2')

        zoned_family_weights = self.zoned_family_weights(
            worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb)

        if not zoned_family_weights:
            return None

        zones = [(zw.zone, zw.family) for zw in zoned_family_weights]

        zoned_family_prob_weights = [
            min(zw.weight, 10) * self.zoned_family_success_rate.zoned_family_success_rate((zw.zone, zw.family)) for zw in zoned_family_weights
        ]

        log.info(f'zoned_family_success_rate {self.zoned_family_success_rate}')
        log.info(f'zoned_family_prob_weights {zoned_family_prob_weights}')

        return random.choices(zones, zoned_family_prob_weights)[0]

    def zoned_family_weights(self, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb) -> Optional[List[ZonedFamilyWeight]]:
        if not self.region_info:
            return None

        _zoned_family_weights = []
        for r in self.region_info.values():
            quota_remaining = {q['metric']: q['limit'] - q['usage'] for q in r['quotas']}

            remaining = quota_remaining['PREEMPTIBLE_CPUS'] / worker_cores
            if worker_local_ssd_data_disk:
                remaining = min(remaining, quota_remaining['LOCAL_SSD_TOTAL_GB'] / 375)
            else:
                remaining = min(remaining, quota_remaining['SSD_TOTAL_GB'] / worker_pd_ssd_data_disk_size_gb)

            weight = max(remaining / len(r['zones']), 1)
            for z in r['zones']:
                zoned_family_name = url_basename(z)
                for family in ZONES_TO_FAMILIES[z]:
                    _zoned_family_weights.append(ZonedFamilyWeight(zoned_family_name, family, weight))

        log.info(f'zoned_family_weights {_zoned_family_weights}')
        return _zoned_family_weights

    async def update_region_quotas(self):
        self.region_info = {name: await self.compute_client.get(f'/regions/{name}') for name in BATCH_GCP_REGIONS}

        self.zones = [url_basename(z) for r in self.region_info.values() for z in r['zones']]

        log.info('updated region quotas')

    async def update_region_quotas_loop(self):
        await periodically_call(60, self.update_region_quotas)
