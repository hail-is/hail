import logging
import random
from typing import Dict, Any, List, Optional, Set, Tuple

from hailtop.aiocloud import aiogoogle
from hailtop.utils import url_basename

from ....utils import WindowFractionCounter

log = logging.getLogger('zones')


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


async def update_region_quotas(compute_client: aiogoogle.GoogleComputeClient, regions: Set[str]
                               ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    region_info = {name: await compute_client.get(f'/regions/{name}') for name in regions}
    zones = [url_basename(z) for r in region_info.values() for z in r['zones']]
    log.info('updated region quotas')
    return region_info, zones


def compute_zone_weights(region_info: Dict[str, Dict[str, Any]], worker_cores: int,
                         worker_local_ssd_data_disk: bool, worker_pd_ssd_data_disk_size_gb: int) -> Optional[List[ZoneWeight]]:
    if not region_info:
        return None

    weights = []
    for r in region_info.values():
        quota_remaining = {q['metric']: q['limit'] - q['usage'] for q in r['quotas']}

        remaining = quota_remaining['PREEMPTIBLE_CPUS'] / worker_cores
        if worker_local_ssd_data_disk:
            remaining = min(remaining, quota_remaining['LOCAL_SSD_TOTAL_GB'] / 375)
        else:
            remaining = min(remaining, quota_remaining['SSD_TOTAL_GB'] / worker_pd_ssd_data_disk_size_gb)

        weight = max(remaining / len(r['zones']), 1)
        for z in r['zones']:
            zone_name = url_basename(z)
            weights.append(ZoneWeight(zone_name, weight))

    log.info(f'zone_weights {weights}')
    return weights


def get_zone(region_info: Dict[str, Dict[str, Any]], zone_success_rate: ZoneSuccessRate, worker_cores: int,
             worker_local_ssd_data_disk: bool, worker_pd_ssd_data_disk_size_gb: int) -> Optional[str]:
    zone_weights = compute_zone_weights(region_info, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb)

    if not zone_weights:
        return None

    zones = [zw.zone for zw in zone_weights]

    zone_prob_weights = [
        min(zw.weight, 10) * zone_success_rate.zone_success_rate(zw.zone) for zw in zone_weights
    ]

    log.info(f'zone_success_rate {zone_success_rate}')
    log.info(f'zone_prob_weights {zone_prob_weights}')

    zone = random.choices(zones, zone_prob_weights)[0]
    return zone
