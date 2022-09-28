import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from hailtop.aiocloud import aiogoogle
from hailtop.utils import url_basename

from ....driver.location import CloudLocationMonitor
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


class ZoneMonitor(CloudLocationMonitor):
    @staticmethod
    async def create(
        compute_client: aiogoogle.GoogleComputeClient,  # BORROWED
        regions: Set[str],
        default_zone: str,
    ) -> 'ZoneMonitor':
        region_info, zones = await fetch_region_quotas(compute_client, regions)
        return ZoneMonitor(compute_client, region_info, zones, regions, default_zone)

    def __init__(
        self,
        compute_client: aiogoogle.GoogleComputeClient,  # BORROWED
        initial_region_info: Dict[str, Dict[str, Any]],
        initial_zones: List[str],
        regions: Set[str],
        default_zone: str,
    ):
        self._compute_client = compute_client
        self._region_info: Dict[str, Dict[str, Any]] = initial_region_info
        self._regions = regions
        self.zones: List[str] = initial_zones
        self._default_zone = default_zone

        self.zone_success_rate = ZoneSuccessRate()

    @property
    def region_quotas(self):
        return self._region_info

    def default_location(self) -> str:
        return self._default_zone

    def choose_location(
        self, cores: int, local_ssd_data_disk: bool, data_disk_size_gb: int, preemptible: bool, region: Optional[str]
    ) -> str:
        zone_weights = self.compute_zone_weights(cores, local_ssd_data_disk, data_disk_size_gb, preemptible, region)

        zones = [zw.zone for zw in zone_weights]

        zone_prob_weights = [
            min(zw.weight, 10) * self.zone_success_rate.zone_success_rate(zw.zone) for zw in zone_weights
        ]

        log.info(f'zone_success_rate {self.zone_success_rate}')
        log.info(f'zone_prob_weights {zone_prob_weights}')

        zone = random.choices(zones, zone_prob_weights)[0]
        return zone

    def compute_zone_weights(
        self,
        worker_cores: int,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        preemptible: bool,
        region: Optional[str],
    ) -> List[ZoneWeight]:
        weights = []
        for region_name, r in self._region_info.items():
            if region is not None and region_name != region:
                continue

            quota_remaining = {q['metric']: q['limit'] - q['usage'] for q in r['quotas']}

            cpu_label = 'PREEMPTIBLE_CPUS' if preemptible else 'CPUS'
            remaining = quota_remaining[cpu_label] / worker_cores

            if local_ssd_data_disk:
                specific_disk_type_quota = quota_remaining['LOCAL_SSD_TOTAL_GB']
            else:
                specific_disk_type_quota = quota_remaining['SSD_TOTAL_GB']
            # FIXME: data_disk_size_gb is assumed to be constant across all instances, but it is
            # passed as a variable parameter to this function!!
            remaining = min(remaining, specific_disk_type_quota / data_disk_size_gb)

            weight = max(remaining / len(r['zones']), 1)
            for z in r['zones']:
                zone_name = url_basename(z)
                weights.append(ZoneWeight(zone_name, weight))

        log.info(f'zone_weights {weights}')
        return weights

    async def update_region_quotas(self):
        self._region_info, self.zones = await fetch_region_quotas(self._compute_client, self._regions)
        log.info('updated region quotas')


async def fetch_region_quotas(
    compute_client: aiogoogle.GoogleComputeClient, regions: Set[str]
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    region_info = {name: await compute_client.get(f'/regions/{name}') for name in regions}
    zones = [url_basename(z) for r in region_info.values() for z in r['zones']]
    return region_info, zones
