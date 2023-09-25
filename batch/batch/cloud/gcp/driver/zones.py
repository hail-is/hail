import collections
import logging
import random
from typing import Any, Dict, List, Set, Tuple

from hailtop.aiocloud import aiogoogle
from hailtop.utils import url_basename

from ....driver.exceptions import RegionsNotSupportedError
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
        machine_family_valid_zones = await fetch_machine_valid_zones(compute_client, regions)
        return ZoneMonitor(compute_client, region_info, zones, regions, default_zone, machine_family_valid_zones)

    def __init__(
        self,
        compute_client: aiogoogle.GoogleComputeClient,  # BORROWED
        initial_region_info: Dict[str, Dict[str, Any]],
        initial_zones: List[str],
        regions: Set[str],
        default_zone: str,
        machine_family_valid_zones: Dict[str, Set[str]],
    ):
        self._compute_client = compute_client
        self._region_info: Dict[str, Dict[str, Any]] = initial_region_info
        self._regions = regions
        self.zones: List[str] = initial_zones
        self._default_zone = default_zone
        self._machine_family_valid_zones: Dict[str, Set[str]] = machine_family_valid_zones

        self.zone_success_rate = ZoneSuccessRate()

    @property
    def region_quotas(self):
        return self._region_info

    def default_location(self) -> str:
        return self._default_zone

    def choose_location(
        self,
        cores: int,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        preemptible: bool,
        regions: List[str],
        machine_type: str,
    ) -> str:
        zone_weights = self.compute_zone_weights(cores, local_ssd_data_disk, data_disk_size_gb, preemptible, regions)

        zones = [zw.zone for zw in zone_weights]
        machine_family = machine_type.split("-")[0]
        if machine_family in self._machine_family_valid_zones:
            valid_zones = self._machine_family_valid_zones[machine_family]
            zones = [z for z in zones if z in valid_zones]
            zone_weights = [zw for zw in zone_weights if zw.zone in valid_zones]
        else:
            zones = []
            zone_weights = []

        if len(zones) == 0:
            raise RegionsNotSupportedError(regions, self._regions)

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
        regions: List[str],
    ) -> List[ZoneWeight]:
        weights = []
        for region_name, r in self._region_info.items():
            if region_name not in regions:
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


async def fetch_machine_valid_zones(
    compute_client: aiogoogle.GoogleComputeClient, regions: Set[str]
) -> Dict[str, Set[str]]:
    region_info = {name: await compute_client.get(f'/regions/{name}') for name in regions}
    zones = [url_basename(z) for r in region_info.values() for z in r['zones']]
    machine_family_valid_zones = collections.defaultdict(set)
    for zone in zones:
        async for machine_type in await compute_client.list(f'/zones/{zone}/machineTypes'):
            machine_family = machine_type['name'].split('-')[0]
            machine_family_valid_zones[machine_family].add(machine_type['zone'])
    return machine_family_valid_zones


async def fetch_region_quotas(
    compute_client: aiogoogle.GoogleComputeClient, regions: Set[str]
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    region_info = {name: await compute_client.get(f'/regions/{name}') for name in regions}
    zones = [url_basename(z) for r in region_info.values() for z in r['zones']]
    return region_info, zones
