import abc
import logging
import random
from typing import TYPE_CHECKING

from hailtop import aiotools
from hailtop.utils import periodically_call

from ..utils import WindowFractionCounter
from ..batch_configuration import GCP_ZONE
if TYPE_CHECKING:
    from .compute_manager import BaseComputeManager  # pylint: disable=cyclic-import


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


class BaseZoneMonitor(abc.ABC):
    def __init__(self, compute_manager: 'BaseComputeManager'):
        self.compute_manager = compute_manager
        self.app = compute_manager.app
        self.zone_success_rate = ZoneSuccessRate()
        self.region_info = None
        self.zones = []
        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(self.update_region_quotas_loop())

    async def shutdown(self):
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

    @abc.abstractmethod
    def _zone_weights(self, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb):
        pass

    def zone_weights(self, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb):
        if not self.region_info:
            return None
        return self._zone_weights(worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb)

    @abc.abstractmethod
    async def update_region_quotas(self):
        pass

    async def update_region_quotas_loop(self):
        await periodically_call(60, self.update_region_quotas)
