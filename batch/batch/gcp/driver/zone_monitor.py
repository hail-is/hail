import logging

from hailtop.utils import url_basename

from ...batch_configuration import BATCH_GCP_REGIONS
from ...driver.zone_monitor import BaseZoneMonitor, ZoneWeight

log = logging.getLogger('zone_monitor')


class ZoneMonitor(BaseZoneMonitor):
    def _zone_weights(self, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb):
        zone_weights = []
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
                zone_weights.append(ZoneWeight(zone_name, weight))

        log.info(f'zone_weights {zone_weights}')
        return zone_weights

    async def update_region_quotas(self):
        self.region_info = {name: await self.compute_manager.compute_client.get(f'/regions/{name}') for name in BATCH_GCP_REGIONS}

        self.zones = [url_basename(z) for r in self.region_info.values() for z in r['zones']]

        log.info('updated region quotas')
