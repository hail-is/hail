from typing import Optional

from ....driver.location import CloudLocationMonitor


class RegionMonitor(CloudLocationMonitor):
    @staticmethod
    async def create(region: str) -> 'RegionMonitor':
        return RegionMonitor(region)

    def __init__(self, region: str):
        self._region = region

    def choose_location(
        self,
        cores: int,  # pylint: disable=unused-argument
        local_ssd_data_disk: bool,  # pylint: disable=unused-argument
        data_disk_size_gb: int,  # pylint: disable=unused-argument
        preemptible: bool,  # pylint: disable=unused-argument
        region: Optional[str],  # pylint: disable=unused-argument
    ) -> str:
        return region or self._region
