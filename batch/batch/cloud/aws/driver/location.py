from typing import List

from ....driver.location import CloudLocationMonitor


class AWSLocationMonitor(CloudLocationMonitor):
    def __init__(self, default_region: str):
        self._default_region = default_region

    def default_location(self) -> str:
        return self._default_region

    def choose_location(
        self,
        cores: int,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        preemptible: bool,
        regions: List[str],
        machine_type: str,
    ) -> str:
        return self._default_region
