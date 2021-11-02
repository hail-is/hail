from ....driver.location import CloudLocationMonitor


class RegionMonitor(CloudLocationMonitor):
    @staticmethod
    async def create(default_region: str) -> 'RegionMonitor':
        return RegionMonitor(default_region)

    def __init__(self, default_region: str):
        self._default_region = default_region

    def default_location(self) -> str:
        return self._default_region

    def choose_location(self,
                        worker_cores: int,  # pylint: disable=unused-argument
                        local_ssd_data_disk: bool,  # pylint: disable=unused-argument
                        data_disk_size_gb: int  # pylint: disable=unused-argument
                        ) -> str:
        return self._default_region
