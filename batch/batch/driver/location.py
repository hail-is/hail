import abc
from typing import Optional


class CloudLocationMonitor(abc.ABC):
    @abc.abstractmethod
    def choose_location(
        self,
        cores: int,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        preemptible: bool,
        region: Optional[str],
    ) -> str:
        raise NotImplementedError
