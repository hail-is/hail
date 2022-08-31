import abc
from typing import Optional


class CloudLocationMonitor(abc.ABC):
    @abc.abstractmethod
    def default_location(self) -> str:
        raise NotImplementedError

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
