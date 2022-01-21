import abc


class CloudLocationMonitor(abc.ABC):
    @abc.abstractmethod
    def default_location(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def choose_location(self, cores: int, local_ssd_data_disk: bool, data_disk_size_gb: int) -> str:
        raise NotImplementedError
