import abc


class CloudLocationMonitor(abc.ABC):
    @abc.abstractmethod
    def default_location(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def choose_location(self,
                        cores: int,
                        worker_local_ssd_data_disk: bool,
                        worker_pd_ssd_data_disk_size_gb: int) -> str:
        raise NotImplementedError
