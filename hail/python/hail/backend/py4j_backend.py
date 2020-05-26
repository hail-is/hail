import abc
from .backend import Backend


class Py4JBackend(Backend):
    @abc.abstractmethod
    def jvm(self):
        pass

    @abc.abstractmethod
    def hail_package(self):
        pass

    @abc.abstractmethod
    def utils_package_object(self):
        pass
