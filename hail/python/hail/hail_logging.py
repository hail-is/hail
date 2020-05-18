import abc
import logging


class Logger(abc.ABC):
    @abc.abstractmethod
    def configure(self):
        pass

    @abc.abstractmethod
    def error(self, msg):
        pass

    @abc.abstractmethod
    def warning(self, msg):
        pass

    @abc.abstractmethod
    def info(self, msg):
        pass


class PythonOnlyLogger(Logger):
    def __init__(self):
        self.logger = logging.getLogger("hail")
        self.logger.setLevel(logging.INFO)

    def configure(self):  # pylint: disable=no-self-use
        logging.basicConfig()

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def info(self, msg):
        self.logger.info(msg)
