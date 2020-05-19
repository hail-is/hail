import abc
import logging


class Logger(abc.ABC):
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
    def __init__(self, skip_logging_configuration=False):
        self.logger = logging.getLogger("hail")
        self.logger.setLevel(logging.INFO)
        if not skip_logging_configuration:
            logging.basicConfig()

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def info(self, msg):
        self.logger.info(msg)
