import abc
import logging

from .utils.java import Env


class Logger(abc.ABC):
    @abc.abstractmethod
    def error(self, msg):
        pass

    @abc.abstractmethod
    def warn(self, msg):
        pass

    @abc.abstractmethod
    def info(self, msg):
        pass


class PythonOnlyLogger:
    def __init__(self):
        self.logger = logging.getLogger().getChild("hail")
        self.logger.setLevel(logging.INFO)

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def info(self, msg):
        self.logger.info(msg)


class Log4jLogger:
    log_pkg = None

    @staticmethod
    def get():
        if Log4jLogger.log_pkg is None:
            Log4jLogger.log_pkg = Env.jutils()
        return Log4jLogger.log_pkg

    def error(self, msg):  # pylint: disable=no-self-use
        Log4jLogger.get().error(msg)

    def warning(self, msg):  # pylint: disable=no-self-use
        Log4jLogger.get().warn(msg)

    def info(self, msg):  # pylint: disable=no-self-use
        Log4jLogger.get().info(msg)
