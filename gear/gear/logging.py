import logging
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['funcNameAndLine'] = "{}:{}".format(record.funcName, record.lineno)
        log_record['hail_log'] = 1


def configure_logging():
    fmt = CustomJsonFormatter('(levelname) (asctime) (filename) (funcNameAndLine) (message)')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logging.basicConfig(handlers=[stream_handler], level=logging.INFO)
