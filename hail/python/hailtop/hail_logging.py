import logging
import sys
from pythonjsonlogger import jsonlogger
from aiohttp.abc import AbstractAccessLogger
import datetime
import time


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        # GCP Logging expects `severity` but jsonlogger uses `levelname`
        log_record['severity'] = record.levelname
        log_record['funcNameAndLine'] = "{}:{}".format(record.funcName, record.lineno)
        log_record['hail_log'] = 1


def configure_logging():
    fmt = CustomJsonFormatter('(severity) (levelname) (asctime) (filename) (funcNameAndLine) (message)')

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logging.basicConfig(handlers=[stream_handler], level=logging.INFO)


class AccessLogger(AbstractAccessLogger):
    def log(self, request, response, duration):
        tz = datetime.timezone(datetime.timedelta(seconds=-time.timezone))
        now = datetime.datetime.now(tz)
        start_time = now - datetime.timedelta(seconds=duration)
        start_time_str = start_time.strftime('[%d/%b/%Y:%H:%M:%S %z]')
        self.logger.info(f'{request.scheme} {request.method} {request.path} '
                         f'done in {duration}s: {response.status}',
                         extra={'remote_address': request.remote,
                                'request_start_time': start_time_str,
                                'request_duration': duration,
                                'response_status': response.status,
                                'x_real_ip': request.headers.get("X-Real-IP")})
