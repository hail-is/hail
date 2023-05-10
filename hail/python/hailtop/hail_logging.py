import datetime
import logging
import sys
from time import timezone

import orjson
from aiohttp.abc import AbstractAccessLogger
from pythonjsonlogger import jsonlogger


def logger_json_serializer(log_record,
                           default=None,
                           cls=None,
                           indent=None,
                           ensure_ascii=False) -> str:
    assert default is None and cls is None and indent is None and ensure_ascii is False
    return orjson.dumps(log_record).decode('utf-8')


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def __init__(self, format_string):
        super().__init__(format_string, json_serializer=logger_json_serializer, json_ensure_ascii=False)

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        # GCP Logging expects `severity` but jsonlogger uses `levelname`
        log_record['severity'] = record.levelname
        log_record['funcNameAndLine'] = f"{record.funcName}:{record.lineno}"
        log_record['hail_log'] = 1


def configure_logging():
    fmt = CustomJsonFormatter('%(severity)s %(levelname)s %(asctime)s %(filename)s %(funcNameAndLine)s %(message)s')

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logging.basicConfig(handlers=[stream_handler], level=logging.INFO)


class AccessLogger(AbstractAccessLogger):
    def log(self, request, response, time):
        tz = datetime.timezone(datetime.timedelta(seconds=-timezone))
        now = datetime.datetime.now(tz)
        start_time = now - datetime.timedelta(seconds=time)
        start_time_str = start_time.strftime('[%d/%b/%Y:%H:%M:%S %z]')
        self.logger.info(f'{request.scheme} {request.method} {request.path} '
                         f'done in {time}s: {response.status}',
                         extra={'remote_address': request.remote,
                                'request_start_time': start_time_str,
                                'request_duration': time,
                                'response_status': response.status,
                                'x_real_ip': request.headers.get("X-Real-IP")})
