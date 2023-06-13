import datetime
import logging
import sys
from time import timezone

import aiohttp
import orjson
from aiohttp.abc import AbstractAccessLogger
from pythonjsonlogger import jsonlogger


class OrJsonEncoder:
    def __init__(self, *args, **kwargs):
        del args
        del kwargs


def logger_json_serializer(log_record,
                           default=None,
                           cls=None,
                           indent=None,
                           ensure_ascii=False) -> str:
    assert default is None and cls is OrJsonEncoder and indent is None and ensure_ascii is False, (default, cls, indent, ensure_ascii)
    return orjson.dumps(log_record).decode('utf-8')


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def __init__(self, format_string):
        super().__init__(format_string, json_encoder=OrJsonEncoder, json_serializer=logger_json_serializer, json_ensure_ascii=False)

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
    path_to_operation = {}

    def log(self, request, response, time):
        tz = datetime.timezone(datetime.timedelta(seconds=-timezone))
        now = datetime.datetime.now(tz)
        start_time = now - datetime.timedelta(seconds=time)
        start_time_str = start_time.strftime('[%d/%b/%Y:%H:%M:%S %z]')
        batch_identifier = request.get('batch_identifier')
        job_identifier = request.get('job_identifier')
        batch_operation = request.get('batch_operation')
        job_queue_time = request.get('job_queue_time')

        extra = {
            'remote_address': request.remote,
            'request_start_time': start_time_str,
            'request_duration': time,
            'response_status': response.status,
            'x_real_ip': request.headers.get("X-Real-IP")
        }

        if batch_identifier:
            extra['batch_identifier'] = batch_identifier
        if job_identifier:
            extra['job_identifier'] = job_identifier
        if batch_operation:
            extra['batch_operation'] = batch_operation
        if job_queue_time:
            extra['job_queue_time'] = job_queue_time

        self.logger.info(f'{request.scheme} {request.method} {request.path} '
                         f'done in {time}s: {response.status}',
                         extra=extra)
