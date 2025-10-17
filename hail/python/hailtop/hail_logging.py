import datetime
import logging
import sys
from time import timezone

import orjson
from aiohttp.abc import AbstractAccessLogger
from pythonjsonlogger import jsonlogger


class OrJsonEncoder:
    def __init__(self, *args, **kwargs):
        del args
        del kwargs


def logger_json_serializer(log_record, default=None, cls=None, indent=None, ensure_ascii=False) -> str:
    assert default is None and cls is OrJsonEncoder and indent is None and ensure_ascii is False, (
        default,
        cls,
        indent,
        ensure_ascii,
    )
    return orjson.dumps(log_record).decode('utf-8')


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def __init__(self, format_string):
        super().__init__(
            format_string, json_encoder=OrJsonEncoder, json_serializer=logger_json_serializer, json_ensure_ascii=False
        )

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        # GCP Logging expects `severity` but jsonlogger uses `levelname`
        log_record['severity'] = record.levelname
        log_record['funcNameAndLine'] = f"{record.funcName}:{record.lineno}"
        log_record['hail_log'] = 1


def configure_logging(level=logging.INFO):
    fmt = CustomJsonFormatter('%(severity)s %(levelname)s %(asctime)s %(filename)s %(funcNameAndLine)s %(message)s')

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(fmt)

    logging.basicConfig(handlers=[stream_handler], level=level)


class AccessLogger(AbstractAccessLogger):
    def log(self, request, response, time):
        tz = datetime.timezone(datetime.timedelta(seconds=-timezone))
        now = datetime.datetime.now(tz)
        start_time = now - datetime.timedelta(seconds=time)
        start_time_str = start_time.strftime('[%d/%b/%Y:%H:%M:%S %z]')

        extra = {
            'remote_address': request.remote,
            'request_start_time': start_time_str,
            'request_duration': time,
            'response_status': response.status,
            'x_real_ip': request.headers.get("X-Real-IP"),
            'user_agent': request.headers.get("User-Agent"),
            'request_method': request.method,
            'request_scheme': request.scheme,
            'query_params': dict(request.query),
        }

        userdata_maybe = request.get('userdata', {})
        userdata_keys = ['username', 'login_id', 'is_developer', 'hail_identity']
        extra.update({k: userdata_maybe[k] for k in userdata_keys if k in userdata_maybe})

        # Try to get formdata in various ways:
        request_data_source = []
        request_data = request.get('request_data', {})
        if len(request_data) > 0:
            request_data_source.append('request_data')

        formdata = request.get('formdata', {})
        if len(formdata) > 0:
            request_data_source.append('formdata')
        request_data.update(formdata)

        post_data = request._post or {}
        if len(post_data) > 0:
            request_data_source.append('post_data')
        request_data.update(post_data)

        extra.update({'request_data_source': request_data_source})
        extra.update({'request_data': {k: request_data[k] for k in request_data.keys() if not k.startswith('_')}})

        api_info_maybe = request.get('api_info', {})
        api_info_keys_and_defaults = {
            'authenticated_users_only': False,
            'developers_only': False,
        }

        for k, default in api_info_keys_and_defaults.items():
            if k in api_info_maybe:
                extra[k] = api_info_maybe[k]
            else:
                extra[k] = default

        self.logger.info(
            f'{request.scheme} {request.method} {request.path} done in {time}s: {response.status}',
            extra={**extra, **request.get('batch_telemetry', {})},
        )
