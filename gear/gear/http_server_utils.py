from collections.abc import Mapping
from typing import Any, Callable, Optional

import orjson
from aiohttp import web


async def json_request(request: web.Request) -> Any:
    result = orjson.loads(await request.read())
    if 'request_data' not in request:
        request['request_data'] = {}

    # Handle both dictionaries and lists
    if isinstance(result, dict):
        request['request_data'].update(result)
    elif isinstance(result, list):
        request['request_data'].update({'list_data': result})
    else:
        request['request_data'].update({'raw_data': result})

    return result


def json_response(
    data: Any, fallback_serializer: Optional[Callable[[Any], Any]] = None, headers: Optional[Mapping] = None
) -> web.Response:
    return web.json_response(body=orjson.dumps(data, default=fallback_serializer), headers=headers)
