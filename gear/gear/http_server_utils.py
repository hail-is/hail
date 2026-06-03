from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Optional

import orjson
from aiohttp import web

from hailtop import __version__

from .system_permissions import SystemPermission

if TYPE_CHECKING:
    from .auth import UserData


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


def version_response(userdata: Optional['UserData']) -> web.Response:
    is_sysadmin = userdata is not None and userdata['system_permissions'].get(
        SystemPermission.READ_DEPLOYED_SYSTEM_STATE, False
    )
    version = __version__ if is_sysadmin else __version__.split('-', maxsplit=1)[0]
    return web.Response(text=version)


def json_response(
    data: Any, fallback_serializer: Optional[Callable[[Any], Any]] = None, headers: Optional[Mapping] = None
) -> web.Response:
    return web.json_response(body=orjson.dumps(data, default=fallback_serializer), headers=headers)
