from collections.abc import Mapping
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import orjson
from aiohttp import web

from hailtop.config import get_deploy_config


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


def validate_redirect_url(next_page: Optional[str]) -> str:
    if not next_page:
        raise web.HTTPBadRequest(text='Invalid next page')
    deploy_config = get_deploy_config()
    valid_next_services = ['batch', 'auth', 'ci', 'monitoring']
    valid_next_domains = [urlparse(deploy_config.external_url(s, '/')).netloc for s in valid_next_services]
    actual_next_page_domain = urlparse(next_page).netloc
    if actual_next_page_domain not in valid_next_domains:
        raise web.HTTPBadRequest(text='Invalid next page')
    return next_page


def json_response(
    data: Any, fallback_serializer: Optional[Callable[[Any], Any]] = None, headers: Optional[Mapping] = None
) -> web.Response:
    return web.json_response(body=orjson.dumps(data, default=fallback_serializer), headers=headers)
