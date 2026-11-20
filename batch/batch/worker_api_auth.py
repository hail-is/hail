"""Authentication for the worker API (`:5000`): callers present the instance's token, the same
per-instance secret the worker presents to the driver."""

import hmac
from typing import Callable, Dict, FrozenSet, Optional

from aiohttp import web
from aiohttp.typedefs import Handler, Middleware

INSTANCE_TOKEN_HEADER = 'X-Hail-Instance-Token'


def worker_api_headers(instance_token: str) -> Dict[str, str]:
    return {INSTANCE_TOKEN_HEADER: instance_token}


def instance_token_middleware(
    get_instance_token: Callable[[], Optional[str]], unauthenticated_paths: FrozenSet[str]
) -> Middleware:
    @web.middleware
    async def middleware(request: web.Request, handler: Handler) -> web.StreamResponse:
        if request.path not in unauthenticated_paths:
            expected = get_instance_token()
            provided = request.headers.get(INSTANCE_TOKEN_HEADER)
            if expected is None or provided is None or not hmac.compare_digest(provided.encode(), expected.encode()):
                raise web.HTTPUnauthorized()
        return await handler(request)

    return middleware
