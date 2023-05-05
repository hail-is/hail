from typing import Any, Callable, Optional
import orjson
from aiohttp import web


async def json_request(request: web.Request) -> Any:
    return orjson.loads(await request.read())


def json_response(a: Any, fallback_serializer: Optional[Callable[[Any], Any]] = None) -> web.Response:
    return web.json_response(body=orjson.dumps(a, default=fallback_serializer))
