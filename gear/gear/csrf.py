import logging
import secrets
from functools import wraps

from aiohttp import web

log = logging.getLogger('gear.auth')


def new_csrf_token():
    return secrets.token_urlsafe(64)


def check_csrf_token(fun):
    @wraps(fun)
    async def wrapped(request, *args, **kwargs):
        token1 = request.cookies.get('_csrf')
        post = await request.post()
        token2 = post.get('_csrf')
        if token1 is not None and token2 is not None and token1 == token2:
            return await fun(request, *args, **kwargs)
        log.info('request made with invalid csrf tokens')
        raise web.HTTPUnauthorized()

    return wrapped
