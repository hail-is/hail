import logging
import secrets

from aiohttp import web

from .auth import AIOHTTPHandler

log = logging.getLogger('gear.auth')


def new_csrf_token():
    return secrets.token_urlsafe(64)


@web.middleware
async def check_csrf_token(request: web.Request, handler: AIOHTTPHandler):
    # CSRF prevention is only relevant to requests that use browser
    # cookies for authentication.
    if request.cookies and request.method in request.POST_METHODS:
        token1 = request.cookies.get('_csrf')
        post = await request.post()
        token2 = post.get('_csrf')
        if token1 is None or token2 is None or token1 != token2:
            log.info('request made with invalid csrf tokens')
            raise web.HTTPUnauthorized()

    return await handler(request)
