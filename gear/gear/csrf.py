import secrets
import logging
from functools import wraps
from aiohttp import web
import aiohttp_jinja2

log = logging.getLogger('gear.auth')


def new_csrf_token():
    return secrets.token_urlsafe(64)


def render_template(service, request, userdata, file, page_context):
    if '_csrf' in request.cookies:
        csrf_token = request.cookies['_csrf']
    else:
        csrf_token = new_csrf_token()

    context = base_context(deploy_config, session, userdata, service)
    context.update(page_context)
    context['csrf_token'] = csrf_token

    response = aiohttp_jinja2.render_template(file, request, context)
    response.set_cookie('_csrf', csrf_token, secure=True, httponly=True)
    return response


def check_csrf_token(fun):
    @wraps(fun)
    async def wrapped(request, *args, **kwargs):
        token1 = request.cookies.get('_csrf')
        post = await request.post()
        token2 = post.get('_csrf')
        if token1 is not None and token2 is not None and token1 == token2:
            return await fun(request, *args, **kwargs)
        log.info('request made with invalid csrf tokens')
        raise web.HTTPForbidden()
    return wrapped
