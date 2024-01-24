import os

from hailtop import httpx
from hailtop.config import get_deploy_config
from hailtop.auth import hail_credentials
from hailtop.aiocloud.common import Session

from web_common import render_template, setup_aiohttp_jinja2, setup_common_static_routes

from aiohttp import web
import aiohttp_session
from aiohttp_session.cookie_storage import EncryptedCookieStorage

routes = web.RouteTableDef()
setup_common_static_routes(routes)

deploy_config = get_deploy_config()

SERVICE = os.environ['SERVICE']
MODULES = {'batch': 'batch.front_end', 'batch-driver': 'batch.driver'}
BC = web.AppKey('backend_client', Session)


@routes.get('/api/{route:.*}')
async def default_proxied_api_route(request: web.Request):
    return web.json_response(await proxy(request))


@routes.get('/{route:.*}')
async def default_proxied_web_route(request: web.Request):
    return await render_html(request, await proxy(request))


async def proxy(request: web.Request):
    backend_client = request.app[BC]
    backend_route = deploy_config.external_url(SERVICE, request.raw_path)
    headers = {'x-hail-return-jinja-context': '1'}
    try:
        async with await backend_client.request(request.method, backend_route, headers=headers) as resp:
            return await resp.json()
    except httpx.ClientResponseError as e:
        if e.status == 404:
            raise web.HTTPNotFound()
        raise


async def render_html(request: web.Request, context: dict):
    # Make links point back to the local dev server and not use
    # the dev namespace path rewrite shenanigans.
    context['page_context']['base_path'] = ''
    return await render_template(SERVICE, request, **context)


async def on_startup(app: web.Application):
    app[BC] = Session(credentials=hail_credentials())


async def on_cleanup(app: web.Application):
    await app[BC].close()


app = web.Application()
setup_aiohttp_jinja2(app, MODULES.get(SERVICE, SERVICE))
app.add_routes(routes)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)
aiohttp_session.setup(app, EncryptedCookieStorage(b'Thirty  two  length  bytes  key.'))

if __name__ == '__main__':
    web.run_app(app)
