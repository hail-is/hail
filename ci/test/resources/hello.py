import os

from aiohttp import web

from gear import setup_aiohttp_session
from hailtop.config import get_deploy_config
from hailtop.tls import internal_server_ssl_context

deploy_config = get_deploy_config()

app = web.Application()
routes = web.RouteTableDef()

SHA = os.environ['HAIL_SHA']


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=unused-argument
    return web.Response()


@routes.get('/sha')
async def get_sha(request):  # pylint: disable=unused-argument
    return web.Response(text=SHA)


setup_aiohttp_session(app)
app.add_routes(routes)
web.run_app(
    deploy_config.prefix_application(app, 'hello'), host='0.0.0.0', port=5000, ssl_context=internal_server_ssl_context()
)
