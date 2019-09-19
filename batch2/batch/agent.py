import uvloop
from aiohttp import web

from hailtop.config import get_deploy_config
from gear import configure_logging

uvloop.install()

configure_logging()

app = web.Application()
routes = web.RouteTableDef()


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


app.add_routes(routes)
web.run_app(app, host='0.0.0.0', port=5000)
