import uvloop
from aiohttp import web
import hail as hl
from gear import setup_aiohttp_session

uvloop.install()

routes = web.RouteTableDef()

deploy_config = get_deploy_config()


@routes.get('/healthcheck')
async def healthcheck(request):
    return web.Response()


def run():
    app = web.Application()
    setup_aiohttp_session(app)
    app.add_routes(routes)
    web.run_app(
        deploy_config.prefix_application(app, 'query'),
        host='0.0.0.0',
        port=5000)
