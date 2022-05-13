from aiohttp import web

from hailtop.config import get_deploy_config
from hailtop.hail_logging import AccessLogger
from hailtop.tls import internal_server_ssl_context
from gear import setup_aiohttp_session

deploy_config = get_deploy_config()

routes = web.RouteTableDef()

@routes.get('/')
def hello(request):
    return web.Response(text="hello")

def run():
    app = web.Application()
    setup_aiohttp_session(app)

    app.add_routes(routes)

    web.run_app(
        deploy_config.prefix_application(app, 'batch-db-proxy'),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=internal_server_ssl_context(),
    )
