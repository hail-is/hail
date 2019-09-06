from aiohttp import web

from hailtop.gear import configure_logging, get_deploy_config

configure_logging()

app = web.Application()
routes = web.RouteTableDef()


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


app.add_routes(routes)
deploy_config = get_deploy_config()
web.run_app(deploy_config.prefix_application(app, 'batch2'), host='0.0.0.0', port=5000)
