from aiohttp import web

from hailtop.gear import get_deploy_config

from .utils import jsonify


app = web.Application()
routes = web.RouteTableDef()


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return jsonify({})


app.add_routes(routes)
deploy_config = get_deploy_config()
web.run_app(deploy_config.prefix_application(app, 'batch2'), host='0.0.0.0', port=5000)
