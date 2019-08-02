from aiohttp import web

from .utils import jsonify


app = web.Application()
routes = web.RouteTableDef()


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return jsonify({})


app.add_routes(routes)
web.run_app(app, host='0.0.0.0', port=5000)