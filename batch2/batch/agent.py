import docker
import uvloop
from aiohttp import web

from .utils import jsonify

uvloop.install()

dc = docker.from_env()

app = web.Application()
routes = web.RouteTableDef()


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return jsonify({})


@routes.get('/helloworld')
async def run_helloworld(request):  # pylint: disable=W0613
    container = dc.containers.run('hello-world', detach=True)
    return jsonify({'name': container.name})


app.add_routes(routes)
web.run_app(app, host='0.0.0.0', port=5000)
