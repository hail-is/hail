import json
import concurrent
import uvloop
from aiohttp import web
from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway
from hailtop.utils import blocking_to_async
from hailtop.config import get_deploy_config
from gear import setup_aiohttp_session, rest_authenticated_users_only

uvloop.install()

routes = web.RouteTableDef()

deploy_config = get_deploy_config()


@routes.get('/healthcheck')
async def healthcheck(request):  # pylint: disable=unused-argument
    return web.Response()


@routes.get('/request')
async def request(request):
    jbackend = request.app['jbackend']
    resp = jbackend.request()
    return web.json_response({'value': resp})


def blocking_get_reference(app, data):
    hail_pkg = app['hail_pkg']
    return json.loads(
        hail_pkg.variant.ReferenceGenome.getReference(data['name']).toJSONString())


@routes.get('/references/get')
@rest_authenticated_users_only
async def get_reference(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    thread_pool = app['thread_pool']
    data = await request.json()
    # FIXME error handling
    result = await blocking_to_async(thread_pool, blocking_get_reference, app, data)
    return web.json_response(result)


async def on_startup(app):
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    app['thread_pool'] = thread_pool

    port = launch_gateway(
        jarpath='/spark-2.4.0-bin-hadoop2.7/jars/py4j-0.10.7.jar',
        classpath='/spark-2.4.0-bin-hadoop2.7/jars/*:/hail.jar',
        die_on_exit=True)
    gateway = JavaGateway(
        gateway_parameters=GatewayParameters(port=port),
        auto_convert=True)
    app['gateway'] = gateway

    hail_pkg = getattr(gateway.jvm, 'is').hail
    app['hail_pkg'] = hail_pkg

    jbackend = hail_pkg.backend.service.ServiceBackend.apply()
    app['jbackend'] = jbackend

    jhc = hail_pkg.HailContext.apply(
        jbackend, 'hail.log', False, False, 50, "/tmp", 3)
    app['jhc'] = jhc


def run():
    app = web.Application()

    setup_aiohttp_session(app)

    app.add_routes(routes)

    app.on_startup.append(on_startup)

    web.run_app(
        deploy_config.prefix_application(app, 'query'),
        host='0.0.0.0',
        port=5000)
