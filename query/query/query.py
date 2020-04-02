import sys
import asyncio
import uvloop
from aiohttp import web
from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway
from hailtop.config import get_deploy_config
from gear import setup_aiohttp_session

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


async def on_startup(app):
    port = launch_gateway(
        jarpath='/spark-2.4.0-bin-hadoop2.7/jars/py4j-0.10.7.jar',
        classpath='/spark-2.4.0-bin-hadoop2.7/jars/*:/hail.jar',
        die_on_exit=True)
    gateway = JavaGateway(
        gateway_parameters=GatewayParameters(port=port),
        auto_convert=True)
    app['gateway'] = gateway

    jbackend = getattr(gateway.jvm, 'is').hail.backend.service.ServiceBackend.apply()
    app['jbackend'] = jbackend


def run():
    app = web.Application()

    setup_aiohttp_session(app)

    app.add_routes(routes)

    app.on_startup.append(on_startup)

    web.run_app(
        deploy_config.prefix_application(app, 'query'),
        host='0.0.0.0',
        port=5000)
