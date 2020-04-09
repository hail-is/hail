import os
import base64
import concurrent
import logging
import uvloop
from aiohttp import web
import kubernetes_asyncio as kube
from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway
from hailtop.utils import blocking_to_async, retry_transient_errors
from hailtop.config import get_deploy_config
from hailtop.ssl import get_ssl_context
from gear import setup_aiohttp_session, rest_authenticated_users_only

uvloop.install()

BATCH_PODS_NAMESPACE = os.environ['HAIL_BATCH_PODS_NAMESPACE']
log = logging.getLogger('batch')
routes = web.RouteTableDef()


async def add_user(app, userdata):
    username = userdata['username']
    users = app['users']
    if username in users:
        return

    jbackend = app['jbackend']
    k8s_client = app['k8s_client']
    gsa_key_secret = await retry_transient_errors(
        k8s_client.read_namespaced_secret,
        userdata['gsa_key_secret_name'],
        BATCH_PODS_NAMESPACE,
        _request_timeout=5.0)
    gsa_key = base64.b64decode(gsa_key_secret.data['key.json']).decode()
    jbackend.addUser(username, gsa_key)
    users.add(username)


@routes.get('/healthcheck')
async def healthcheck(request):  # pylint: disable=unused-argument
    return web.Response()


def blocking_execute(jbackend, username, code):
    return jbackend.execute(username, code)


@routes.post('/execute')
@rest_authenticated_users_only
async def execute(request, userdata):
    app = request.app
    thread_pool = app['thread_pool']
    jbackend = app['jbackend']
    code = await request.json()
    log.info(f'execute: {code}')
    await add_user(app, userdata)
    result = await blocking_to_async(thread_pool, blocking_execute, jbackend, userdata['username'], code)
    log.info(f'result: {result}')
    return web.json_response(text=result)


def blocking_value_type(jbackend, username, code):
    return jbackend.valueType(username, code)


@routes.post('/type/value')
@rest_authenticated_users_only
async def value_type(request, userdata):
    app = request.app
    thread_pool = app['thread_pool']
    jbackend = app['jbackend']
    code = await request.json()
    log.info(f'value type: {code}')
    await add_user(app, userdata)
    result = await blocking_to_async(thread_pool, blocking_value_type, jbackend, userdata['username'], code)
    log.info(f'result: {result}')
    return web.json_response(text=result)


def blocking_table_type(jbackend, username, code):
    return jbackend.tableType(username, code)


@routes.post('/type/table')
@rest_authenticated_users_only
async def table_type(request, userdata):
    app = request.app
    thread_pool = app['thread_pool']
    jbackend = app['jbackend']
    code = await request.json()
    log.info(f'table type: {code}')
    await add_user(app, userdata)
    result = await blocking_to_async(thread_pool, blocking_table_type, jbackend, userdata['username'], code)
    log.info(f'result: {result}')
    return web.json_response(text=result)


def blocking_matrix_type(jbackend, username, code):
    return jbackend.matrixTableType(username, code)


@routes.post('/type/matrix')
@rest_authenticated_users_only
async def matrix_type(request, userdata):
    app = request.app
    thread_pool = app['thread_pool']
    jbackend = app['jbackend']
    code = await request.json()
    log.info(f'matrix type: {code}')
    await add_user(app, userdata)
    result = await blocking_to_async(thread_pool, blocking_matrix_type, jbackend, userdata['username'], code)
    log.info(f'result: {result}')
    return web.json_response(text=result)


def blocking_blockmatrix_type(jbackend, username, code):
    return jbackend.blockMatrixType(username, code)


@routes.post('/type/blockmatrix')
@rest_authenticated_users_only
async def blockmatrix_type(request, userdata):
    app = request.app
    thread_pool = app['thread_pool']
    jbackend = app['jbackend']
    code = await request.json()
    log.info(f'blockmatrix type: {code}')
    await add_user(app, userdata)
    result = await blocking_to_async(thread_pool, blocking_blockmatrix_type, jbackend, userdata['username'], code)
    log.info(f'result: {result}')
    return web.json_response(text=result)


def blocking_get_reference(app, data):
    hail_pkg = app['hail_pkg']
    return hail_pkg.variant.ReferenceGenome.getReference(data['name']).toJSONString()


@routes.get('/references/get')
@rest_authenticated_users_only
async def get_reference(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    thread_pool = app['thread_pool']
    data = await request.json()
    result = await blocking_to_async(thread_pool, blocking_get_reference, app, data)
    return web.json_response(text=result)


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

    app['users'] = set()

    kube.config.load_incluster_config()
    k8s_client = kube.client.CoreV1Api()
    app['k8s_client'] = k8s_client


def run():
    app = web.Application()

    setup_aiohttp_session(app)

    app.add_routes(routes)

    app.on_startup.append(on_startup)

    deploy_config = get_deploy_config()
    web.run_app(
        deploy_config.prefix_application(app, 'query'),
        host='0.0.0.0',
        port=5000,
        ssl_context=get_ssl_context())
