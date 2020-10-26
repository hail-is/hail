import os
import base64
import concurrent
import logging
import uvloop
import asyncio
import aiohttp
from aiohttp import web
import kubernetes_asyncio as kube
from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway
from hailtop.utils import blocking_to_async, retry_transient_errors
from hailtop.config import get_deploy_config
from hailtop.tls import get_in_cluster_server_ssl_context
from hailtop.hail_logging import AccessLogger
from gear import setup_aiohttp_session, rest_authenticated_users_only, rest_authenticated_developers_only

uvloop.install()

BATCH_PODS_NAMESPACE = os.environ['HAIL_BATCH_PODS_NAMESPACE']
log = logging.getLogger('batch')
routes = web.RouteTableDef()


def java_to_web_response(jresp):
    status = jresp.status()
    value = jresp.value()
    log.info(f'response status {status} value {value}')
    if status in (400, 500):
        return web.Response(status=status, text=value)
    assert status == 200, status
    return web.json_response(status=status, text=value)


async def send_ws_response(thread_pool, ws, f, *args, **kwargs):
    jresp = await blocking_to_async(thread_pool, f, *args, **kwargs)
    status = jresp.status()
    value = jresp.value()
    log.info(f'response status {status} value {value}')
    if status in (400, 500):
        await ws.send_json({'error': value})
    else:
        assert status == 200, status
        await ws.send_json({'status': status, 'result': value})


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


def blocking_execute(jbackend, userdata, body):
    return jbackend.execute(userdata['username'], userdata['session_id'], body['billing_project'], body['bucket'], body['code'])


def blocking_value_type(jbackend, userdata, body):
    return jbackend.valueType(userdata['username'], body['code'])


def blocking_table_type(jbackend, userdata, body):
    return jbackend.tableType(userdata['username'], body['code'])


def blocking_matrix_type(jbackend, userdata, body):
    return jbackend.matrixTableType(userdata['username'], body['code'])


def blocking_blockmatrix_type(jbackend, userdata, body):
    return jbackend.blockMatrixType(userdata['username'], body['code'])


def blocking_get_reference(jbackend, userdata, body):   # pylint: disable=unused-argument
    return jbackend.referenceGenome(userdata['username'], body['name'])


async def handle_ws_response(request, userdata, cmd_str, f):
    log.info('connecting websocket')
    app = request.app
    jbackend = app['jbackend']

    await add_user(app, userdata)
    log.info('connecting websocket')
    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
    task = None
    try:
        await ws.prepare(request)
        app['sockets'].add(ws)
        log.info(f'websocket prepared: {ws}')
        body = await ws.receive_json()

        log.info(f"{cmd_str}: {body}")
        await ws.send_str(cmd_str)
        task = asyncio.ensure_future(send_ws_response(app['thread_pool'], ws, f, jbackend, userdata, body))
        r = await ws.receive()
        log.info('Received websocket message. Expected CLOSE, got {r}')
        assert r.type == aiohttp.WSMsgType.CLOSE
        return ws
    finally:
        if not ws.closed:
            await ws.close()
            log.info('Websocket was not closed. Closing.')
        if task is not None and not task.done():
            task.cancel()
            log.info('Task has been cancelled due to websocket closure.')
        log.info('websocket connection closed')
        app['sockets'].remove(ws)


@routes.get('/api/v1alpha/execute')
@rest_authenticated_users_only
async def execute(request, userdata):
    return await handle_ws_response(request, userdata, 'execute', blocking_execute)


@routes.get('/api/v1alpha/type/value')
@rest_authenticated_users_only
async def value_type(request, userdata):
    return await handle_ws_response(request, userdata, 'type/value', blocking_value_type)


@routes.get('/api/v1alpha/type/table')
@rest_authenticated_users_only
async def table_type(request, userdata):
    return await handle_ws_response(request, userdata, 'type/table', blocking_table_type)


@routes.get('/api/v1alpha/type/matrix')
@rest_authenticated_users_only
async def matrix_type(request, userdata):
    return await handle_ws_response(request, userdata, 'type/matrix', blocking_matrix_type)


@routes.get('/api/v1alpha/type/blockmatrix')
@rest_authenticated_users_only
async def blockmatrix_type(request, userdata):
    return await handle_ws_response(request, userdata, 'type/blockmatrix', blocking_blockmatrix_type)


@routes.get('/api/v1alpha/references/get')
@rest_authenticated_users_only
async def get_reference(request, userdata):  # pylint: disable=unused-argument
    return await handle_ws_response(request, userdata, 'references/get', blocking_get_reference)


@routes.get('/api/v1alpha/flags/get')
@rest_authenticated_developers_only
async def get_flags(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    jresp = await blocking_to_async(app['thread_pool'], app['jbackend'].flags)
    return java_to_web_response(jresp)


@routes.get('/api/v1alpha/flags/get/{flag}')
@rest_authenticated_developers_only
async def get_flag(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    f = request.match_info['flag']
    jresp = await blocking_to_async(app['thread_pool'], app['jbackend'].getFlag, f)
    return java_to_web_response(jresp)


@routes.get('/api/v1alpha/flags/set/{flag}')
@rest_authenticated_developers_only
async def set_flag(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    f = request.match_info['flag']
    v = request.query.get('value')
    if v is None:
        jresp = await blocking_to_async(app['thread_pool'], app['jbackend'].unsetFlag, f)
    else:
        jresp = await blocking_to_async(app['thread_pool'], app['jbackend'].setFlag, f, v)
    return java_to_web_response(jresp)


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
        jbackend, 'hail.log', False, False, 50, False, 3)
    app['jhc'] = jhc

    app['users'] = set()
    app['sockets'] = set()

    kube.config.load_incluster_config()
    k8s_client = kube.client.CoreV1Api()
    app['k8s_client'] = k8s_client


async def on_shutdown(app):
    for ws in app['sockets']:
        await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message='Server shutdown')


def run():
    app = web.Application()

    setup_aiohttp_session(app)

    app.add_routes(routes)

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    deploy_config = get_deploy_config()
    web.run_app(
        deploy_config.prefix_application(app, 'query'),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=get_in_cluster_server_ssl_context())
