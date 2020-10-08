import glob
import subprocess
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


def blocking_execute(jbackend, username, session_id, billing_project, bucket, code):
    return jbackend.execute(username, session_id, billing_project, bucket, code)


@routes.post('/execute')
@rest_authenticated_users_only
async def execute(request, userdata):
    app = request.app
    thread_pool = app['thread_pool']
    jbackend = app['jbackend']
    body = await request.json()
    billing_project = body['billing_project']
    bucket = body['bucket']
    code = body['code']
    log.info(f'execute: {code}')
    await add_user(app, userdata)
    jresp = await blocking_to_async(thread_pool, blocking_execute, jbackend, userdata['username'], userdata['session_id'], billing_project, bucket, code)
    return java_to_web_response(jresp)


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
    jresp = await blocking_to_async(thread_pool, blocking_value_type, jbackend, userdata['username'], code)
    return java_to_web_response(jresp)


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
    jresp = await blocking_to_async(thread_pool, blocking_table_type, jbackend, userdata['username'], code)
    return java_to_web_response(jresp)


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
    jresp = await blocking_to_async(thread_pool, blocking_matrix_type, jbackend, userdata['username'], code)
    return java_to_web_response(jresp)


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
    jresp = await blocking_to_async(thread_pool, blocking_blockmatrix_type, jbackend, userdata['username'], code)
    return java_to_web_response(jresp)


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

    spark_home = os.environ.get('SPARK_HOME')
    if spark_home is None:
        find_spark_home = subprocess.run('find_spark_home.py', capture_output=True)
        if find_spark_home.returncode != 0:
            raise ValueError(f'''SPARK_HOME is not set and find_spark_home.py returned non-zero exit code:
STDOUT:
{find_spark_home.stdout}
STDERR:
{find_spark_home.stderr}''')
        spark_home = find_spark_home.stdout.decode().strip()
    py4j_jars = glob.glob(spark_home + '/jars/py4j*.jar')
    if len(py4j_jars) != 1:
        raise ValueError(f'Could not find a unique py4j jar. Found: {py4j_jars}')
    py4j_jar = py4j_jars[0]
    port = launch_gateway(
        jarpath=py4j_jar,
        classpath='{spark_home}/jars/*:/hail.jar',
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
        access_log_class=AccessLogger,
        ssl_context=get_in_cluster_server_ssl_context())
