from typing import Dict
import traceback
import os
import base64
import concurrent
import logging
import uvloop
import asyncio
import signal
from aiohttp import web
import kubernetes_asyncio as kube
from prometheus_async.aio.web import server_stats  # type: ignore
from collections import defaultdict
from hailtop.utils import blocking_to_async, retry_transient_errors, dump_all_stacktraces
from hailtop.config import get_deploy_config
from hailtop.tls import internal_server_ssl_context
from hailtop.hail_logging import AccessLogger
from hailtop import version
from gear import (
    setup_aiohttp_session,
    rest_authenticated_users_only,
    rest_authenticated_developers_only,
    monitor_endpoint,
)

from .sockets import connect_to_java

uvloop.install()

DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
log = logging.getLogger(__name__)
routes = web.RouteTableDef()


async def add_user(app, userdata):
    username = userdata['username']
    users = app['users']
    if username in users:
        return

    k8s_client = app['k8s_client']
    gsa_key_secret = await retry_transient_errors(
        k8s_client.read_namespaced_secret, userdata['gsa_key_secret_name'], DEFAULT_NAMESPACE, _request_timeout=5.0
    )

    if username in users:
        return
    gsa_key = base64.b64decode(gsa_key_secret.data['key.json']).decode()
    with connect_to_java() as java:
        java.add_user(username, gsa_key)
    users.add(username)


@routes.get('/healthcheck')
async def healthcheck(request):  # pylint: disable=unused-argument
    return web.Response()


def blocking_execute(userdata, body):
    with connect_to_java() as java:
        log.info(f'executing {body["token"]}')
        return java.execute(
            userdata['username'],
            userdata['session_id'],
            body['billing_project'],
            body['bucket'],
            body['code'],
            body['token'],
        )


def blocking_load_references_from_dataset(userdata, body):
    with connect_to_java() as java:
        return java.load_references_from_dataset(
            userdata['username'], body['billing_project'], body['bucket'], body['path']
        )


def blocking_value_type(userdata, body):
    with connect_to_java() as java:
        return java.value_type(userdata['username'], body['code'])


def blocking_table_type(userdata, body):
    with connect_to_java() as java:
        return java.table_type(userdata['username'], body['code'])


def blocking_matrix_type(userdata, body):
    with connect_to_java() as java:
        return java.matrix_table_type(userdata['username'], body['code'])


def blocking_blockmatrix_type(userdata, body):
    with connect_to_java() as java:
        return java.block_matrix_type(userdata['username'], body['code'])


def blocking_get_reference(userdata, body):  # pylint: disable=unused-argument
    with connect_to_java() as java:
        return java.reference_genome(userdata['username'], body['name'])


async def handle_ws_response(request, userdata, endpoint, f):
    app = request.app
    user_queries: Dict[str, asyncio.Future] = request.app['queries'][userdata['username']]

    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
    await ws.prepare(request)
    body = await ws.receive_json()

    query = user_queries.get(body['token'])
    if query is None:
        await add_user(app, userdata)
        query = asyncio.ensure_future(retry_transient_errors(blocking_to_async, app['thread_pool'], f, userdata, body))
        user_queries[body['token']] = query

    try:
        receive = asyncio.ensure_future(
            ws.receive_str()
        )  # receive automatically ping-pongs which keeps the socket alive
        await asyncio.wait([receive, query], return_when=asyncio.FIRST_COMPLETED)
        if receive.done():
            # we expect no messages from the client
            response = receive.result()
            raise AssertionError(f'{endpoint}: client broke the protocol by sending: {response}')
        if not query.done():
            return
        if query.exception() is not None:
            exc = query.exception()
            exc_str = traceback.format_exception(type(exc), exc, exc.__traceback__)
            await ws.send_json({'status': 500, 'value': exc_str})
        else:
            await ws.send_json({'status': 200, 'value': query.result()})
        assert (await receive) == 'bye'
        del user_queries[body['token']]
    finally:
        receive.cancel()
        query.cancel()
        await ws.close()
    return ws


@routes.get('/api/v1alpha/execute')
@monitor_endpoint
@rest_authenticated_users_only
async def execute(request, userdata):
    return await handle_ws_response(request, userdata, 'execute', blocking_execute)


@routes.get('/api/v1alpha/load_references_from_dataset')
@monitor_endpoint
@rest_authenticated_users_only
async def load_references_from_dataset(request, userdata):
    return await handle_ws_response(
        request, userdata, 'load_references_from_dataset', blocking_load_references_from_dataset
    )


@routes.get('/api/v1alpha/type/value')
@monitor_endpoint
@rest_authenticated_users_only
async def value_type(request, userdata):
    return await handle_ws_response(request, userdata, 'type/value', blocking_value_type)


@routes.get('/api/v1alpha/type/table')
@monitor_endpoint
@rest_authenticated_users_only
async def table_type(request, userdata):
    return await handle_ws_response(request, userdata, 'type/table', blocking_table_type)


@routes.get('/api/v1alpha/type/matrix')
@monitor_endpoint
@rest_authenticated_users_only
async def matrix_type(request, userdata):
    return await handle_ws_response(request, userdata, 'type/matrix', blocking_matrix_type)


@routes.get('/api/v1alpha/type/blockmatrix')
@monitor_endpoint
@rest_authenticated_users_only
async def blockmatrix_type(request, userdata):
    return await handle_ws_response(request, userdata, 'type/blockmatrix', blocking_blockmatrix_type)


@routes.get('/api/v1alpha/references/get')
@monitor_endpoint
@rest_authenticated_users_only
async def get_reference(request, userdata):  # pylint: disable=unused-argument
    return await handle_ws_response(request, userdata, 'references/get', blocking_get_reference)


@routes.get('/api/v1alpha/flags/get')
@monitor_endpoint
@rest_authenticated_developers_only
async def get_flags(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    with connect_to_java() as java:
        jresp = await blocking_to_async(app['thread_pool'], java.flags)
    return web.json_response(jresp)


@routes.get('/api/v1alpha/flags/get/{flag}')
@monitor_endpoint
@rest_authenticated_developers_only
async def get_flag(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    f = request.match_info['flag']
    with connect_to_java() as java:
        jresp = await blocking_to_async(app['thread_pool'], java.get_flag, f)
    return web.json_response(jresp)


@routes.get('/api/v1alpha/flags/set/{flag}')
@monitor_endpoint
@rest_authenticated_developers_only
async def set_flag(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    f = request.match_info['flag']
    v = request.query.get('value')
    with connect_to_java() as java:
        if v is None:
            jresp = await blocking_to_async(app['thread_pool'], java.unset_flag, f)
        else:
            jresp = await blocking_to_async(app['thread_pool'], java.set_flag, f, v)
    return web.json_response(jresp)


@routes.get('/api/v1alpha/version')
async def rest_get_version(request):  # pylint: disable=W0613
    return web.Response(text=version())


async def on_startup(app):
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    app['thread_pool'] = thread_pool
    app['user_keys'] = dict()
    app['users'] = set()
    app['queries'] = defaultdict(dict)

    kube.config.load_incluster_config()
    k8s_client = kube.client.CoreV1Api()
    app['k8s_client'] = k8s_client


async def on_cleanup(app):
    if 'k8s_client' in app:
        del app['k8s_client']
    await asyncio.gather(*(t for t in asyncio.all_tasks() if t is not asyncio.current_task()))


async def on_shutdown(_):
    # Filter the asyncio.current_task(), because if we await
    # the current task we'll end up in a deadlock
    remaining_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    log.info(f"On shutdown request received, with {len(remaining_tasks)} remaining tasks")
    await asyncio.wait(remaining_tasks)
    log.info("All tasks on shutdown have completed")


def run():
    app = web.Application()

    setup_aiohttp_session(app)

    app.add_routes(routes)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/metrics", server_stats)

    asyncio.get_event_loop().add_signal_handler(signal.SIGUSR1, dump_all_stacktraces)

    deploy_config = get_deploy_config()
    web.run_app(
        deploy_config.prefix_application(app, 'query'),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=internal_server_ssl_context(),
    )
