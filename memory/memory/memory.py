import aioredis
import asyncio
import base64
import concurrent
import json
import logging
import os
import uvloop
import signal
from aiohttp import web
import kubernetes_asyncio as kube
from prometheus_async.aio.web import server_stats  # type: ignore
from typing import Set

from hailtop.config import get_deploy_config
from hailtop.google_storage import GCS
from hailtop.hail_logging import AccessLogger
from hailtop.tls import internal_server_ssl_context
from hailtop.utils import AsyncWorkerPool, retry_transient_errors, dump_all_stacktraces
from gear import setup_aiohttp_session, rest_authenticated_users_only, monitor_endpoint

uvloop.install()

DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
log = logging.getLogger('memory')
routes = web.RouteTableDef()

socket = '/redis/redis.sock'


@routes.get('/healthcheck')
async def healthcheck(request):  # pylint: disable=unused-argument
    return web.Response()


@routes.get('/api/v1alpha/objects')
@monitor_endpoint
@rest_authenticated_users_only
async def get_object(request, userdata):
    filepath = request.query.get('q')
    userinfo = await get_or_add_user(request.app, userdata)
    username = userdata['username']
    log.info(f'memory: request for object {filepath} from user {username}')
    maybe_file = await get_file_or_none(request.app, username, userinfo['fs'], filepath)
    if maybe_file is None:
        raise web.HTTPNotFound()
    return web.Response(body=maybe_file)


@routes.post('/api/v1alpha/objects')
@monitor_endpoint
@rest_authenticated_users_only
async def write_object(request, userdata):
    filepath = request.query.get('q')
    userinfo = await get_or_add_user(request.app, userdata)
    username = userdata['username']
    data = await request.read()
    log.info(f'memory: post for object {filepath} from user {username}')

    file_key = make_redis_key(username, filepath)
    files = request.app['files_in_progress']
    files.add(file_key)

    await persist_in_gcs(userinfo['fs'], files, file_key, filepath, data)
    await cache_file(request.app['redis_pool'], files, file_key, filepath, data)
    return web.Response(status=200)


async def get_or_add_user(app, userdata):
    users = app['users']
    username = userdata['username']
    if username not in users:
        k8s_client = app['k8s_client']
        gsa_key_secret = await retry_transient_errors(
            k8s_client.read_namespaced_secret, userdata['gsa_key_secret_name'], DEFAULT_NAMESPACE, _request_timeout=5.0
        )
        gsa_key = base64.b64decode(gsa_key_secret.data['key.json']).decode()
        users[username] = {'fs': GCS(blocking_pool=app['thread_pool'], key=json.loads(gsa_key))}
    return users[username]


def make_redis_key(username, filepath):
    return f'{ username }_{ filepath }'


async def get_file_or_none(app, username, fs, filepath):
    file_key = make_redis_key(username, filepath)
    redis_pool: aioredis.ConnectionsPool = app['redis_pool']

    (body,) = await redis_pool.execute('HMGET', file_key, 'body')
    if body is not None:
        log.info(f"memory: Retrieved file {filepath} for user {username}")
        return body

    log.info(f"memory: Couldn't retrieve file {filepath} for user {username}: current version not in cache")
    if file_key not in app['files_in_progress']:
        try:
            log.info(f"memory: Loading {filepath} to cache for user {username}")
            app['files_in_progress'].add(file_key)
            app['worker_pool'].call_nowait(load_file, redis_pool, app['files_in_progress'], file_key, fs, filepath)
        except asyncio.QueueFull:
            pass
    return None


async def load_file(redis, files, file_key, fs, filepath):
    try:
        log.info(f"memory: {file_key}: reading.")
        data = await fs.read_binary_gs_file(filepath)
        log.info(f"memory: {file_key}: read {filepath}")
    except Exception as e:
        files.remove(file_key)
        raise e

    await cache_file(redis, files, file_key, filepath, data)


async def persist_in_gcs(fs: GCS, files: Set[str], file_key: str, filepath: str, data: str):
    try:
        log.info(f"memory: {file_key}: persisting.")
        await fs.write_gs_file_from_string(filepath, data)
        log.info(f"memory: {file_key}: persisted {filepath}")
    except Exception as e:
        files.remove(file_key)
        raise e


async def cache_file(redis: aioredis.ConnectionsPool, files: Set[str], file_key: str, filepath: str, data: str):
    try:
        await redis.execute('HMSET', file_key, 'body', data)
        log.info(f"memory: {file_key}: stored {filepath}")
    finally:
        files.remove(file_key)


async def on_startup(app):
    app['thread_pool'] = concurrent.futures.ThreadPoolExecutor()
    app['worker_pool'] = AsyncWorkerPool(parallelism=100, queue_size=10)
    app['files_in_progress'] = set()
    app['users'] = {}
    kube.config.load_incluster_config()
    k8s_client = kube.client.CoreV1Api()
    app['k8s_client'] = k8s_client
    app['redis_pool']: aioredis.ConnectionsPool = await aioredis.create_pool(socket)


async def on_cleanup(app):
    try:
        app['thread_pool'].shutdown()
    finally:
        try:
            app['worker_pool'].shutdown()
        finally:
            try:
                app['redis_pool'].close()
            finally:
                del app['k8s_client']
                await asyncio.gather(*(t for t in asyncio.all_tasks() if t is not asyncio.current_task()))


def run():
    app = web.Application()

    setup_aiohttp_session(app)
    app.add_routes(routes)
    app.router.add_get("/metrics", server_stats)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    asyncio.get_event_loop().add_signal_handler(signal.SIGUSR1, dump_all_stacktraces)

    deploy_config = get_deploy_config()
    web.run_app(
        deploy_config.prefix_application(app, 'memory'),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=internal_server_ssl_context(),
    )
