import asyncio
import base64
import json
import logging
import os
import signal
from collections import defaultdict

import aioredis
import kubernetes_asyncio.client
import kubernetes_asyncio.config
import uvloop
from aiohttp import web
from prometheus_async.aio.web import server_stats  # type: ignore

from gear import AuthClient, monitor_endpoints_middleware, setup_aiohttp_session
from gear.clients import get_cloud_async_fs_factory
from hailtop import httpx
from hailtop.aiotools import AsyncFS
from hailtop.config import get_deploy_config
from hailtop.hail_logging import AccessLogger
from hailtop.tls import internal_server_ssl_context
from hailtop.utils import dump_all_stacktraces, retry_transient_errors

uvloop.install()

DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
log = logging.getLogger('memory')
routes = web.RouteTableDef()

socket = '/redis/redis.sock'

ASYNC_FS_FACTORY = get_cloud_async_fs_factory()

auth = AuthClient()


@routes.get('/healthcheck')
async def healthcheck(request):  # pylint: disable=unused-argument
    return web.Response()


@routes.get('/api/v1alpha/objects')
@auth.rest_authenticated_users_only
async def get_object(request, userdata):
    filepath = request.query.get('q')
    userinfo = await get_or_add_user(request.app, userdata)
    username = userdata['username']
    maybe_file = await get_file_or_none(request.app, username, userinfo['fs'], filepath)
    if maybe_file is None:
        raise web.HTTPNotFound()
    return web.Response(body=maybe_file)


@routes.post('/api/v1alpha/objects')
@auth.rest_authenticated_users_only
async def write_object(request, userdata):
    filepath = request.query.get('q')
    userinfo = await get_or_add_user(request.app, userdata)
    username = userdata['username']
    data = await request.read()

    file_key = make_redis_key(username, filepath)

    async def persist_and_cache():
        try:
            await persist(userinfo['fs'], filepath, data)
            await cache_file(request.app['redis_pool'], file_key, data)
            return data
        finally:
            del request.app['files_in_progress'][file_key]

    fut = asyncio.ensure_future(persist_and_cache())
    request.app['files_in_progress'][file_key] = fut
    await fut
    return web.Response(status=200)


async def get_or_add_user(app, userdata):
    users = app['users']
    userlocks = app['userlocks']
    username = userdata['username']
    if username not in users:
        async with userlocks[username]:
            if username not in users:
                k8s_client = app['k8s_client']
                hail_credentials_secret = await retry_transient_errors(
                    k8s_client.read_namespaced_secret,
                    userdata['hail_credentials_secret_name'],
                    DEFAULT_NAMESPACE,
                    _request_timeout=5.0,
                )
                cloud_credentials_data = json.loads(base64.b64decode(hail_credentials_secret.data['key.json']).decode())
                users[username] = {'fs': ASYNC_FS_FACTORY.from_credentials_data(cloud_credentials_data)}
    return users[username]


def make_redis_key(username, filepath):
    return f'{ username }_{ filepath }'


async def get_file_or_none(app, username, fs: AsyncFS, filepath):
    file_key = make_redis_key(username, filepath)
    redis_pool: aioredis.ConnectionsPool = app['redis_pool']

    (body,) = await redis_pool.execute('HMGET', file_key, 'body')
    if body is not None:
        return body

    if file_key in app['files_in_progress']:
        return await app['files_in_progress'][file_key]

    async def load_and_cache():
        try:
            data = await load_file(fs, filepath)
            await cache_file(redis_pool, file_key, data)
            return data
        except FileNotFoundError:
            return None
        finally:
            del app['files_in_progress'][file_key]

    fut = asyncio.ensure_future(load_and_cache())
    app['files_in_progress'][file_key] = fut
    return await fut


async def load_file(fs: AsyncFS, filepath):
    data = await fs.read(filepath)
    return data


async def persist(fs: AsyncFS, filepath: str, data: bytes):
    await fs.write(filepath, data)


async def cache_file(redis: aioredis.ConnectionsPool, file_key: str, data: bytes):
    await redis.execute('HMSET', file_key, 'body', data)


async def on_startup(app):
    app['client_session'] = httpx.client_session()
    app['files_in_progress'] = {}
    app['users'] = {}
    app['userlocks'] = defaultdict(asyncio.Lock)
    kubernetes_asyncio.config.load_incluster_config()
    k8s_client = kubernetes_asyncio.client.CoreV1Api()
    app['k8s_client'] = k8s_client
    app['redis_pool'] = await aioredis.create_pool(socket)


async def on_cleanup(app):
    try:
        app['redis_pool'].close()
    finally:
        try:
            del app['k8s_client']
        finally:
            try:
                await app['client_session'].close()
            finally:
                try:
                    for items in app['users'].values():
                        try:
                            await items['fs'].close()
                        except:
                            pass
                finally:
                    await asyncio.gather(*(t for t in asyncio.all_tasks() if t is not asyncio.current_task()))


def run():
    app = web.Application(middlewares=[monitor_endpoints_middleware])

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
