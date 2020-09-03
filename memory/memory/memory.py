import aioredis
import asyncio
import base64
import concurrent
import json
import logging
import os
import uvloop
from aiohttp import web
import kubernetes_asyncio as kube

from hailtop.config import get_deploy_config
from hailtop.google_storage import GCS
from hailtop.hail_logging import AccessLogger
from hailtop.tls import get_in_cluster_server_ssl_context
from hailtop.utils import AsyncWorkerPool, retry_transient_errors
from gear import setup_aiohttp_session, rest_authenticated_users_only

uvloop.install()

BATCH_PODS_NAMESPACE = os.environ['HAIL_BATCH_PODS_NAMESPACE']
log = logging.getLogger('batch')
routes = web.RouteTableDef()

socket = '/redis/redis.sock'

@routes.get('/healthcheck')
async def healthcheck(request):  # pylint: disable=unused-argument
    return web.Response()

@routes.get('/api/v1alpha/objects/')
@rest_authenticated_users_only
async def get_object(request, userdata):
    filename = request.query.get('q')
    userinfo = await get_or_add_user(request.app, userdata)
    result = await get_file_or_none(request.app, userdata['username'], userinfo, filename)
    if result is None:
        raise web.HTTPNotFound()
    etag, body = result
    return web.Response(headers={'ETag': etag}, body=body)

async def get_or_add_user(app, userdata):
    users = app['users']
    username = userdata['username']
    if username not in users:
        k8s_client = app['k8s_client']
        gsa_key_secret = await retry_transient_errors(
            k8s_client.read_namespaced_secret,
            userdata['gsa_key_secret_name'],
            BATCH_PODS_NAMESPACE,
            _request_timeout=5.0)
        gsa_key = base64.b64decode(gsa_key_secret.data['key.json']).decode()
        users[username] = {'fs': GCS(blocking_pool=app['thread_pool'], key=json.loads(gsa_key))}
    return users[username]

def make_redis_key(username, filepath):
    return f'{ username }_{ filepath }'

async def get_file_or_none(app, username, userinfo, filepath):
    filekey = make_redis_key(username, filepath)
    try:
        etag = fs.get_etag(filepath)
        if etag is None:
            await app['redis_pool'].execute('DEL', filekey)
            return None
    except Exception as e:
        log.info("memory: Couldn't retrieve file {filepath} for user {username} with error {e}")
        return None

    cached_etag = await app['redis_pool'].execute('HGET', filekey, 'etag')
    if cached_etag is not None and cached_etag == etag:
        result = await app['redis_pool'].execute('HGET', filekey, 'body')
        return cached_etag, result

    if filekey not in app['files_in_progress']:
        try:
            app['worker_pool'].call_nowait(load_file, app['redis_pool'], app['files_in_progress'], filekey, userinfo['fs'], filepath)
        except asyncio.QueueFull:
            pass
    return None

async def load_file(redis, files, file_key, fs, filepath):
    try:
        files.add(file_key)
        data = await fs.read_binary_gs_file(filepath)
        etag = await fs.get_etag(filepath)
        await redis.execute('HMSET', file_key, 'etag', etag, 'body', data)
    finally:
        files.remove(file_key)

async def on_startup(app):
    app['thread_pool'] = concurrent.futures.ThreadPoolExecutor()
    app['worker_pool'] = AsyncWorkerPool(parallelism=4, queue_size=10)
    app['files_in_progress'] = set()
    app['users'] = {}
    kube.config.load_incluster_config()
    k8s_client = kube.client.CoreV1Api()
    app['k8s_client'] = k8s_client
    app['redis_pool'] = await aioredis.create_pool(socket)

def run():
    app = web.Application()

    setup_aiohttp_session(app)
    app.add_routes(routes)
    app.on_startup.append(on_startup)

    deploy_config = get_deploy_config()
    web.run_app(
        deploy_config.prefix_application(app, 'memory'),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=get_in_cluster_server_ssl_context())
