import secrets
import logging
from functools import wraps
import concurrent
import asyncio
from aiohttp import web
import kubernetes_asyncio as kube
import google.oauth2.service_account
from prometheus_async.aio.web import server_stats
from gear import Database
from hailtop.auth import async_get_userinfo
from hailtop.config import get_deploy_config

# import uvloop

from ..batch import mark_job_complete
from ..log_store import LogStore
from ..batch_configuration import REFRESH_INTERVAL_IN_SECONDS, \
    BATCH_NAMESPACE
from ..google_compute import GServices

from .instance_pool import InstancePool
from .scheduler import Scheduler

# uvloop.install()

log = logging.getLogger('batch')

log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')

routes = web.RouteTableDef()

deploy_config = get_deploy_config()


def batch_only(fun):
    @wraps(fun)
    async def wrapped(request):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            raise web.HTTPUnauthorized()
        if not auth_header.startswith('Bearer '):
            raise web.HTTPUnauthorized()
        token = auth_header[7:]
        if not secrets.compare_digest(token, request.app['internal_token']):
            raise web.HTTPUnauthorized()

        return await fun(request)
    return wrapped


def instances_only(fun):
    @wraps(fun)
    async def wrapped(request):
        app = request.app

        instance_name = request.headers.get('X-Hail-Instance-Name')
        if not instance_name:
            raise web.HTTPUnauthorized()

        instance_pool = app['inst_pool']
        instance = instance_pool.name_instance.get(instance_name)
        if not instance:
            raise web.HTTPUnauthorized()
        if instance.state not in ('pending', 'active'):
            raise web.HTTPUnauthorized()

        auth_header = request.headers.get('Authorization')
        if not auth_header:
            raise web.HTTPUnauthorized()
        if not auth_header.startswith('Bearer '):
            raise web.HTTPUnauthorized()
        token = auth_header[7:]

        db = request.app['db']
        record = await db.execute_and_fetchone(
            'SELECT state FROM instances WHERE id = %s AND token = %s;',
            (instance_name, token))
        if not record:
            raise web.HTTPUnauthorized()

        return await fun(request, instance)
    return wrapped


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


@routes.patch('/api/v1alpha/batches/{user}/{batch_id}/close')
@batch_only
async def close_batch(request):
    db = request.app['db']

    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])

    record = db.execute_and_fetchone(
        '''
SELECT state FROM batches WHERE user = %s AND id = %s;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    request.app['scheduler_state_changed'].set()

    return web.Response()


@routes.patch('/api/v1alpha/batches/{user}/{batch_id}/cancel')
@batch_only
async def cancel_batch(request):
    db = request.app['db']

    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])

    record = db.execute_and_fetchone(
        '''
SELECT state FROM batches WHERE user = %s AND id = %s;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    request.app['cancel_state_changed'].set()
    request.app['scheduler_state_changed'].set()

    return web.Response()


@routes.delete('/api/v1alpha/batches/{user}/{batch_id}')
@batch_only
async def delete_batch(request):
    db = request.app['db']

    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])

    record = db.execute_and_fetchone(
        '''
SELECT state FROM batches WHERE user = %s AND id = %s;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    request.app['cancel_state_changed'].set()
    request.app['scheduler_state_changed'].set()

    return web.Response()


async def db_cleanup_event_loop(db, log_store):
    while True:
        try:
            async for record in db.execute_and_fetchall('''
SELECT id FROM batches
WHERE deleted AND (NOT closed OR n_jobs = n_completed);
'''):
                batch_id = record['id']
                await log_store.delete_batch_logs(batch_id)
                await db.just_execute('DELETE FROM batches WHERE id = %s;',
                                      (batch_id,))
        except Exception:
            log.exception(f'in db cleanup loop')
        await asyncio.sleep(REFRESH_INTERVAL_IN_SECONDS)


async def activate_instance_1(request, instance):
    body = await request.json()
    ip_address = body['ip_address']

    log.info(f'activating {instance}')
    await instance.activate(ip_address)
    await instance.mark_healthy()
    return web.Response()


@routes.post('/api/v1alpha/instances/activate')
@instances_only
async def activate_instance(request, instance):
    return await asyncio.shield(activate_instance_1(request, instance))


async def deactivate_instance_1(instance):
    log.info(f'deactivating {instance}')
    await instance.deactivate()
    await instance.mark_healthy()
    return web.Response()


@routes.post('/api/v1alpha/instances/deactivate')
@instances_only
async def deactivate_instance(request, instance):  # pylint: disable=unused-argument
    return await asyncio.shield(deactivate_instance_1(instance))


async def job_complete_1(request, instance):
    body = await request.json()
    status = body['status']

    batch_id = status['batch_id']
    job_id = status['job_id']

    status_state = status['state']
    if status_state == 'succeeded':
        new_state = 'Success'
    elif status_state == 'error':
        new_state = 'Error'
    else:
        assert status_state == 'failed', status_state
        new_state = 'Failed'

    await mark_job_complete(request.app, batch_id, job_id, new_state, status)

    await instance.mark_healthy()

    return web.Response()


@routes.post('/api/v1alpha/instances/job_complete')
@instances_only
async def job_complete(request, instance):
    return await asyncio.shield(job_complete_1(request, instance))


async def on_startup(app):
    userinfo = await async_get_userinfo()
    log.info(f'running as {userinfo["username"]}')

    bucket_name = userinfo['bucket_name']
    log.info(f'bucket_name {bucket_name}')

    pool = concurrent.futures.ThreadPoolExecutor()
    app['blocking_pool'] = pool

    kube.config.load_incluster_config()
    k8s_client = kube.client.CoreV1Api()
    app['k8s_client'] = k8s_client

    db = Database()
    await db.async_init()
    app['db'] = db

    row = await db.execute_and_fetchone(
        'SELECT token FROM tokens WHERE name = %s;',
        'instance_id')
    instance_id = row['token']
    log.info(f'instance_id {instance_id}')
    app['instance_id'] = instance_id

    row = await db.execute_and_fetchone(
        'SELECT token FROM tokens WHERE name = %s;',
        'internal')
    app['internal_token'] = row['token']

    machine_name_prefix = f'batch2-worker-{BATCH_NAMESPACE}-'
    credentials = google.oauth2.service_account.Credentials.from_service_account_file(
        '/batch-gsa-key/privateKeyData')
    gservices = GServices(machine_name_prefix, credentials)
    app['gservices'] = gservices

    scheduler_state_changed = asyncio.Event()
    app['scheduler_state_changed'] = scheduler_state_changed

    cancel_state_changed = asyncio.Event()
    app['cancel_state_changed'] = cancel_state_changed

    log_store = LogStore(bucket_name, instance_id, pool, credentials)
    app['log_store'] = log_store

    inst_pool = InstancePool(app, machine_name_prefix)
    app['inst_pool'] = inst_pool
    await inst_pool.async_init()

    scheduler = Scheduler(app)
    await scheduler.async_init()
    app['scheduler'] = scheduler

    asyncio.ensure_future(db_cleanup_event_loop(db, log_store))


async def on_cleanup(app):
    blocking_pool = app['blocking_pool']
    blocking_pool.shutdown()


def run():
    app = web.Application(client_max_size=None)

    app.add_routes(routes)
    app.router.add_get("/metrics", server_stats)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    web.run_app(deploy_config.prefix_application(app, 'batch2-driver'), host='0.0.0.0', port=5000)
