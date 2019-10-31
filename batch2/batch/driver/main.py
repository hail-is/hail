import os
import asyncio
import concurrent
import logging
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
    INSTANCE_ID, BATCH_NAMESPACE
from ..google_compute import GServices

from .instance_pool import InstancePool
from .scheduler import Scheduler

# uvloop.install()

log = logging.getLogger('batch')

log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')
log.info(f'INSTANCE_ID = {INSTANCE_ID}')

routes = web.RouteTableDef()

deploy_config = get_deploy_config()


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


@routes.patch('/api/v1alpha/batches/{user}/{batch_id}/close')
async def close_batch(request):
    db = request.app['db']

    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])

    record = db.execute_and_fetchone(
        '''
SELECT state FROM batch WHERE user = %s AND batch_id = %s;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    request.app['scheduler_state_changed'].set()

    return web.Response()


@routes.patch('/api/v1alpha/batches/{user}/{batch_id}/cancel')
async def cancel_batch(request):
    db = request.app['db']

    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])

    record = db.execute_and_fetchone(
        '''
SELECT state FROM batch WHERE  user = %s AND batch_id = %s;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    request.app['cancel_state_changed'].set()
    request.app['scheduler_state_changed'].set()

    return web.Response()


@routes.delete('/api/v1alpha/batches/{user}/{batch_id}')
async def delete_batch(request):
    db = request.app['db']

    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])

    record = db.execute_and_fetchone(
        '''
SELECT state FROM batch WHERE user = %s AND batch_id = %s;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    request.app['cancel_state_changed'].set()
    request.app['scheduler_state_changed'].set()

    return web.Response()


async def db_cleanup_event_loop(db):
    while True:
        try:
            await db.just_execute('''
DELETE FROM batch
WHERE deleted AND (NOT closed OR n_jobs = n_completed)
''')
        except Exception:
            log.exception(f'in db cleanup loop')
        await asyncio.sleep(REFRESH_INTERVAL_IN_SECONDS)


async def activate_instance_1(request):
    app = request.app
    inst_pool = app['inst_pool']

    body = await request.json()
    inst_token = body['inst_token']
    ip_address = body['ip_address']

    instance = inst_pool.token_inst.get(inst_token)
    if not instance:
        log.warning(f'/activate_worker from unknown instance {inst_token}')
        raise web.HTTPNotFound()

    log.info(f'activating {instance}')
    await instance.activate(ip_address)
    return web.Response()


@routes.post('/api/v1alpha/instances/activate')
async def activate_instance(request):
    return await asyncio.shield(activate_instance_1(request))


async def deactivate_instance_1(request):
    inst_pool = request.app['inst_pool']

    body = await request.json()
    inst_token = body['inst_token']

    instance = inst_pool.token_inst.get(inst_token)
    if not instance:
        log.warning(f'/deactivate_worker from unknown instance {inst_token}')
        raise web.HTTPNotFound()

    log.info(f'deactivating {instance}')
    await instance.deactivate()
    return web.Response()


@routes.post('/api/v1alpha/instances/deactivate')
async def deactivate_instance(request):
    return await asyncio.shield(deactivate_instance_1(request))


async def job_complete_1(request):
    app = request.app
    inst_pool = app['inst_pool']

    body = await request.json()
    inst_token = body['inst_token']
    status = body['status']

    instance = inst_pool.token_inst.get(inst_token)
    if not instance:
        log.warning(f'job_complete from unknown instance {inst_token}')
        raise web.HTTPNotFound()

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

    await mark_job_complete(app, batch_id, job_id, new_state, status)

    return web.Response()


@routes.post('/api/v1alpha/instances/job_complete')
async def job_complete(request):
    return await asyncio.shield(job_complete_1(request))


async def on_startup(app):
    userinfo = await async_get_userinfo()
    log.info(f'running as {userinfo["username"]}')

    bucket_name = userinfo['bucket_name']
    log.info(f'bucket_name {bucket_name}')

    app['log_root'] = f'gs://{bucket_name}/batch2/logs/{INSTANCE_ID}'

    pool = concurrent.futures.ThreadPoolExecutor()
    app['blocking_pool'] = pool

    kube.config.load_incluster_config()
    k8s_client = kube.client.CoreV1Api()
    app['k8s_client'] = k8s_client

    db = Database()
    await db.async_init()
    app['db'] = db

    machine_name_prefix = f'batch2-worker-{BATCH_NAMESPACE}-'

    batch_gsa_key = os.environ.get('BATCH_GSA_KEY', '/batch-gsa-key/privateKeyData')
    credentials = google.oauth2.service_account.Credentials.from_service_account_file(batch_gsa_key)
    gservices = GServices(machine_name_prefix, credentials)
    app['gservices'] = gservices

    scheduler_state_changed = asyncio.Event()
    app['scheduler_state_changed'] = scheduler_state_changed

    cancel_state_changed = asyncio.Event()
    app['cancel_state_changed'] = cancel_state_changed

    inst_pool = InstancePool(app, machine_name_prefix)
    await inst_pool.async_init()
    app['inst_pool'] = inst_pool

    scheduler = Scheduler(app)
    await scheduler.async_init()
    app['scheduler'] = scheduler

    app['log_store'] = LogStore(app)

    asyncio.ensure_future(db_cleanup_event_loop(db))


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
