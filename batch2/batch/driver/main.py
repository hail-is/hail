import os
import asyncio
import concurrent
import logging

from aiohttp import web
import kubernetes as kube
import google.oauth2.service_account

from prometheus_async.aio.web import server_stats

from hailtop.auth import async_get_userinfo
from hailtop.config import get_deploy_config

# import uvloop

from ..batch import Batch, Job
from ..log_store import LogStore
from ..batch_configuration import KUBERNETES_TIMEOUT_IN_SECONDS, REFRESH_INTERVAL_IN_SECONDS, \
    INSTANCE_ID, BATCH_NAMESPACE
from ..database import BatchDatabase
from ..google_compute import GServices

from .instance_pool import InstancePool
from .scheduler import Scheduler
from .k8s import K8s

# uvloop.install()

log = logging.getLogger('batch')

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')
log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')
log.info(f'INSTANCE_ID = {INSTANCE_ID}')

routes = web.RouteTableDef()

deploy_config = get_deploy_config()


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


@routes.patch('/api/v1alpha/batches/{user}/{batch_id}/close')
async def close_batch(request):
    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])
    batch = await Batch.from_db(request.app['db'], batch_id, user)
    if not batch:
        raise web.HTTPNotFound()
    asyncio.ensure_future(batch._close_jobs())
    return web.Response()


@routes.patch('/api/v1alpha/batches/{user}/{batch_id}/cancel')
async def cancel_batch(request):
    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])
    batch = await Batch.from_db(request.app['db'], batch_id, user)
    if not batch:
        raise web.HTTPNotFound()
    asyncio.ensure_future(batch._cancel_jobs())
    return web.Response()


@routes.delete('/api/v1alpha/batches/{user}/{batch_id}')
async def delete_batch(request):
    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])
    batch = await Batch.from_db(request.app['db'], batch_id, user)
    if not batch:
        raise web.HTTPNotFound()
    # FIXME call from front end.  Can't yet, becuase then
    # Batch.from_db won't be able to find batch
    await batch.mark_deleted()
    asyncio.ensure_future(batch._cancel_jobs())
    return web.Response()


async def db_cleanup_event_loop(app):
    await asyncio.sleep(1)
    while True:
        try:
            # FIXME
            for record in await app['db'].batch.get_finished_deleted_records():
                batch = Batch.from_record(app['db'], record, deleted=True)
                await app['db'].batch.delete_record(batch.id)
        except Exception as exc:  # pylint: disable=W0703
            log.exception(f'Could not delete batches due to exception: {exc}')
        await asyncio.sleep(REFRESH_INTERVAL_IN_SECONDS)


async def activate_instance_1(request):
    inst_pool = request.app['inst_pool']

    body = await request.json()
    inst_token = body['inst_token']
    ip_address = body['ip_address']

    instance = inst_pool.token_inst.get(inst_token)
    if not instance:
        log.warning(f'/activate_worker from unknown instance {inst_token}')
        raise web.HTTPNotFound()

    log.info(f'activating {instance}')
    inst_pool.activate_instance(instance, ip_address)
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
    inst_pool.deactivate_instance(instance)
    return web.Response()


@routes.post('/api/v1alpha/instances/deactivate')
async def deactivate_instance(request):
    return await asyncio.shield(deactivate_instance_1(request))


async def job_complete_1(request):
    inst_pool = request.app['inst_pool']

    body = await request.json()
    inst_token = body['inst_token']
    status = body['status']

    instance = inst_pool.token_inst.get(inst_token)
    if not instance:
        log.warning(f'job_complete from unknown instance {inst_token}')
        raise web.HTTPNotFound()

    batch_id = status['batch_id']
    job_id = status['job_id']
    user = status['user']

    job = Job.from_db(request.app['db'], batch_id, job_id, user)
    if job is None:
        log.warning(f'job_complete from unknown job ({batch_id}, {job_id}), {instance}')
        return web.HTTPNotFound()
    log.info(f'job_complete from {job}, instance {inst_token}')
    await job.mark_complete(status)

    return web.Response()


@routes.post('/api/v1alpha/instances/job_complete')
async def job_complete(request):
    return await asyncio.shield(job_complete_1(request))


async def on_startup(app):
    pool = concurrent.futures.ThreadPoolExecutor()
    app['blocking_pool'] = pool

    kube.config.load_incluster_config()
    k8s_client = kube.client.CoreV1Api()
    app['k8s_client'] = k8s_client

    k8s = K8s(pool, KUBERNETES_TIMEOUT_IN_SECONDS, BATCH_NAMESPACE, k8s_client)

    userinfo = await async_get_userinfo()
    log.info(f'running as {userinfo["username"]}')

    bucket_name = userinfo['bucket_name']
    log.info(f'bucket_name {bucket_name}')

    db = await BatchDatabase('/sql-config/sql-config.json')
    app['db'] = db

    machine_name_prefix = f'batch2-worker-{BATCH_NAMESPACE}-'

    batch_gsa_key = os.environ.get('BATCH_GSA_KEY', '/batch-gsa-key/privateKeyData')
    credentials = google.oauth2.service_account.Credentials.from_service_account_file(batch_gsa_key)
    gservices = GServices(machine_name_prefix, credentials)
    app['gservices'] = gservices

    inst_pool = InstancePool(db, gservices, k8s, bucket_name, machine_name_prefix)
    await inst_pool.async_init()
    app['inst_pool'] = inst_pool

    scheduler = Scheduler(db, inst_pool)
    await scheduler.async_init()
    app['scheduler'] = scheduler

    app['log_store'] = LogStore(pool, INSTANCE_ID, bucket_name)

    asyncio.ensure_future(db_cleanup_event_loop(app))


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
