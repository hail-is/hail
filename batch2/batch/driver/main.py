import asyncio
import concurrent
import logging

from aiohttp import web
import kubernetes as kube

import prometheus_client as pc
from prometheus_async.aio.web import server_stats

from hailtop.auth import async_get_userinfo
from hailtop.config import get_deploy_config

# import uvloop

from ..batch import Batch, Job
from ..log_store import LogStore
from ..batch_configuration import KUBERNETES_TIMEOUT_IN_SECONDS, REFRESH_INTERVAL_IN_SECONDS, \
    INSTANCE_ID, BATCH_NAMESPACE
from ..database import BatchDatabase

from .driver import Driver
from .k8s import K8s

# uvloop.install()

log = logging.getLogger('batch')

POD_EVICTIONS = pc.Counter('batch_pod_evictions', 'Count of batch pod evictions')

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
    batch = await Batch.from_db(request.app, batch_id, user)
    if not batch:
        raise web.HTTPNotFound()
    asyncio.ensure_future(batch._close_jobs())
    return web.Response()


@routes.patch('/api/v1alpha/batches/{user}/{batch_id}/cancel')
async def cancel_batch(request):
    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])
    batch = await Batch.from_db(request.app, batch_id, user)
    if not batch:
        raise web.HTTPNotFound()
    asyncio.ensure_future(batch._cancel_jobs())
    return web.Response()


@routes.delete('/api/v1alpha/batches/{user}/{batch_id}')
async def delete_batch(request):
    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])
    batch = await Batch.from_db(request.app, batch_id, user)
    if not batch:
        raise web.HTTPNotFound()
    # FIXME call from front end.  Can't yet, becuase then
    # Batch.from_db won't be able to find batch
    await batch.mark_deleted()
    asyncio.ensure_future(batch._cancel_jobs())
    return web.Response()


async def update_job_with_pod(app, job, pod_status):  # pylint: disable=R0911
    if pod_status:
        pod_name = pod_status["name"]
    else:
        pod_name = None

    log.info(f'update job {job.id if job else "None"} with pod {pod_name}')
    if job and job._state == 'Pending':
        if pod_status:
            log.error(f'pending job {job.id} has pod {pod_name}, ignoring')
        return

    if pod_status and (not job or job.is_complete()):
        await app['driver'].delete_pod(name=pod_name)
        return

    if job and job._cancelled and not job.always_run and job._state == 'Running':
        await job.set_state('Cancelled')
        await job._delete_pod()
        return

    if not pod_status:
        log.info(f'job {job.id} no pod found, rescheduling')
        await job.mark_unscheduled()
        return

    if pod_status:
        pod_state = pod_status["state"]
    else:
        pod_state = None

    if pod_state and pod_state in ('succeeded', 'error', 'failed'):
        log.info(f'job {job.id} mark complete')
        await job.mark_complete(pod_status)
        return


async def pod_changed(app, pod_status):
    job = await Job.from_pod(app, pod_status)
    await update_job_with_pod(app, job, pod_status)


async def refresh_pods(app):
    log.info(f'refreshing pods')

    pod_jobs = [Job.from_record(app, record)
                async for record in app['db'].jobs.get_records_where({'state': 'Running'})]

    pod_statuses = app['driver'].list_pods()
    log.info(f'batch had {len(pod_statuses)} pods')

    seen_pods = set()
    for pod_status in pod_statuses:
        pod_name = pod_status["name"]
        seen_pods.add(pod_name)
        asyncio.ensure_future(pod_changed(app, pod_status))

    if len(seen_pods) != len(pod_jobs):
        log.info('restarting running jobs with pods not seen in batch')

    async def restart_job(job):
        log.info(f'restarting job {job.id}')
        await update_job_with_pod(app, job, None)
    asyncio.gather(*[restart_job(job)
                     for job in pod_jobs
                     if job._pod_name not in seen_pods])


async def driver_event_loop(app):
    await asyncio.sleep(1)
    while True:
        try:
            pod_status = await app['driver'].complete_queue.get()
            await pod_changed(app, pod_status)
        except Exception:  # pylint: disable=broad-except
            log.exception(f'driver event loop failed due to exception')


async def polling_event_loop(app):
    await asyncio.sleep(1)
    while True:
        try:
            await refresh_pods(app)
        except Exception:  # pylint: disable=broad-except
            log.exception(f'polling event loop failed due to exception')
        await asyncio.sleep(60 * 10)


async def db_cleanup_event_loop(app):
    await asyncio.sleep(1)
    while True:
        try:
            for record in await app['db'].batch.get_finished_deleted_records():
                batch = Batch.from_record(app, record, deleted=True)
                await batch.delete()
        except Exception as exc:  # pylint: disable=W0703
            log.exception(f'Could not delete batches due to exception: {exc}')
        await asyncio.sleep(REFRESH_INTERVAL_IN_SECONDS)


@routes.post('/api/v1alpha/instances/activate')
async def activate_worker(request):
    return await asyncio.shield(request.app['driver'].activate_worker(request))


@routes.post('/api/v1alpha/instances/deactivate')
async def deactivate_worker(request):
    return await asyncio.shield(request.app['driver'].deactivate_worker(request))


@routes.post('/api/v1alpha/instances/pod_complete')
async def pod_complete(request):
    return await asyncio.shield(request.app['driver'].pod_complete(request))


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

    driver = Driver(db, k8s, bucket_name)
    app['driver'] = driver
    app['log_store'] = LogStore(pool, INSTANCE_ID, bucket_name)

    await driver.initialize()
    await refresh_pods(app)  # this is really slow for large N

    asyncio.ensure_future(driver.run())
    # we need a polling event loop in case a delete happens before a create job, but this is too slow
    # we also need a polling loop in case pod creation fails
    # asyncio.ensure_future(polling_event_loop(app))
    asyncio.ensure_future(driver_event_loop(app))
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
