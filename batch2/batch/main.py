import asyncio
import concurrent
import logging
import os
import traceback
import json
import time

import aiohttp_jinja2
from aiohttp import web
import cerberus
import kubernetes as kube
import prometheus_client as pc
from prometheus_async.aio import time as prom_async_time
from prometheus_async.aio.web import server_stats

from hailtop.utils import blocking_to_async
from hailtop.auth import async_get_userinfo
from hailtop.config import get_deploy_config
from gear import setup_aiohttp_session, \
    rest_authenticated_users_only, web_authenticated_users_only, \
    check_csrf_token
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, render_template

# import uvloop

from .batch import Batch, Job
from .log_store import LogStore
from .database import JobsBuilder
from .datetime_json import JSON_ENCODER
from .globals import get_db
from .batch_configuration import KUBERNETES_TIMEOUT_IN_SECONDS, REFRESH_INTERVAL_IN_SECONDS, \
    POD_VOLUME_SIZE, INSTANCE_ID, BATCH_IMAGE, BATCH_NAMESPACE
from .driver import Driver
from .k8s import K8s
from .utils import abort, jsonify

from . import schemas

# uvloop.install()

log = logging.getLogger('batch')

REQUEST_TIME = pc.Summary('batch_request_latency_seconds', 'Batch request latency in seconds', ['endpoint', 'verb'])
REQUEST_TIME_GET_JOB = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id', verb="GET")
REQUEST_TIME_GET_JOB_LOG = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id/log', verb="GET")
REQUEST_TIME_GET_POD_STATUS = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id/pod_status', verb="GET")
REQUEST_TIME_GET_BATCHES = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches', verb="GET")
REQUEST_TIME_POST_CREATE_JOBS = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/create', verb="POST")
REQUEST_TIME_POST_CREATE_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/create', verb='POST')
REQUEST_TIME_POST_GET_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id', verb='GET')
REQUEST_TIME_PATCH_CANCEL_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/cancel', verb="PATCH")
REQUEST_TIME_PATCH_CLOSE_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/close', verb="PATCH")
REQUEST_TIME_DELETE_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id', verb="DELETE")
REQUEST_TIME_GET_BATCH_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id', verb='GET')
REQUEST_TIME_POST_CANCEL_BATCH_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id/cancel', verb='POST')
REQUEST_TIME_GET_BATCHES_UI = REQUEST_TIME.labels(endpoint='/batches', verb='GET')
REQUEST_TIME_GET_LOGS_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id/jobs/job_id/log', verb="GET")
REQUEST_TIME_GET_POD_STATUS_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id/jobs/job_id/pod_status', verb="GET")

POD_EVICTIONS = pc.Counter('batch_pod_evictions', 'Count of batch pod evictions')
READ_POD_LOG_FAILURES = pc.Counter('batch_read_pod_log_failures', 'Count of batch read_pod_log failures')

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')
log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')
log.info(f'POD_VOLUME_SIZE {POD_VOLUME_SIZE}')
log.info(f'INSTANCE_ID = {INSTANCE_ID}')
log.info(f'BATCH_IMAGE = {BATCH_IMAGE}')

routes = web.RouteTableDef()

deploy_config = get_deploy_config()


def create_job(app, jobs_builder, batch_id, userdata, parameters):  # pylint: disable=R0912
    pod_spec = app['k8s_client'].api_client._ApiClient__deserialize(
        parameters['spec'], kube.client.V1PodSpec)

    job_id = parameters.get('job_id')
    parent_ids = parameters.get('parent_ids', [])
    input_files = parameters.get('input_files')
    output_files = parameters.get('output_files')
    pvc_size = parameters.get('pvc_size')
    if pvc_size is None and (input_files or output_files):
        pvc_size = POD_VOLUME_SIZE
    always_run = parameters.get('always_run', False)

    if len(pod_spec.containers) != 1:
        abort(400, f'only one container allowed in pod_spec {pod_spec}')

    if pod_spec.containers[0].name != 'main':
        abort(400, f'container name must be "main" was {pod_spec.containers[0].name}')

    if not pod_spec.containers[0].resources:
        pod_spec.containers[0].resources = kube.client.V1ResourceRequirements()
    if not pod_spec.containers[0].resources.requests:
        pod_spec.containers[0].resources.requests = {}
    if 'cpu' not in pod_spec.containers[0].resources.requests:
        pod_spec.containers[0].resources.requests['cpu'] = '100m'
    if 'memory' not in pod_spec.containers[0].resources.requests:
        pod_spec.containers[0].resources.requests['memory'] = '500M'

    state = 'Running' if len(parent_ids) == 0 else 'Pending'

    job = Job.create_job(
        app,
        jobs_builder,
        batch_id=batch_id,
        job_id=job_id,
        pod_spec=pod_spec,
        attributes=parameters.get('attributes'),
        callback=parameters.get('callback'),
        parent_ids=parent_ids,
        input_files=input_files,
        output_files=output_files,
        userdata=userdata,
        always_run=always_run,
        pvc_size=pvc_size,
        state=state)

    return job


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}')
@prom_async_time(REQUEST_TIME_GET_JOB)
@rest_authenticated_users_only
async def get_job(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']

    job = await Job.from_db(request.app, batch_id, job_id, user)
    if not job:
        abort(404)
    return jsonify(job.to_dict())


async def _get_job_log(app, batch_id, job_id, user):
    job = await Job.from_db(app, batch_id, job_id, user)
    if not job:
        abort(404)

    job_log = await job._read_logs()
    if job_log:
        return job_log
    abort(404)


async def _get_pod_status(app, batch_id, job_id, user):
    job = await Job.from_db(app, batch_id, job_id, user)
    if not job:
        abort(404)

    pod_statuses = await job._read_pod_status()
    if pod_statuses:
        return JSON_ENCODER.encode(pod_statuses)
    abort(404)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
@prom_async_time(REQUEST_TIME_GET_JOB_LOG)
@rest_authenticated_users_only
async def get_job_log(request, userdata):  # pylint: disable=R1710
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    job_log = await _get_job_log(request.app, batch_id, job_id, user)
    return jsonify(job_log)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/pod_status')
@prom_async_time(REQUEST_TIME_GET_POD_STATUS)
@rest_authenticated_users_only
async def get_pod_status(request, userdata):  # pylint: disable=R1710
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    pod_spec = await _get_pod_status(request.app, batch_id, job_id, user)
    return jsonify(pod_spec)


async def _get_batches_list(app, params, user):
    complete = params.get('complete')
    if complete:
        complete = complete == '1'
    success = params.get('success')
    if success:
        success = success == '1'
    attributes = {}
    for k, v in params.items():
        if k in ('complete', 'success'):  # params does not support deletion
            continue
        if not k.startswith('a:'):
            abort(400, f'unknown query parameter {k}')
        attributes[k[2:]] = v

    records = await app['db'].batch.find_records(user=user,
                                                 complete=complete,
                                                 success=success,
                                                 deleted=False,
                                                 attributes=attributes)

    return [await Batch.from_record(app, batch).to_dict(include_jobs=False)
            for batch in records]


@routes.get('/api/v1alpha/batches')
@prom_async_time(REQUEST_TIME_GET_BATCHES)
@rest_authenticated_users_only
async def get_batches_list(request, userdata):
    params = request.query
    user = userdata['username']
    return jsonify(await _get_batches_list(request.app, params, user))


@routes.post('/api/v1alpha/batches/{batch_id}/jobs/create')
@prom_async_time(REQUEST_TIME_POST_CREATE_JOBS)
@rest_authenticated_users_only
async def create_jobs(request, userdata):
    start = time.time()
    app = request.app
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']

    start1 = time.time()
    batch = await Batch.from_db(app, batch_id, user)
    log.info(f'took {round(time.time() - start1, 3)} seconds to get batch from db')

    if not batch:
        abort(404)
    if batch.closed:
        abort(400, f'batch {batch_id} is already closed')

    start2 = time.time()
    jobs_parameters = await request.json()
    log.info(f'took {round(time.time() - start2, 3)} seconds to get data from server')

    start3 = time.time()
    validator = cerberus.Validator(schemas.job_array_schema)
    if not await blocking_to_async(app['blocking_pool'], validator.validate, jobs_parameters):
        abort(400, 'invalid request: {}'.format(validator.errors))
    log.info(f"took {round(time.time() - start3, 3)} seconds to validate spec")

    start4 = time.time()
    jobs_builder = JobsBuilder(app['db'])
    try:
        for job_params in jobs_parameters['jobs']:
            create_job(app, jobs_builder, batch.id, userdata, job_params)

        success = await jobs_builder.commit()
        if not success:
            abort(400, f'insertion of jobs in db failed')
    finally:
        await jobs_builder.close()

    log.info(f'took {round(time.time() - start4, 3)} seconds to commit jobs to db')

    log.info(f'took {round(time.time() - start, 3)} seconds to create jobs from start to finish')
    return jsonify({})


@routes.post('/api/v1alpha/batches/create')
@prom_async_time(REQUEST_TIME_POST_CREATE_BATCH)
@rest_authenticated_users_only
async def create_batch(request, userdata):
    start = time.time()
    parameters = await request.json()

    validator = cerberus.Validator(schemas.batch_schema)
    if not validator.validate(parameters):
        abort(400, 'invalid request: {}'.format(validator.errors))

    batch = await Batch.create_batch(
        request.app,
        attributes=parameters.get('attributes'),
        callback=parameters.get('callback'),
        userdata=userdata,
        n_jobs=parameters['n_jobs'])
    if batch is None:
        abort(400, f'creation of batch in db failed')

    log.info(f'took {round(time.time() - start, 3)} seconds to initialize batch {batch.id} in db')
    return jsonify(await batch.to_dict(include_jobs=False))


async def _get_batch(app, batch_id, user, limit=None, offset=None):
    batch = await Batch.from_db(app, batch_id, user)
    if not batch:
        abort(404)
    return await batch.to_dict(include_jobs=True, limit=limit, offset=offset)


async def _cancel_batch(app, batch_id, user):
    batch = await Batch.from_db(app, batch_id, user)
    if not batch:
        abort(404)
    asyncio.ensure_future(batch.cancel())


@routes.get('/api/v1alpha/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_POST_GET_BATCH)
@rest_authenticated_users_only
async def get_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    params = request.query
    limit = params.get('limit')
    offset = params.get('offset')
    return jsonify(await _get_batch(request.app, batch_id, user, limit=limit, offset=offset))


@routes.patch('/api/v1alpha/batches/{batch_id}/cancel')
@prom_async_time(REQUEST_TIME_PATCH_CANCEL_BATCH)
@rest_authenticated_users_only
async def cancel_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    await _cancel_batch(request.app, batch_id, user)
    return jsonify({})


@routes.patch('/api/v1alpha/batches/{batch_id}/close')
@prom_async_time(REQUEST_TIME_PATCH_CLOSE_BATCH)
@rest_authenticated_users_only
async def close_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    batch = await Batch.from_db(request.app, batch_id, user)
    if not batch:
        abort(404)
    await batch.close()
    return jsonify({})


@routes.delete('/api/v1alpha/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_DELETE_BATCH)
@rest_authenticated_users_only
async def delete_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    batch = await Batch.from_db(request.app, batch_id, user)
    if not batch:
        abort(404)
    asyncio.ensure_future(batch.mark_deleted())
    return jsonify({})


@routes.get('/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_GET_BATCH_UI)
@web_authenticated_users_only()
async def ui_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    params = request.query
    limit = params.get('limit')
    offset = params.get('offset')
    page_context = {
        'batch': await _get_batch(request.app, batch_id, user, limit=limit, offset=offset)
    }
    return await render_template('batch2', request, userdata, 'batch.html', page_context)


@routes.post('/batches/{batch_id}/cancel')
@prom_async_time(REQUEST_TIME_POST_CANCEL_BATCH_UI)
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def ui_cancel_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    await _cancel_batch(request.app, batch_id, user)
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


@routes.get('/batches', name='batches')
@prom_async_time(REQUEST_TIME_GET_BATCHES_UI)
@web_authenticated_users_only()
async def ui_batches(request, userdata):
    params = request.query
    user = userdata['username']
    batches = await _get_batches_list(request.app, params, user)
    page_context = {
        'batch_list': batches[::-1]
    }
    return await render_template('batch2', request, userdata, 'batches.html', page_context)


@routes.get('/batches/{batch_id}/jobs/{job_id}/log')
@prom_async_time(REQUEST_TIME_GET_LOGS_UI)
@web_authenticated_users_only()
async def ui_get_job_log(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    page_context = {
        'batch_id': batch_id,
        'job_id': job_id,
        'job_log': await _get_job_log(request.app, batch_id, job_id, user)
    }
    return await render_template('batch2', request, userdata, 'job_log.html', page_context)


@routes.get('/batches/{batch_id}/jobs/{job_id}/pod_status')
@prom_async_time(REQUEST_TIME_GET_POD_STATUS_UI)
@aiohttp_jinja2.template('pod_status.html')
@web_authenticated_users_only()
async def ui_get_pod_status(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    page_context = {
        'batch_id': batch_id,
        'job_id': job_id,
        'pod_status': json.dumps(
            json.loads(await _get_pod_status(request.app, batch_id, job_id, user)), indent=2)
    }
    return await render_template('batch2', request, userdata, 'pod_status.html', page_context)


@routes.get('')
@routes.get('/')
@web_authenticated_users_only()
async def index(request, userdata):
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


async def update_job_with_pod(app, job, pod):  # pylint: disable=R0911
    log.info(f'update job {job.id if job else "None"} with pod {pod.metadata.name if pod else "None"}')
    if job and job._state == 'Pending':
        if pod:
            log.error(f'job {job.id} has pod {pod.metadata.name}, ignoring')
        return

    if pod and (not job or job.is_complete()):
        err = await app['driver'].delete_pod(name=pod.metadata.name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            log.info(f'failed to delete pod {pod.metadata.name} for job {job.id if job else "None"} due to {err}, ignoring')
        return

    if job and job._cancelled and not job.always_run and job._state == 'Running':
        await job.set_state('Cancelled')
        await job._delete_pod()
        return

    if pod and pod.status and pod.status.phase == 'Pending':
        all_container_statuses = pod.status.container_statuses or []
        for container_status in all_container_statuses:
            if (container_status.state and
                    container_status.state.waiting and
                    container_status.state.waiting.reason):
                await job.mark_complete(pod)
                return

    if not pod:
        log.info(f'job {job.id} no pod found, rescheduling')
        await job.mark_unscheduled()
        return

    if pod and pod.status and pod.status.reason == 'Evicted':
        POD_EVICTIONS.inc()
        log.info(f'job {job.id} mark unscheduled -- pod was evicted')
        await job.mark_unscheduled()
        return

    if pod and pod.status and pod.status.phase in ('Succeeded', 'Failed'):
        log.info(f'job {job.id} mark complete')
        await job.mark_complete(pod)
        return

    if pod and pod.status and pod.status.phase == 'Unknown':
        log.info(f'job {job.id} mark unscheduled -- pod phase is unknown')
        await job.mark_unscheduled()
        return


async def pod_changed(app, pod):
    job = await Job.from_pod(app, pod)
    await update_job_with_pod(app, job, pod)


async def refresh_pods(app):
    log.info(f'refreshing pods')

    pod_jobs = [Job.from_record(app, record) for record in await app['db'].jobs.get_records_where({'state': 'Running'})]

    pods = app['driver'].list_pods()
    log.info(f'batch had {len(pods)} pods')

    seen_pods = set()
    for pod_dict in pods:
        pod = app['k8s_client'].api_client._ApiClient__deserialize(pod_dict, kube.client.V1Pod)
        pod_name = pod.metadata.name
        seen_pods.add(pod_name)
        asyncio.ensure_future(pod_changed(app, pod))

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
            object = await app['driver'].complete_queue.get()
            pod = app['k8s_client'].api_client._ApiClient__deserialize(object, kube.client.V1Pod)
            await pod_changed(app, pod)
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
# @rest_authenticated_users_only
async def activate_worker(request):
    return await asyncio.shield(request.app['driver'].activate_worker(request))


@routes.post('/api/v1alpha/instances/deactivate')
# @rest_authenticated_users_only
async def deactivate_worker(request):
    return await asyncio.shield(request.app['driver'].deactivate_worker(request))


@routes.post('/api/v1alpha/instances/pod_complete')
# @rest_authenticated_users_only
async def pod_complete(request):
    return await asyncio.shield(request.app['driver'].pod_complete(request))


async def on_startup(app):
    pool = concurrent.futures.ThreadPoolExecutor()
    app['blocking_pool'] = pool

    if 'BATCH_USE_KUBE_CONFIG' in os.environ:
        kube.config.load_kube_config()
    else:
        kube.config.load_incluster_config()
    v1 = kube.client.CoreV1Api()
    app['k8s_client'] = v1

    k8s = K8s(pool, KUBERNETES_TIMEOUT_IN_SECONDS, BATCH_NAMESPACE, v1)

    userinfo = await async_get_userinfo()
    log.info(f'running as {userinfo["username"]}')

    bucket_name = userinfo['bucket_name']
    log.info(f'bucket_name {bucket_name}')

    driver = Driver(k8s, bucket_name)
    app['driver'] = driver
    app['log_store'] = LogStore(pool, INSTANCE_ID, bucket_name)
    app['db'] = get_db()

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
    setup_aiohttp_session(app)

    setup_aiohttp_jinja2(app, 'batch')
    setup_common_static_routes(routes)
    app.add_routes(routes)
    app.router.add_get("/metrics", server_stats)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    web.run_app(deploy_config.prefix_application(app, 'batch2'), host='0.0.0.0', port=5000)
