import os
import concurrent
import logging
import json
import time
import asyncio
import aiohttp
from aiohttp import web
import aiohttp_jinja2
import cerberus
import prometheus_client as pc
from prometheus_async.aio import time as prom_async_time
from prometheus_async.aio.web import server_stats

from hailtop.utils import request_retry_transient_errors
from hailtop.auth import async_get_userinfo
from hailtop.config import get_deploy_config
from hailtop import batch_client
from gear import execute_and_fetchone, setup_aiohttp_session, \
    rest_authenticated_users_only, web_authenticated_users_only, \
    check_csrf_token
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, render_template

# import uvloop

from ..globals import tasks
from ..utils import parse_cpu_in_mcpu
from ..batch import Batch, job_record_to_dict
from ..log_store import LogStore
from ..database import BatchDatabase, JobsBuilder
from ..batch_configuration import INSTANCE_ID

from . import schemas

# uvloop.install()

log = logging.getLogger('batch.front_end')

REQUEST_TIME = pc.Summary('batch2_request_latency_seconds', 'Batch request latency in seconds', ['endpoint', 'verb'])
REQUEST_TIME_GET_JOB = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id', verb="GET")
REQUEST_TIME_GET_JOB_LOG = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id/log', verb="GET")
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
REQUEST_TIME_GET_JOB_STATUS_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id/jobs/job_id/status', verb="GET")

log.info(f'INSTANCE_ID = {INSTANCE_ID}')

routes = web.RouteTableDef()

deploy_config = get_deploy_config()

BATCH_JOB_DEFAULT_CPU = os.environ.get('HAIL_BATCH_JOB_DEFAULT_CPU', '1')
BATCH_JOB_DEFAULT_MEMORY = os.environ.get('HAIL_BATCH_JOB_DEFAULT_MEMORY', '3.75G')


def create_job(app, userdata, jobs_builder, batch_id, job_spec):  # pylint: disable=R0912
    job_id = job_spec['job_id']
    parent_ids = job_spec.pop('parent_ids', [])
    always_run = job_spec.pop('always_run', False)

    resources = job_spec.get('resources')
    if not resources:
        resources = {}
        job_spec['resources'] = resources
    if 'cpu' not in resources:
        resources['cpu'] = BATCH_JOB_DEFAULT_CPU
    if 'memory' not in resources:
        resources['memory'] = BATCH_JOB_DEFAULT_MEMORY

    secrets = job_spec.get('secrets')
    if not secrets:
        secrets = []
        job_spec['secrets'] = secrets
    secrets.append({
        'namespace': 'batch-pods',  # FIXME unused
        'name': userdata['gsa_key_secret_name'],
        'mount_path': '/gsa-key',
        'mount_in_copy': True
    })

    state = 'Ready' if len(parent_ids) == 0 else 'Pending'

    directory = app['log_store'].gs_job_output_directory(batch_id, job_id)

    jobs_builder.create_job(
        batch_id=batch_id,
        job_id=job_id,
        state=state,
        directory=directory,
        spec=json.dumps(job_spec),
        always_run=always_run,
        cores_mcpu=parse_cpu_in_mcpu(resources['cpu']),
        instance_id=None,
        status=None,
        n_pending_parents=len(parent_ids),
        cancel=0)

    for parent in parent_ids:
        jobs_builder.create_job_parent(
            batch_id=batch_id,
            job_id=job_id,
            parent_id=parent)


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}')
@prom_async_time(REQUEST_TIME_GET_JOB)
@rest_authenticated_users_only
async def get_job(request, userdata):
    db = request.app['db']

    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']

    record = await execute_and_fetchone(
        db.pool, '''
SELECT *
FROM jobs
INNER JOIN batch
  ON jobs.batch_id = batch.id
WHERE batch_id = %s AND job_id = %s AND user = %s
''',
        (batch_id, job_id, user))

    if not record:
        raise web.HTTPNotFound()
    return web.json_response(job_record_to_dict(record))


async def _get_job_log_from_record(app, batch_id, job_id, record):
    state = record['state']

    ip_address = record['ip_address']
    if state == 'Running' and ip_address is not None:
        async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            try:
                url = (f'http://{ip_address}:5000'
                       f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}/logs')
                resp = await request_retry_transient_errors(session, 'GET', url)
                return await resp.json()
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    return None
                raise

    if state in ('Error', 'Failed', 'Success'):
        log_store = app['log_store']
        directory = record['directory']

        async def _read_log_from_gcs(task):
            log = await log_store.read_gs_file(LogStore.container_log_path(directory, task))
            return task, log

        return dict(await asyncio.gather(*[_read_log_from_gcs(task) for task in tasks]))

    return None


async def _get_job_log(app, batch_id, job_id, user):
    db = app['db']

    record = await execute_and_fetchone(db.pool, '''
SELECT jobs.state, ip_address, directory
FROM jobs
INNER JOIN batch
  ON jobs.batch_id = batch.id
LEFT JOIN instances
  ON jobs.instance_id = instances.id
WHERE batch_id = %s AND job_id = %s AND user = %s;
''',
                                        (batch_id, job_id, user))
    if not record:
        raise web.HTTPNotFound()
    log = await _get_job_log_from_record(app, batch_id, job_id, record)
    if log:
        return log
    raise web.HTTPNotFound()


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
@prom_async_time(REQUEST_TIME_GET_JOB_LOG)
@rest_authenticated_users_only
async def get_job_log(request, userdata):  # pylint: disable=R1710
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    job_log = await _get_job_log(request.app, batch_id, job_id, user)
    return web.json_response(job_log)


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
            raise web.HTTPBadRequest(reason=f'unknown query parameter {k}')
        attributes[k[2:]] = v

    records = await app['db'].batch.find_records(user=user,
                                                 complete=complete,
                                                 success=success,
                                                 deleted=False,
                                                 attributes=attributes)

    return [await Batch.from_record(app['db'], batch).to_dict(include_jobs=False)
            for batch in records]


@routes.get('/api/v1alpha/batches')
@prom_async_time(REQUEST_TIME_GET_BATCHES)
@rest_authenticated_users_only
async def get_batches_list(request, userdata):
    params = request.query
    user = userdata['username']
    return web.json_response(await _get_batches_list(request.app, params, user))


@routes.post('/api/v1alpha/batches/{batch_id}/jobs/create')
@prom_async_time(REQUEST_TIME_POST_CREATE_JOBS)
@rest_authenticated_users_only
async def create_jobs(request, userdata):
    start = time.time()
    app = request.app
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']

    start1 = time.time()
    batch = await Batch.from_db(app['db'], batch_id, user)
    log.info(f'took {round(time.time() - start1, 3)} seconds to get batch from db')

    if not batch:
        raise web.HTTPNotFound()
    if batch.closed:
        raise web.HTTPBadRequest(reason=f'batch {batch_id} is already closed')

    start2 = time.time()
    job_specs = await request.json()
    log.info(f'took {round(time.time() - start2, 3)} seconds to get data from server')

    start3 = time.time()
    try:
        batch_client.validate.validate_jobs(job_specs)
    except batch_client.validate.ValidationError as e:
        raise web.HTTPBadRequest(reason=e.reason)
    log.info(f"took {round(time.time() - start3, 3)} seconds to validate spec")

    start4 = time.time()
    jobs_builder = JobsBuilder(app['db'])
    try:
        for job_spec in job_specs:
            create_job(app, userdata, jobs_builder, batch.id, job_spec)

        success = await jobs_builder.commit()
        if not success:
            raise web.HTTPBadRequest(reason=f'insertion of jobs in db failed')
    finally:
        await jobs_builder.close()

    log.info(f'took {round(time.time() - start4, 3)} seconds to commit jobs to db')

    log.info(f'took {round(time.time() - start, 3)} seconds to create jobs from start to finish')
    return web.Response()


@routes.post('/api/v1alpha/batches/create')
@prom_async_time(REQUEST_TIME_POST_CREATE_BATCH)
@rest_authenticated_users_only
async def create_batch(request, userdata):
    start = time.time()
    parameters = await request.json()

    validator = cerberus.Validator(schemas.batch_schema)
    if not validator.validate(parameters):
        raise web.HTTPBadRequest(reason='invalid request: {}'.format(validator.errors))

    batch = await Batch.create_batch(
        request.app['db'],
        attributes=parameters.get('attributes'),
        callback=parameters.get('callback'),
        userdata=userdata,
        n_jobs=parameters['n_jobs'])
    if batch is None:
        raise web.HTTPBadRequest(reason=f'creation of batch in db failed')

    log.info(f'took {round(time.time() - start, 3)} seconds to initialize batch {batch.id} in db')
    return web.json_response(await batch.to_dict(include_jobs=False))


async def _get_batch(app, batch_id, user, include_jobs):
    batch = await Batch.from_db(app['db'], batch_id, user)
    if not batch:
        raise web.HTTPNotFound()
    return await batch.to_dict(include_jobs=include_jobs)


async def _cancel_batch(app, batch_id, user):
    batch = await Batch.from_db(app['db'], batch_id, user)
    if not batch:
        raise web.HTTPNotFound()
    if not batch.closed:
        raise web.HTTPBadRequest(reason='cannot cancel open batch')
    await batch.cancel()
    async with aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
        await request_retry_transient_errors(
            session, 'PATCH',
            deploy_config.url('batch2-driver', f'/api/v1alpha/batches/{user}/{batch_id}/cancel'))
    return web.Response()


@routes.get('/api/v1alpha/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_POST_GET_BATCH)
@rest_authenticated_users_only
async def get_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    params = request.query
    include_jobs = params.get('include_jobs') == '1'
    return web.json_response(await _get_batch(request.app, batch_id, user, include_jobs))


@routes.patch('/api/v1alpha/batches/{batch_id}/cancel')
@prom_async_time(REQUEST_TIME_PATCH_CANCEL_BATCH)
@rest_authenticated_users_only
async def cancel_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    await _cancel_batch(request.app, batch_id, user)
    return web.Response()


@routes.patch('/api/v1alpha/batches/{batch_id}/close')
@prom_async_time(REQUEST_TIME_PATCH_CLOSE_BATCH)
@rest_authenticated_users_only
async def close_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    batch = await Batch.from_db(request.app['db'], batch_id, user)
    if not batch:
        raise web.HTTPNotFound()
    await batch.close()
    async with aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
        await request_retry_transient_errors(
            session, 'PATCH',
            deploy_config.url('batch2-driver', f'/api/v1alpha/batches/{user}/{batch_id}/close'))
    return web.Response()


@routes.delete('/api/v1alpha/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_DELETE_BATCH)
@rest_authenticated_users_only
async def delete_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    batch = await Batch.from_db(request.app['db'], batch_id, user)
    if not batch:
        raise web.HTTPNotFound()
    await batch.mark_deleted()
    if batch.closed:
        async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            try:
                await request_retry_transient_errors(
                    session, 'DELETE',
                    deploy_config.url('batch2-driver', f'/api/v1alpha/batches/{user}/{batch_id}'))
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    pass
                else:
                    raise
    return web.Response()


@routes.get('/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_GET_BATCH_UI)
@web_authenticated_users_only()
async def ui_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    page_context = {
        'batch': await _get_batch(request.app, batch_id, user, include_jobs=True)
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


@routes.get('/batches/{batch_id}/jobs/{job_id}/status')
@prom_async_time(REQUEST_TIME_GET_JOB_STATUS_UI)
@aiohttp_jinja2.template('job_status.html')
@web_authenticated_users_only()
async def ui_get_job_status(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']

    record = await execute_and_fetchone('''
SELECT status
FROM jobs
INNER JOIN batch
  ON jobs.batch_id = batch.id
WHERE batch_id = %s AND job_id = %s AND user = %s
''',
                                        (batch_id, job_id, user))
    if not record:
        raise web.HTTPNotFound()
    status = record['status']
    page_context = {
        'batch_id': batch_id,
        'job_id': job_id,
        'job_status': json.dumps(
            json.loads(status), indent=2)
    }
    return await render_template('batch2', request, userdata, 'job_status.html', page_context)


@routes.get('')
@routes.get('/')
@web_authenticated_users_only()
async def index(request, userdata):
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


async def on_startup(app):
    pool = concurrent.futures.ThreadPoolExecutor()
    app['blocking_pool'] = pool

    userinfo = await async_get_userinfo()
    log.info(f'running as {userinfo["username"]}')

    bucket_name = userinfo['bucket_name']
    log.info(f'bucket_name {bucket_name}')

    app['log_store'] = LogStore(pool, INSTANCE_ID, bucket_name)
    app['db'] = await BatchDatabase('/sql-config/sql-config.json')


async def on_cleanup(app):
    blocking_pool = app['blocking_pool']
    blocking_pool.shutdown()


def run():
    app = web.Application(client_max_size=None)
    setup_aiohttp_session(app)

    setup_aiohttp_jinja2(app, 'batch.front_end')
    setup_common_static_routes(routes)
    app.add_routes(routes)
    app.router.add_get("/metrics", server_stats)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    web.run_app(deploy_config.prefix_application(app, 'batch2'), host='0.0.0.0', port=5000)
