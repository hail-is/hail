import os
import concurrent
import logging
import json
import asyncio
import aiohttp
from aiohttp import web
import aiohttp_session
import humanize
import cerberus
import prometheus_client as pc
from prometheus_async.aio import time as prom_async_time
from prometheus_async.aio.web import server_stats
import google.oauth2.service_account
import google.api_core.exceptions
from hailtop.utils import request_retry_transient_errors
from hailtop.auth import async_get_userinfo
from hailtop.config import get_deploy_config
from hailtop import batch_client
from hailtop.batch_client.aioclient import Job
from gear import Database, setup_aiohttp_session, \
    rest_authenticated_users_only, web_authenticated_users_only, \
    check_csrf_token
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, render_template, \
    set_message

# import uvloop

from ..utils import parse_cpu_in_mcpu, LoggingTimer
from ..batch import batch_record_to_dict, job_record_to_dict
from ..log_store import LogStore
from ..database import CallError, check_call_procedure
from ..batch_configuration import BATCH_PODS_NAMESPACE

from . import schemas

# uvloop.install()

log = logging.getLogger('batch.front_end')

REQUEST_TIME = pc.Summary('batch2_request_latency_seconds', 'Batch request latency in seconds', ['endpoint', 'verb'])
REQUEST_TIME_GET_JOBS = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs', verb="GET")
REQUEST_TIME_GET_JOB = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id', verb="GET")
REQUEST_TIME_GET_JOBS = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs', verb="GET")
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
REQUEST_TIME_GET_JOB_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id/jobs/job_id', verb="GET")

routes = web.RouteTableDef()

deploy_config = get_deploy_config()

BATCH_JOB_DEFAULT_CPU = os.environ.get('HAIL_BATCH_JOB_DEFAULT_CPU', '1')
BATCH_JOB_DEFAULT_MEMORY = os.environ.get('HAIL_BATCH_JOB_DEFAULT_MEMORY', '3.75G')


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


async def _get_batch_jobs(app, batch_id, user):
    db = app['db']

    record = await db.execute_and_fetchone(
        '''
SELECT id FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''', (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    jobs = [
        job_record_to_dict(record)
        async for record
        in db.execute_and_fetchall(
            'SELECT * FROM jobs WHERE batch_id = %s',
            (batch_id,))
    ]
    return sorted(jobs, key=lambda j: j['job_id'])


@routes.get('/api/v1alpha/batches/{batch_id}/jobs')
@prom_async_time(REQUEST_TIME_GET_JOBS)
@rest_authenticated_users_only
async def get_jobs(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    return web.json_response(await _get_batch_jobs(request.app, batch_id, user))


async def _get_job_log_from_record(app, batch_id, job_id, record):
    state = record['state']
    ip_address = record['ip_address']
    if state == 'Running':
        async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            try:
                url = (f'http://{ip_address}:5000'
                       f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
                resp = await request_retry_transient_errors(session, 'GET', url)
                return await resp.json()
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    return None
                raise

    if state in ('Error', 'Failed', 'Success'):
        log_store = app['log_store']

        async def _read_log_from_gcs(task):
            try:
                log = await log_store.read_log_file(batch_id, job_id, task)
            except google.api_core.exceptions.NotFound:
                log = None
            return task, log

        spec = json.loads(record['spec'])
        tasks = []
        input_files = spec.get('input_files')
        if input_files:
            tasks.append('input')
        tasks.append('main')
        output_files = spec.get('output_files')
        if output_files:
            tasks.append('output')

        return dict(await asyncio.gather(*[_read_log_from_gcs(task) for task in tasks]))

    return None


async def _get_job_log(app, batch_id, job_id, user):
    db = app['db']

    record = await db.execute_and_fetchone('''
SELECT jobs.state, jobs.spec, ip_address
FROM jobs
INNER JOIN batches
  ON jobs.batch_id = batches.id
LEFT JOIN instances
  ON jobs.instance_name = instances.name
WHERE user = %s AND batch_id = %s AND NOT deleted AND job_id = %s;
''',
                                           (user, batch_id, job_id))
    if not record:
        raise web.HTTPNotFound()
    return await _get_job_log_from_record(app, batch_id, job_id, record)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
@prom_async_time(REQUEST_TIME_GET_JOB_LOG)
@rest_authenticated_users_only
async def get_job_log(request, userdata):  # pylint: disable=R1710
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    job_log = await _get_job_log(request.app, batch_id, job_id, user)
    return web.json_response(job_log)


async def _get_batches(app, params, user):
    db = app['db']

    where_conditions = ['user = %s', 'NOT deleted']
    where_args = [user]

    complete = params.get('complete')
    if complete is not None:
        complete_expr = '(closed AND n_completed = n_jobs)'
        if complete == '1':
            where_conditions.append(complete_expr)
        else:
            where_conditions.append(f'(NOT {complete_expr})')
    success = params.get('success')
    if success is not None:
        success_expr = '(closed AND n_succeeded = n_jobs)'
        if success == '1':
            where_conditions.append(success_expr)
        else:
            where_conditions.append(f'(NOT {success_expr})')

    for k, v in params.items():
        if k in ('complete', 'success'):  # params does not support deletion
            continue
        if not k.startswith('a:'):
            raise web.HTTPBadRequest(reason=f'unknown query parameter {k}')

        where_conditions.append('''
(EXISTS (SELECT * FROM `batch_attributes`
         WHERE `batch_attributes`.batch_id = batches.id AND
           `batch_attributes`.`key` = %s AND `batch_attributes`.`value` = %s))
''')
        where_args.append(k[2:])
        where_args.append(v)

    sql = f'''
SELECT * FROM batches
WHERE {" AND ".join(where_conditions)};
'''

    return [batch_record_to_dict(record)
            async for record in db.execute_and_fetchall(sql, where_args)]


@routes.get('/api/v1alpha/batches')
@prom_async_time(REQUEST_TIME_GET_BATCHES)
@rest_authenticated_users_only
async def get_batches(request, userdata):
    params = request.query
    user = userdata['username']
    return web.json_response(await _get_batches(request.app, params, user))


@routes.post('/api/v1alpha/batches/{batch_id}/jobs/create')
@prom_async_time(REQUEST_TIME_POST_CREATE_JOBS)
@rest_authenticated_users_only
async def create_jobs(request, userdata):
    app = request.app
    db = app['db']

    batch_id = int(request.match_info['batch_id'])

    user = userdata['username']
    # restrict to what's necessary; in particular, drop the session
    # which is sensitive
    userdata = {
        'username': user,
        'bucket_name': userdata['bucket_name'],
        'gsa_key_secret_name': userdata['gsa_key_secret_name'],
        'jwt_secret_name': userdata['jwt_secret_name']
    }

    async with LoggingTimer(f'batch {batch_id} create jobs') as timer:
        async with timer.step('fetch batch'):
            record = await db.execute_and_fetchone(
                '''
SELECT closed FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''',
                (user, batch_id))

        if not record:
            raise web.HTTPNotFound()
        if record['closed']:
            raise web.HTTPBadRequest(reason=f'batch {batch_id} is already closed')

        async with timer.step('get request json'):
            job_specs = await request.json()

        async with timer.step('validate job_specs'):
            try:
                batch_client.validate.validate_jobs(job_specs)
            except batch_client.validate.ValidationError as e:
                raise web.HTTPBadRequest(reason=e.reason)

        async with timer.step('build db args'):
            jobs_args = []
            job_parents_args = []
            job_attributes_args = []

            for spec in job_specs:
                job_id = spec['job_id']
                parent_ids = spec.pop('parent_ids', [])
                always_run = spec.pop('always_run', False)
                attributes = spec.get('attributes')

                resources = spec.get('resources')
                if not resources:
                    resources = {}
                    spec['resources'] = resources
                if 'cpu' not in resources:
                    resources['cpu'] = BATCH_JOB_DEFAULT_CPU
                if 'memory' not in resources:
                    resources['memory'] = BATCH_JOB_DEFAULT_MEMORY

                cores_mcpu = parse_cpu_in_mcpu(resources['cpu'])

                secrets = spec.get('secrets')
                if not secrets:
                    secrets = []
                    spec['secrets'] = secrets
                secrets.append({
                    'namespace': BATCH_PODS_NAMESPACE,
                    'name': userdata['gsa_key_secret_name'],
                    'mount_path': '/gsa-key',
                    'mount_in_copy': True
                })

                env = spec.get('env')
                if not env:
                    env = []
                    spec['env'] = env

                state = 'Ready' if len(parent_ids) == 0 else 'Pending'

                jobs_args.append(
                    (batch_id, job_id, state, json.dumps(spec),
                     always_run, cores_mcpu, len(parent_ids)))

                for parent_id in parent_ids:
                    job_parents_args.append(
                        (batch_id, job_id, parent_id))

                if attributes:
                    for k, v in attributes.items():
                        job_attributes_args.append(
                            (batch_id, job_id, k, v))

        async with timer.step('insert jobs'):
            async with db.pool.acquire() as conn:
                await conn.begin()
                async with conn.cursor() as cursor:
                    await cursor.executemany('''
INSERT INTO jobs (batch_id, job_id, state, spec, always_run, cores_mcpu, n_pending_parents)
VALUES (%s, %s, %s, %s, %s, %s, %s);
''',
                                             jobs_args)
                async with conn.cursor() as cursor:
                    await cursor.executemany('''
INSERT INTO `job_parents` (batch_id, job_id, parent_id)
VALUES (%s, %s, %s);
''',
                                             job_parents_args)
                async with conn.cursor() as cursor:
                    await cursor.executemany('''
INSERT INTO `job_attributes` (batch_id, job_id, `key`, `value`)
VALUES (%s, %s, %s, %s);
''',
                                             job_attributes_args)
                await conn.commit()

        return web.Response()


@routes.post('/api/v1alpha/batches/create')
@prom_async_time(REQUEST_TIME_POST_CREATE_BATCH)
@rest_authenticated_users_only
async def create_batch(request, userdata):
    app = request.app
    db = app['db']

    batch_spec = await request.json()

    validator = cerberus.Validator(schemas.batch_schema)
    if not validator.validate(batch_spec):
        raise web.HTTPBadRequest(reason=f'invalid request: {validator.errors}')

    user = userdata['username']
    attributes = batch_spec.get('attributes')
    async with db.pool.acquire() as conn:
        await conn.begin()
        async with conn.cursor() as cursor:
            await cursor.execute(
                '''
INSERT INTO batches (userdata, user, attributes, callback, n_jobs)
VALUES (%s, %s, %s, %s, %s);
''',
                (json.dumps(userdata), user, json.dumps(attributes),
                 batch_spec.get('callback'), batch_spec['n_jobs']))
            id = cursor.lastrowid

        if attributes:
            async with conn.cursor() as cursor:
                await cursor.executemany(
                    '''
INSERT INTO `batch_attributes` (batch_id, `key`, `value`)
VALUES (%s, %s, %s)
''',
                    [(id, k, v) for k, v in attributes.items()])
        await conn.commit()

    return web.json_response({'id': id})


async def _get_batch(app, batch_id, user):
    db = app['db']

    record = await db.execute_and_fetchone(
        '''
SELECT * FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''', (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    return batch_record_to_dict(record)


async def _cancel_batch(app, batch_id, user):
    db = app['db']

    record = await db.execute_and_fetchone(
        '''
SELECT closed FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()
    if not record['closed']:
        raise web.HTTPBadRequest(reason='cannot cancel open batch {batch_id}')

    await db.execute_update(
        'UPDATE batches SET cancelled = closed WHERE id = %s;', (batch_id,))

    async with aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
        await request_retry_transient_errors(
            session, 'PATCH',
            deploy_config.url('batch2-driver', f'/api/v1alpha/batches/{user}/{batch_id}/cancel'),
            headers=app['driver_headers'])

    return web.Response()


@routes.get('/api/v1alpha/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_POST_GET_BATCH)
@rest_authenticated_users_only
async def get_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    return web.json_response(await _get_batch(request.app, batch_id, user))


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

    app = request.app
    db = app['db']

    record = await db.execute_and_fetchone(
        '''
SELECT closed FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    try:
        await check_call_procedure(
            db, 'CALL close_batch(%s);', (batch_id))
    except CallError as e:
        # 2: wrong number of jobs
        if e.rv['rc'] == 2:
            expected_n_jobs = e.rv['expected_n_jobs']
            actual_n_jobs = e.rv['actual_n_jobs']
            raise web.HTTPBadRequest(
                reason=f'wrong number of jobs: expected {expected_n_jobs}, actual {actual_n_jobs}')
        raise

    async with aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
        await request_retry_transient_errors(
            session, 'PATCH',
            deploy_config.url('batch2-driver', f'/api/v1alpha/batches/{user}/{batch_id}/close'),
            headers=app['driver_headers'])

    return web.Response()


@routes.delete('/api/v1alpha/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_DELETE_BATCH)
@rest_authenticated_users_only
async def delete_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']

    app = request.app
    db = app['db']

    record = await db.execute_and_fetchone(
        '''
SELECT closed FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    await db.execute_update(
        'UPDATE batches SET cancelled = closed, deleted = 1 WHERE id = %s;', (batch_id,))

    if record['closed']:
        async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            try:
                await request_retry_transient_errors(
                    session, 'DELETE',
                    deploy_config.url('batch2-driver', f'/api/v1alpha/batches/{user}/{batch_id}'),
                    headers=app['driver_headers'])
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
    app = request.app
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']

    batch = await _get_batch(app, batch_id, user)
    jobs = await _get_batch_jobs(app, batch_id, user)
    for job in jobs:
        job['exit_code'] = Job.exit_code(job)
        job['duration'] = humanize.naturaldelta(Job.total_duration(job))
    batch['jobs'] = jobs
    page_context = {
        'batch': batch
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
    session = await aiohttp_session.get_session(request)
    set_message(session, 'Batch {batch_id} cancelled.', 'info')
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


@routes.get('/batches', name='batches')
@prom_async_time(REQUEST_TIME_GET_BATCHES_UI)
@web_authenticated_users_only()
async def ui_batches(request, userdata):
    params = request.query
    user = userdata['username']
    batches = await _get_batches(request.app, params, user)
    page_context = {
        'batch_list': batches[::-1]
    }
    return await render_template('batch2', request, userdata, 'batches.html', page_context)


async def _get_job_running_status(record):
    state = record['state']
    if state != 'Running':
        return None

    assert record['status'] is None

    batch_id = record['batch_id']
    job_id = record['job_id']
    ip_address = record['ip_address']
    async with aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
        try:
            url = (f'http://{ip_address}:5000'
                   f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}/status')
            resp = await request_retry_transient_errors(session, 'GET', url)
            return await resp.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return None
            raise


async def _get_job(app, batch_id, job_id, user):
    db = app['db']

    record = await db.execute_and_fetchone('''
SELECT jobs.*, ip_address
FROM jobs
INNER JOIN batches
  ON jobs.batch_id = batches.id
LEFT JOIN instances
  ON jobs.instance_name = instances.name
WHERE user = %s AND batch_id = %s AND NOT deleted AND job_id = %s;
''',
                                           (user, batch_id, job_id))
    if not record:
        raise web.HTTPNotFound()

    running_status = await _get_job_running_status(record)
    return job_record_to_dict(record, running_status)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}')
@prom_async_time(REQUEST_TIME_GET_JOB)
@rest_authenticated_users_only
async def get_job(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']

    status = await _get_job(request.app, batch_id, job_id, user)
    return web.json_response(status)


@routes.get('/batches/{batch_id}/jobs/{job_id}')
@prom_async_time(REQUEST_TIME_GET_JOB_UI)
@web_authenticated_users_only()
async def ui_get_job(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']

    job_status = await _get_job(request.app, batch_id, job_id, user)
    page_context = {
        'batch_id': batch_id,
        'job_id': job_id,
        'job_log': await _get_job_log(request.app, batch_id, job_id, user),
        'job_status': json.dumps(job_status, indent=2)
    }
    return await render_template('batch2', request, userdata, 'job.html', page_context)


@routes.get('')
@routes.get('/')
@web_authenticated_users_only()
async def index(request, userdata):
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


async def on_startup(app):
    userinfo = await async_get_userinfo()
    log.info(f'running as {userinfo["username"]}')

    bucket_name = userinfo['bucket_name']
    log.info(f'bucket_name {bucket_name}')

    pool = concurrent.futures.ThreadPoolExecutor()
    app['blocking_pool'] = pool

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
    app['driver_headers'] = {
        'Authorization': f'Bearer {row["token"]}'
    }

    credentials = google.oauth2.service_account.Credentials.from_service_account_file(
        '/batch-gsa-key/privateKeyData')
    app['log_store'] = LogStore(bucket_name, instance_id, pool, credentials=credentials)


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
