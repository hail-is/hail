import os
import concurrent
import logging
import json
import random
import datetime
import asyncio
import aiohttp
from aiohttp import web
import aiohttp_session
import prometheus_client as pc
import pymysql
from prometheus_async.aio import time as prom_async_time
from prometheus_async.aio.web import server_stats
import google.oauth2.service_account
import google.api_core.exceptions
from hailtop.utils import (time_msecs, time_msecs_str, humanize_timedelta_msecs,
                           request_retry_transient_errors, run_if_changed,
                           retry_long_running, LoggingTimer)
from hailtop.batch_client.parse import parse_cpu_in_mcpu, parse_memory_in_bytes
from hailtop.config import get_deploy_config
from hailtop.tls import get_server_ssl_context, ssl_client_session
from gear import (Database, setup_aiohttp_session,
                  rest_authenticated_users_only, web_authenticated_users_only,
                  web_authenticated_developers_only, check_csrf_token, transaction,
                  AccessLogger)
from web_common import (setup_aiohttp_jinja2, setup_common_static_routes,
                        render_template, set_message)

# import uvloop

from ..utils import (adjust_cores_for_memory_request, worker_memory_per_core_gb,
                     cost_from_msec_mcpu, adjust_cores_for_packability, coalesce)
from ..batch import batch_record_to_dict, job_record_to_dict
from ..log_store import LogStore
from ..database import CallError, check_call_procedure
from ..batch_configuration import (BATCH_PODS_NAMESPACE, BATCH_BUCKET_NAME,
                                   DEFAULT_NAMESPACE, WORKER_LOGS_BUCKET_NAME)
from ..globals import HTTP_CLIENT_MAX_SIZE, BATCH_FORMAT_VERSION
from ..spec_writer import SpecWriter
from ..batch_format_version import BatchFormatVersion

from .validate import ValidationError, validate_batch, validate_jobs

# uvloop.install()

log = logging.getLogger('batch.front_end')

REQUEST_TIME = pc.Summary('batch_request_latency_seconds', 'Batch request latency in seconds', ['endpoint', 'verb'])
REQUEST_TIME_GET_JOBS = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs', verb="GET")
REQUEST_TIME_GET_JOB = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id', verb="GET")
REQUEST_TIME_GET_JOB_LOG = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id/log', verb="GET")
REQUEST_TIME_GET_ATTEMPTS = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id/attempts', verb="GET")
REQUEST_TIME_GET_BATCHES = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches', verb="GET")
REQUEST_TIME_POST_CREATE_JOBS = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/create', verb="POST")
REQUEST_TIME_POST_CREATE_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/create', verb='POST')
REQUEST_TIME_POST_GET_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id', verb='GET')
REQUEST_TIME_PATCH_CANCEL_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/cancel', verb="PATCH")
REQUEST_TIME_PATCH_CLOSE_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/close', verb="PATCH")
REQUEST_TIME_DELETE_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id', verb="DELETE")
REQUEST_TIME_GET_BATCH_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id', verb='GET')
REQUEST_TIME_POST_CANCEL_BATCH_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id/cancel', verb='POST')
REQUEST_TIME_POST_DELETE_BATCH_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id/delete', verb='POST')
REQUEST_TIME_GET_BATCHES_UI = REQUEST_TIME.labels(endpoint='/batches', verb='GET')
REQUEST_TIME_GET_JOB_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id/jobs/job_id', verb="GET")
REQUEST_TIME_GET_BILLING_UI = REQUEST_TIME.labels(endpoint='/billing', verb="GET")
REQUEST_TIME_GET_BILLING_PROJECTS_UI = REQUEST_TIME.labels(endpoint='/billing_projects', verb="GET")
REQUEST_TIME_POST_BILLING_PROJECT_REMOVE_USER_UI = REQUEST_TIME.labels(endpoint='/billing_projects/billing_project/users/user/remove', verb="POST")
REQUEST_TIME_POST_BILLING_PROJECT_ADD_USER_UI = REQUEST_TIME.labels(endpoint='/billing_projects/billing_project/users/add', verb="POST")
REQUEST_TIME_POST_CREATE_BILLING_PROJECT_UI = REQUEST_TIME.labels(endpoint='/billing_projects/create', verb="POST")

routes = web.RouteTableDef()

deploy_config = get_deploy_config()

BATCH_JOB_DEFAULT_CPU = os.environ.get('HAIL_BATCH_JOB_DEFAULT_CPU', '1')
BATCH_JOB_DEFAULT_MEMORY = os.environ.get('HAIL_BATCH_JOB_DEFAULT_MEMORY', '3.75G')


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


async def _query_batch_jobs(request, batch_id):
    state_query_values = {
        'pending': ['Pending'],
        'ready': ['Ready'],
        'running': ['Running'],
        'live': ['Ready', 'Running'],
        'cancelled': ['Cancelled'],
        'error': ['Error'],
        'failed': ['Failed'],
        'bad': ['Error', 'Failed'],
        'success': ['Success'],
        'done': ['Cancelled', 'Error', 'Failed', 'Success']
    }

    db = request.app['db']

    # batch has already been validated
    where_conditions = [
        '(jobs.batch_id = %s)'
    ]
    where_args = [batch_id]

    last_job_id = request.query.get('last_job_id')
    if last_job_id is not None:
        last_job_id = int(last_job_id)
        where_conditions.append('(jobs.job_id > %s)')
        where_args.append(last_job_id)

    q = request.query.get('q', '')
    terms = q.split()
    for t in terms:
        if t[0] == '!':
            negate = True
            t = t[1:]
        else:
            negate = False

        if '=' in t:
            k, v = t.split('=', 1)
            condition = '''
(EXISTS (SELECT * FROM `job_attributes`
         WHERE `job_attributes`.batch_id = jobs.batch_id AND
           `job_attributes`.job_id = jobs.job_id AND
           `job_attributes`.`key` = %s AND
           `job_attributes`.`value` = %s))
'''
            args = [k, v]
        elif t.startswith('has:'):
            k = t[4:]
            condition = '''
(EXISTS (SELECT * FROM `job_attributes`
         WHERE `job_attributes`.batch_id = jobs.batch_id AND
           `job_attributes`.job_id = jobs.job_id AND
           `job_attributes`.`key` = %s))
'''
            args = [k]
        elif t in state_query_values:
            values = state_query_values[t]
            condition = ' OR '.join([
                '(jobs.state = %s)' for v in values])
            condition = f'({condition})'
            args = values
        else:
            session = await aiohttp_session.get_session(request)
            set_message(session, f'Invalid search term: {t}.', 'error')
            return ([], None)

        if negate:
            condition = f'(NOT {condition})'

        where_conditions.append(condition)
        where_args.extend(args)

    sql = f'''
SELECT jobs.*, batches.format_version, job_attributes.value AS name, SUM(`usage` * rate) AS cost
FROM jobs
INNER JOIN batches ON jobs.batch_id = batches.id
LEFT JOIN job_attributes
  ON jobs.batch_id = job_attributes.batch_id AND
     jobs.job_id = job_attributes.job_id AND
     job_attributes.`key` = 'name'
LEFT JOIN aggregated_job_resources
  ON jobs.batch_id = aggregated_job_resources.batch_id AND
     jobs.job_id = aggregated_job_resources.job_id
LEFT JOIN resources
  ON aggregated_job_resources.resource = resources.resource
WHERE {' AND '.join(where_conditions)}
GROUP BY jobs.batch_id, jobs.job_id
ORDER BY jobs.batch_id, jobs.job_id ASC
LIMIT 50;
'''
    sql_args = where_args

    jobs = [job_record_to_dict(record, record['name'])
            async for record
            in db.select_and_fetchall(sql, sql_args)]

    if len(jobs) == 50:
        last_job_id = jobs[-1]['job_id']
    else:
        last_job_id = None

    return (jobs, last_job_id)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs')
@prom_async_time(REQUEST_TIME_GET_JOBS)
@rest_authenticated_users_only
async def get_jobs(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']

    db = request.app['db']
    record = await db.select_and_fetchone(
        '''
SELECT * FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''', (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    jobs, last_job_id = await _query_batch_jobs(request, batch_id)
    resp = {
        'jobs': jobs
    }
    if last_job_id is not None:
        resp['last_job_id'] = last_job_id
    return web.json_response(resp)


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
        batch_format_version = BatchFormatVersion(record['format_version'])

        async def _read_log_from_gcs(task):
            try:
                data = await log_store.read_log_file(batch_format_version, batch_id, job_id, record['attempt_id'], task)
            except google.api_core.exceptions.NotFound:
                id = (batch_id, job_id)
                log.exception(f'missing log file for {id}')
                data = None
            return task, data

        spec = json.loads(record['spec'])
        tasks = []

        has_input_files = batch_format_version.get_spec_has_input_files(spec)
        if has_input_files:
            tasks.append('input')

        tasks.append('main')

        has_output_files = batch_format_version.get_spec_has_output_files(spec)
        if has_output_files:
            tasks.append('output')

        return dict(await asyncio.gather(*[_read_log_from_gcs(task) for task in tasks]))

    return None


async def _get_job_log(app, batch_id, job_id, user):
    db = app['db']

    record = await db.select_and_fetchone('''
SELECT jobs.state, jobs.spec, ip_address, format_version, jobs.attempt_id
FROM jobs
INNER JOIN batches
  ON jobs.batch_id = batches.id
LEFT JOIN attempts
  ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id AND jobs.attempt_id = attempts.attempt_id
LEFT JOIN instances
  ON attempts.instance_name = instances.name
WHERE user = %s AND jobs.batch_id = %s AND NOT deleted AND jobs.job_id = %s;
''',
                                          (user, batch_id, job_id))
    if not record:
        raise web.HTTPNotFound()
    return await _get_job_log_from_record(app, batch_id, job_id, record)


async def _get_attributes(app, record):
    db = app['db']

    batch_id = record['batch_id']
    job_id = record['job_id']
    format_version = BatchFormatVersion(record['format_version'])

    if not format_version.has_full_spec_in_gcs():
        spec = json.loads(record['spec'])
        return spec.get('attributes')

    records = db.select_and_fetchall('''
SELECT `key`, `value`
FROM job_attributes
WHERE batch_id = %s AND job_id = %s;
''',
                                     (batch_id, job_id))
    return {record['key']: record['value'] async for record in records}


async def _get_full_job_spec(app, record):
    db = app['db']
    log_store = app['log_store']

    batch_id = record['batch_id']
    job_id = record['job_id']
    format_version = BatchFormatVersion(record['format_version'])

    if not format_version.has_full_spec_in_gcs():
        return json.loads(record['spec'])

    token, start_job_id = await SpecWriter.get_token_start_id(db, batch_id, job_id)

    try:
        spec = await log_store.read_spec_file(batch_id, token, start_job_id, job_id)
        return json.loads(spec)
    except google.api_core.exceptions.NotFound:
        id = (batch_id, job_id)
        log.exception(f'missing spec file for {id}')
        return None


async def _get_full_job_status(app, record):
    log_store = app['log_store']

    batch_id = record['batch_id']
    job_id = record['job_id']
    attempt_id = record['attempt_id']
    state = record['state']
    format_version = BatchFormatVersion(record['format_version'])

    if state in ('Pending', 'Ready', 'Cancelled'):
        return

    if state in ('Error', 'Failed', 'Success'):
        if not format_version.has_full_status_in_gcs():
            return json.loads(record['status'])

        try:
            status = await log_store.read_status_file(batch_id, job_id, attempt_id)
            return json.loads(status)
        except google.api_core.exceptions.NotFound:
            id = (batch_id, job_id)
            log.exception(f'missing status file for {id}')
            return None

    assert state == 'Running'
    assert record['status'] is None

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


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
@prom_async_time(REQUEST_TIME_GET_JOB_LOG)
@rest_authenticated_users_only
async def get_job_log(request, userdata):  # pylint: disable=R1710
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    job_log = await _get_job_log(request.app, batch_id, job_id, user)
    return web.json_response(job_log)


async def _query_batches(request, user):
    db = request.app['db']

    where_conditions = ['user = %s', 'NOT deleted']
    where_args = [user]

    last_batch_id = request.query.get('last_batch_id')
    if last_batch_id is not None:
        last_batch_id = int(last_batch_id)
        where_conditions.append('(id < %s)')
        where_args.append(last_batch_id)

    q = request.query.get('q', '')
    terms = q.split()
    for t in terms:
        if t[0] == '!':
            negate = True
            t = t[1:]
        else:
            negate = False

        if '=' in t:
            k, v = t.split('=', 1)
            condition = '''
(EXISTS (SELECT * FROM `batch_attributes`
         WHERE `batch_attributes`.batch_id = id AND
           `batch_attributes`.`key` = %s AND
           `batch_attributes`.`value` = %s))
'''
            args = [k, v]
        elif t.startswith('has:'):
            k = t[4:]
            condition = '''
(EXISTS (SELECT * FROM `batch_attributes`
         WHERE `batch_attributes`.batch_id = id AND
           `batch_attributes`.`key` = %s))
'''
            args = [k]
        elif t == 'open':
            condition = "(`state` = 'open')"
            args = []
        elif t == 'closed':
            condition = "(`state` != 'open')"
            args = []
        elif t == 'complete':
            condition = "(`state` = 'complete')"
            args = []
        elif t == 'running':
            condition = "(`state` = 'running')"
            args = []
        elif t == 'cancelled':
            condition = '(cancelled)'
            args = []
        elif t == 'failure':
            condition = '(n_failed > 0)'
            args = []
        elif t == 'success':
            # need complete because there might be no jobs
            condition = "(`state` = 'complete' AND n_succeeded = n_jobs)"
            args = []
        else:
            session = await aiohttp_session.get_session(request)
            set_message(session, f'Invalid search term: {t}.', 'error')
            return ([], None)

        if negate:
            condition = f'(NOT {condition})'

        where_conditions.append(condition)
        where_args.extend(args)

    sql = f'''
SELECT batches.*, SUM(`usage` * rate) AS cost
FROM batches
LEFT JOIN aggregated_batch_resources
  ON batches.id = aggregated_batch_resources.batch_id
LEFT JOIN resources
  ON aggregated_batch_resources.resource = resources.resource
WHERE {' AND '.join(where_conditions)}
GROUP BY batches.id
ORDER BY batches.id DESC
LIMIT 50;
'''
    sql_args = where_args

    batches = [batch_record_to_dict(batch)
               async for batch
               in db.select_and_fetchall(sql, sql_args)]

    if len(batches) == 50:
        last_batch_id = batches[-1]['id']
    else:
        last_batch_id = None

    return (batches, last_batch_id)


@routes.get('/api/v1alpha/batches')
@prom_async_time(REQUEST_TIME_GET_BATCHES)
@rest_authenticated_users_only
async def get_batches(request, userdata):
    user = userdata['username']
    batches, last_batch_id = await _query_batches(request, user)
    body = {
        'batches': batches
    }
    if last_batch_id is not None:
        body['last_batch_id'] = last_batch_id
    return web.json_response(body)


def check_service_account_permissions(user, sa):
    if sa is None:
        return
    if user == 'ci':
        if sa['name'] in ('ci-agent', 'admin'):
            if DEFAULT_NAMESPACE == 'default':  # real-ci needs access to all namespaces
                return
            if sa['namespace'] == BATCH_PODS_NAMESPACE:
                return
    if user == 'test':
        if sa['name'] == 'test-batch-sa' and sa['namespace'] == BATCH_PODS_NAMESPACE:
            return
    raise web.HTTPBadRequest(reason=f'unauthorized service account {(sa["namespace"], sa["name"])} for user {user}')


@routes.post('/api/v1alpha/batches/{batch_id}/jobs/create')
@prom_async_time(REQUEST_TIME_POST_CREATE_JOBS)
@rest_authenticated_users_only
async def create_jobs(request, userdata):
    app = request.app
    db = app['db']
    log_store = app['log_store']

    worker_type = app['worker_type']
    worker_cores = app['worker_cores']

    batch_id = int(request.match_info['batch_id'])

    user = userdata['username']

    # restrict to what's necessary; in particular, drop the session
    # which is sensitive
    userdata = {
        'username': user,
        'gsa_key_secret_name': userdata['gsa_key_secret_name'],
        'tokens_secret_name': userdata['tokens_secret_name']
    }

    async with LoggingTimer(f'batch {batch_id} create jobs') as timer:
        async with timer.step('fetch batch'):
            record = await db.select_and_fetchone(
                '''
SELECT `state`, format_version FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''',
                (user, batch_id))

        if not record:
            raise web.HTTPNotFound()
        if record['state'] != 'open':
            raise web.HTTPBadRequest(reason=f'batch {batch_id} is not open')
        batch_format_version = BatchFormatVersion(record['format_version'])

        async with timer.step('get request json'):
            job_specs = await request.json()

        async with timer.step('validate job_specs'):
            try:
                validate_jobs(job_specs)
            except ValidationError as e:
                raise web.HTTPBadRequest(reason=e.reason)

        async with timer.step('build db args'):
            spec_writer = SpecWriter(log_store, batch_id)

            jobs_args = []
            job_parents_args = []
            job_attributes_args = []

            n_ready_jobs = 0
            ready_cores_mcpu = 0
            n_ready_cancellable_jobs = 0
            ready_cancellable_cores_mcpu = 0

            prev_job_idx = None
            start_job_id = None

            for spec in job_specs:
                job_id = spec['job_id']
                parent_ids = spec.pop('parent_ids', [])
                always_run = spec.pop('always_run', False)

                if batch_format_version.has_full_spec_in_gcs():
                    attributes = spec.pop('attributes', None)
                else:
                    attributes = spec.get('attributes')

                id = (batch_id, job_id)

                if start_job_id is None:
                    start_job_id = job_id

                if batch_format_version.has_full_spec_in_gcs() and prev_job_idx:
                    if job_id != prev_job_idx + 1:
                        raise web.HTTPBadRequest(
                            reason=f'noncontiguous job ids found in the spec: {prev_job_idx} -> {job_id}')
                prev_job_idx = job_id

                resources = spec.get('resources')
                if not resources:
                    resources = {}
                    spec['resources'] = resources
                if 'cpu' not in resources:
                    resources['cpu'] = BATCH_JOB_DEFAULT_CPU
                if 'memory' not in resources:
                    resources['memory'] = BATCH_JOB_DEFAULT_MEMORY

                req_cores_mcpu = parse_cpu_in_mcpu(resources['cpu'])
                req_memory_bytes = parse_memory_in_bytes(resources['memory'])

                if req_cores_mcpu == 0:
                    raise web.HTTPBadRequest(
                        reason=f'bad resource request for job {id}: '
                        f'cpu cannot be 0')

                cores_mcpu = adjust_cores_for_memory_request(req_cores_mcpu, req_memory_bytes, worker_type)
                cores_mcpu = adjust_cores_for_packability(cores_mcpu)

                if cores_mcpu > worker_cores * 1000:
                    total_memory_available = worker_memory_per_core_gb(worker_type) * worker_cores
                    raise web.HTTPBadRequest(
                        reason=f'resource requests for job {id} are unsatisfiable: '
                        f'requested: cpu={resources["cpu"]}, memory={resources["memory"]} '
                        f'maximum: cpu={worker_cores}, memory={total_memory_available}G')

                secrets = spec.get('secrets')
                if not secrets:
                    secrets = []

                if len(secrets) != 0 and user != 'ci':
                    secrets = [(secret["namespace"], secret["name"]) for secret in secrets]
                    raise web.HTTPBadRequest(reason=f'unauthorized secret {secrets} for user {user}')

                for secret in secrets:
                    if user != 'ci':
                        raise web.HTTPBadRequest(reason=f'unauthorized secret {(secret["namespace"], secret["name"])}')

                spec['secrets'] = secrets
                secrets.append({
                    'namespace': BATCH_PODS_NAMESPACE,
                    'name': userdata['gsa_key_secret_name'],
                    'mount_path': '/gsa-key',
                    'mount_in_copy': True
                })

                sa = spec.get('service_account')
                check_service_account_permissions(user, sa)

                env = spec.get('env')
                if not env:
                    env = []
                    spec['env'] = env

                if len(parent_ids) == 0:
                    state = 'Ready'
                    n_ready_jobs += 1
                    ready_cores_mcpu += cores_mcpu
                    if not always_run:
                        n_ready_cancellable_jobs += 1
                        ready_cancellable_cores_mcpu += cores_mcpu
                else:
                    state = 'Pending'

                spec_writer.add(json.dumps(spec))
                db_spec = batch_format_version.db_spec(spec)

                jobs_args.append(
                    (batch_id, job_id, state, json.dumps(db_spec),
                     always_run, cores_mcpu, len(parent_ids)))

                for parent_id in parent_ids:
                    job_parents_args.append(
                        (batch_id, job_id, parent_id))

                if attributes:
                    for k, v in attributes.items():
                        job_attributes_args.append(
                            (batch_id, job_id, k, v))

        if batch_format_version.has_full_spec_in_gcs():
            async with timer.step('write spec to gcs'):
                await spec_writer.write()

        rand_token = random.randint(0, app['n_tokens'] - 1)
        n_jobs = len(job_specs)

        async with timer.step('insert jobs'):
            @transaction(db)
            async def insert(tx):
                try:
                    await tx.execute_many('''
INSERT INTO jobs (batch_id, job_id, state, spec, always_run, cores_mcpu, n_pending_parents)
VALUES (%s, %s, %s, %s, %s, %s, %s);
''',
                                          jobs_args)
                except pymysql.err.IntegrityError as err:
                    # 1062 ER_DUP_ENTRY https://dev.mysql.com/doc/refman/5.7/en/server-error-reference.html#error_er_dup_entry
                    if err.args[0] == 1062:
                        log.info(f'bunch containing job {(batch_id, jobs_args[0][1])} already inserted ({err})')
                        return
                    raise
                try:
                    await tx.execute_many('''
INSERT INTO `job_parents` (batch_id, job_id, parent_id)
VALUES (%s, %s, %s);
''',
                                          job_parents_args)
                except pymysql.err.IntegrityError as err:
                    # 1062 ER_DUP_ENTRY https://dev.mysql.com/doc/refman/5.7/en/server-error-reference.html#error_er_dup_entry
                    if err.args[0] == 1062:
                        raise web.HTTPBadRequest(
                            text=f'bunch contains job with duplicated parents ({err})')
                    raise
                await tx.execute_many('''
INSERT INTO `job_attributes` (batch_id, job_id, `key`, `value`)
VALUES (%s, %s, %s, %s);
''',
                                      job_attributes_args)
                await tx.execute_update('''
INSERT INTO batches_staging (batch_id, token, n_jobs, n_ready_jobs, ready_cores_mcpu)
VALUES (%s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  n_jobs = n_jobs + %s,
  n_ready_jobs = n_ready_jobs + %s,
  ready_cores_mcpu = ready_cores_mcpu + %s;
''',
                                        (batch_id, rand_token,
                                         n_jobs, n_ready_jobs, ready_cores_mcpu,
                                         n_jobs, n_ready_jobs, ready_cores_mcpu))
                await tx.execute_update('''
INSERT INTO batch_cancellable_resources (batch_id, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
VALUES (%s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  n_ready_cancellable_jobs = n_ready_cancellable_jobs + %s,
  ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + %s;
''',
                                        (batch_id, rand_token,
                                         n_ready_cancellable_jobs, ready_cancellable_cores_mcpu,
                                         n_ready_cancellable_jobs, ready_cancellable_cores_mcpu))

                if batch_format_version.has_full_spec_in_gcs():
                    await tx.execute_update('''
INSERT INTO batch_bunches (batch_id, token, start_job_id)
VALUES (%s, %s, %s);
''',
                                            (batch_id, spec_writer.token, start_job_id))

            try:
                await insert()  # pylint: disable=no-value-for-parameter
            except aiohttp.web.HTTPException:
                raise
            except Exception as err:
                raise ValueError(f'encountered exception while inserting a bunch'
                                 f'jobs_args={json.dumps(jobs_args)}'
                                 f'job_parents_args={json.dumps(job_parents_args)}') from err
    return web.Response()


@routes.post('/api/v1alpha/batches/create')
@prom_async_time(REQUEST_TIME_POST_CREATE_BATCH)
@rest_authenticated_users_only
async def create_batch(request, userdata):
    app = request.app
    db = app['db']

    batch_spec = await request.json()

    try:
        validate_batch(batch_spec)
    except ValidationError as e:
        raise web.HTTPBadRequest(reason=e.reason)

    user = userdata['username']

    # restrict to what's necessary; in particular, drop the session
    # which is sensitive
    userdata = {
        'username': user,
        'gsa_key_secret_name': userdata['gsa_key_secret_name'],
        'tokens_secret_name': userdata['tokens_secret_name']
    }

    billing_project = batch_spec['billing_project']
    token = batch_spec['token']

    attributes = batch_spec.get('attributes')

    @transaction(db)
    async def insert(tx):
        rows = tx.execute_and_fetchall(
            '''
SELECT * FROM billing_project_users
WHERE billing_project = %s AND user = %s
LOCK IN SHARE MODE;
''',
            (billing_project, user))
        rows = [row async for row in rows]
        if len(rows) != 1:
            assert len(rows) == 0
            raise web.HTTPForbidden(reason=f'unknown billing project {billing_project}')

        maybe_batch = await tx.execute_and_fetchone(
            '''
SELECT * FROM batches
WHERE token = %s AND user = %s FOR UPDATE;
''',
            (token, user))

        if maybe_batch is not None:
            return maybe_batch['id']

        now = time_msecs()
        id = await tx.execute_insertone(
            '''
INSERT INTO batches (userdata, user, billing_project, attributes, callback, n_jobs, time_created, token, state, format_version)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
''',
            (json.dumps(userdata), user, billing_project, json.dumps(attributes),
             batch_spec.get('callback'), batch_spec['n_jobs'],
             now, token, 'open', BATCH_FORMAT_VERSION))

        if attributes:
            await tx.execute_many(
                '''
INSERT INTO `batch_attributes` (batch_id, `key`, `value`)
VALUES (%s, %s, %s)
''',
                [(id, k, v) for k, v in attributes.items()])
        return id
    id = await insert()  # pylint: disable=no-value-for-parameter
    return web.json_response({'id': id})


async def _get_batch(app, batch_id, user):
    db = app['db']

    record = await db.select_and_fetchone('''
SELECT batches.*, SUM(`usage` * rate) AS cost FROM batches
LEFT JOIN aggregated_batch_resources
       ON batches.id = aggregated_batch_resources.batch_id
LEFT JOIN resources
       ON aggregated_batch_resources.resource = resources.resource
WHERE user = %s AND id = %s AND NOT deleted
GROUP BY batches.id;
''', (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    return batch_record_to_dict(record)


async def _cancel_batch(app, batch_id, user):
    db = app['db']

    record = await db.select_and_fetchone(
        '''
SELECT `state` FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()
    if record['state'] == 'open':
        raise web.HTTPBadRequest(reason=f'cannot cancel open batch {batch_id}')

    await db.just_execute(
        'CALL cancel_batch(%s);', (batch_id,))

    app['cancel_batch_state_changed'].set()

    return web.Response()


async def _delete_batch(app, batch_id, user):
    db = app['db']

    record = await db.select_and_fetchone(
        '''
SELECT `state` FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    await db.just_execute(
        'CALL cancel_batch(%s);', (batch_id,))
    await db.execute_update(
        'UPDATE batches SET deleted = 1 WHERE id = %s;', (batch_id,))

    if record['state'] == 'running':
        app['delete_batch_state_changed'].set()


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

    record = await db.select_and_fetchone(
        '''
SELECT 1 FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''',
        (user, batch_id))
    if not record:
        raise web.HTTPNotFound()

    try:
        now = time_msecs()
        await check_call_procedure(
            db, 'CALL close_batch(%s, %s);', (batch_id, now))
    except CallError as e:
        # 2: wrong number of jobs
        if e.rv['rc'] == 2:
            expected_n_jobs = e.rv['expected_n_jobs']
            actual_n_jobs = e.rv['actual_n_jobs']
            raise web.HTTPBadRequest(
                reason=f'wrong number of jobs: expected {expected_n_jobs}, actual {actual_n_jobs}')
        raise

    async with ssl_client_session(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
        await request_retry_transient_errors(
            session, 'PATCH',
            deploy_config.url('batch-driver', f'/api/v1alpha/batches/{user}/{batch_id}/close'),
            headers=app['driver_headers'])

    return web.Response()


@routes.delete('/api/v1alpha/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_DELETE_BATCH)
@rest_authenticated_users_only
async def delete_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    await _delete_batch(request.app, batch_id, user)
    return web.Response()


@routes.get('/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_GET_BATCH_UI)
@web_authenticated_users_only()
async def ui_batch(request, userdata):
    app = request.app
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']

    batch = await _get_batch(app, batch_id, user)

    jobs, last_job_id = await _query_batch_jobs(request, batch_id)
    for j in jobs:
        j['duration'] = humanize_timedelta_msecs(j['duration'])
    batch['jobs'] = jobs

    page_context = {
        'batch': batch,
        'q': request.query.get('q'),
        'last_job_id': last_job_id
    }
    return await render_template('batch', request, userdata, 'batch.html', page_context)


@routes.post('/batches/{batch_id}/cancel')
@prom_async_time(REQUEST_TIME_POST_CANCEL_BATCH_UI)
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def ui_cancel_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    await _cancel_batch(request.app, batch_id, user)
    session = await aiohttp_session.get_session(request)
    set_message(session, f'Batch {batch_id} cancelled.', 'info')
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


@routes.post('/batches/{batch_id}/delete')
@prom_async_time(REQUEST_TIME_POST_DELETE_BATCH_UI)
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def ui_delete_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    await _delete_batch(request.app, batch_id, user)
    session = await aiohttp_session.get_session(request)
    set_message(session, f'Batch {batch_id} deleted.', 'info')
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


@routes.get('/batches', name='batches')
@prom_async_time(REQUEST_TIME_GET_BATCHES_UI)
@web_authenticated_users_only()
async def ui_batches(request, userdata):
    user = userdata['username']
    batches, last_batch_id = await _query_batches(request, user)
    page_context = {
        'batches': batches,
        'q': request.query.get('q'),
        'last_batch_id': last_batch_id
    }
    return await render_template('batch', request, userdata, 'batches.html', page_context)


async def _get_job(app, batch_id, job_id, user):
    db = app['db']

    record = await db.select_and_fetchone('''
SELECT jobs.*, ip_address, format_version, SUM(`usage` * rate) AS cost
FROM jobs
INNER JOIN batches
  ON jobs.batch_id = batches.id
LEFT JOIN attempts
  ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id AND jobs.attempt_id = attempts.attempt_id
LEFT JOIN instances
  ON attempts.instance_name = instances.name
LEFT JOIN aggregated_job_resources
  ON jobs.batch_id = aggregated_job_resources.batch_id AND
     jobs.job_id = aggregated_job_resources.job_id
LEFT JOIN resources
  ON aggregated_job_resources.resource = resources.resource
WHERE user = %s AND jobs.batch_id = %s AND NOT deleted AND jobs.job_id = %s
GROUP BY jobs.batch_id, jobs.job_id;
''',
                                          (user, batch_id, job_id))
    if not record:
        raise web.HTTPNotFound()

    full_status, full_spec, attributes = await asyncio.gather(
        _get_full_job_status(app, record),
        _get_full_job_spec(app, record),
        _get_attributes(app, record)
    )

    job = job_record_to_dict(record, attributes.get('name'))
    job['status'] = full_status
    job['spec'] = full_spec
    if attributes:
        job['attributes'] = attributes
    return job


async def _get_attempts(app, batch_id, job_id, user):
    db = app['db']

    attempts = db.select_and_fetchall('''
SELECT attempts.*
FROM jobs
INNER JOIN batches ON jobs.batch_id = batches.id
LEFT JOIN attempts ON jobs.batch_id = attempts.batch_id and jobs.job_id = attempts.job_id
WHERE user = %s AND jobs.batch_id = %s AND NOT deleted AND jobs.job_id = %s;
''',
                                      (user, batch_id, job_id))

    attempts = [attempt async for attempt in attempts]
    if len(attempts) == 0:
        raise web.HTTPNotFound()
    if len(attempts) == 1 and attempts[0]['attempt_id'] is None:
        return None
    for attempt in attempts:
        start_time = attempt['start_time']
        if start_time is not None:
            attempt['start_time'] = time_msecs_str(start_time)
        else:
            del attempt['start_time']

        end_time = attempt['end_time']
        if end_time is not None:
            attempt['end_time'] = time_msecs_str(end_time)
        else:
            del attempt['end_time']

        if start_time is not None:
            # elapsed time if attempt is still running
            if end_time is None:
                end_time = time_msecs()
            duration_msecs = max(end_time - start_time, 0)
            attempt['duration'] = humanize_timedelta_msecs(duration_msecs)

    return attempts


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/attempts')
@prom_async_time(REQUEST_TIME_GET_ATTEMPTS)
@rest_authenticated_users_only
async def get_attempts(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']

    attempts = await _get_attempts(request.app, batch_id, job_id, user)
    return web.json_response(attempts)


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
    app = request.app
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']

    job_status, attempts, job_log = await asyncio.gather(_get_job(app, batch_id, job_id, user),
                                                         _get_attempts(app, batch_id, job_id, user),
                                                         _get_job_log(app, batch_id, job_id, user))

    page_context = {
        'batch_id': batch_id,
        'job_id': job_id,
        'job_log': job_log,
        'attempts': attempts,
        'job_status': json.dumps(job_status, indent=2)
    }
    return await render_template('batch', request, userdata, 'job.html', page_context)


async def _query_billing(request):
    db = request.app['db']

    date_format = '%m/%d/%Y'

    default_start = datetime.datetime.now().replace(day=1)
    default_start = datetime.datetime.strftime(default_start, date_format)

    default_end = datetime.datetime.now()
    default_end = datetime.datetime.strftime(default_end, date_format)

    async def parse_error(msg):
        session = await aiohttp_session.get_session(request)
        set_message(session, msg, 'error')
        return ([], default_start, default_end)

    start_query = request.query.get('start', default_start)
    try:
        start = datetime.datetime.strptime(start_query, date_format)
        start = start.timestamp() * 1000
    except ValueError:
        return await parse_error(f"Invalid value for start '{start_query}'; must be in the format of MM/DD/YYYY.")

    end_query = request.query.get('end', default_end)
    try:
        end = datetime.datetime.strptime(end_query, date_format)
        end = (end + datetime.timedelta(days=1)).timestamp() * 1000
    except ValueError:
        return await parse_error(f"Invalid value for end '{end_query}'; must be in the format of MM/DD/YYYY.")

    if start > end:
        return await parse_error(f'Invalid search; start must be earlier than end.')

    sql = f'''
SELECT
  billing_project,
  `user`,
  CAST(SUM(IF(format_version < 3, msec_mcpu, 0)) AS SIGNED) as msec_mcpu,
  SUM(IF(format_version >= 3, `usage` * rate, NULL)) as cost
FROM batches
LEFT JOIN aggregated_batch_resources
  ON aggregated_batch_resources.batch_id = batches.id
LEFT JOIN resources
  ON resources.resource = aggregated_batch_resources.resource
WHERE `time_completed` >= %s AND `time_completed` <= %s
GROUP BY billing_project, `user`;
'''

    sql_args = (start, end)

    def billing_record_to_dict(record):
        cost_msec_mcpu = cost_from_msec_mcpu(record['msec_mcpu'])
        cost_resources = record['cost']
        record['cost'] = coalesce(cost_msec_mcpu, 0) + coalesce(cost_resources, 0)
        del record['msec_mcpu']
        return record

    billing = [billing_record_to_dict(record)
               async for record
               in db.select_and_fetchall(sql, sql_args)]

    return (billing, start_query, end_query)


@routes.get('/billing')
@prom_async_time(REQUEST_TIME_GET_BILLING_UI)
@web_authenticated_developers_only()
async def ui_get_billing(request, userdata):
    billing, start, end = await _query_billing(request)

    billing_by_user = {}
    billing_by_project = {}
    for record in billing:
        billing_project = record['billing_project']
        user = record['user']
        cost = record['cost']
        billing_by_user[user] = billing_by_user.get(user, 0) + cost
        billing_by_project[billing_project] = billing_by_project.get(billing_project, 0) + cost

    billing_by_project = [{'billing_project': billing_project, 'cost': f'${cost:.4f}'} for billing_project, cost in billing_by_project.items()]
    billing_by_project.sort(key=lambda record: record['billing_project'])

    billing_by_user = [{'user': user, 'cost': f'${cost:.4f}'} for user, cost in billing_by_user.items()]
    billing_by_user.sort(key=lambda record: record['user'])

    billing_by_project_user = [{'billing_project': record['billing_project'], 'user': record['user'], 'cost': f'${record["cost"]:.4f}'}
                               for record in billing]
    billing_by_project_user.sort(key=lambda record: (record['billing_project'], record['user']))

    page_context = {
        'billing_by_project': billing_by_project,
        'billing_by_user': billing_by_user,
        'billing_by_project_user': billing_by_project_user,
        'start': start,
        'end': end
    }
    return await render_template('batch', request, userdata, 'billing.html', page_context)


@routes.get('/billing_projects')
@prom_async_time(REQUEST_TIME_GET_BILLING_PROJECTS_UI)
@web_authenticated_developers_only()
async def ui_get_billing_projects(request, userdata):
    db = request.app['db']

    @transaction(db, read_only=True)
    async def select(tx):
        billing_projects = {}
        async for record in tx.execute_and_fetchall(
                'SELECT * FROM billing_projects LOCK IN SHARE MODE;'):
            name = record['name']
            billing_projects[name] = []
        async for record in tx.execute_and_fetchall(
                'SELECT * FROM billing_project_users LOCK IN SHARE MODE;'):
            billing_project = record['billing_project']
            user = record['user']
            billing_projects[billing_project].append(user)
        return billing_projects
    billing_projects = await select()  # pylint: disable=no-value-for-parameter
    page_context = {
        'billing_projects': billing_projects
    }
    return await render_template('batch', request, userdata, 'billing_projects.html', page_context)


@routes.post('/billing_projects/{billing_project}/users/{user}/remove')
@prom_async_time(REQUEST_TIME_POST_BILLING_PROJECT_REMOVE_USER_UI)
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
async def post_billing_projects_remove_user(request, userdata):  # pylint: disable=unused-argument
    db = request.app['db']
    billing_project = request.match_info['billing_project']
    user = request.match_info['user']

    session = await aiohttp_session.get_session(request)

    @transaction(db)
    async def delete(tx):
        row = await tx.execute_and_fetchone(
            '''
SELECT billing_projects.name as billing_project, user
FROM billing_projects
LEFT JOIN (SELECT * FROM billing_project_users
    WHERE billing_project = %s AND user = %s FOR UPDATE) AS t
  ON billing_projects.name = t.billing_project
WHERE billing_projects.name = %s;
''',
            (billing_project, user, billing_project))
        if not row:
            set_message(session, f'No such billing project {billing_project}.', 'error')
            raise web.HTTPFound(deploy_config.external_url('batch', f'/billing_projects'))
        assert row['billing_project'] == billing_project

        if row['user'] is None:
            set_message(session, f'User {user} is not member of billing project {billing_project}.', 'info')
            raise web.HTTPFound(deploy_config.external_url('batch', f'/billing_projects'))

        await tx.just_execute(
            '''
DELETE FROM billing_project_users
WHERE billing_project = %s AND user = %s;
''',
            (billing_project, user))
    await delete()  # pylint: disable=no-value-for-parameter
    set_message(session, f'Removed user {user} from billing project {billing_project}.', 'info')
    return web.HTTPFound(deploy_config.external_url('batch', f'/billing_projects'))


@routes.post('/billing_projects/{billing_project}/users/add')
@prom_async_time(REQUEST_TIME_POST_BILLING_PROJECT_ADD_USER_UI)
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
async def post_billing_projects_add_user(request, userdata):  # pylint: disable=unused-argument
    db = request.app['db']
    post = await request.post()
    user = post['user']
    billing_project = request.match_info['billing_project']

    session = await aiohttp_session.get_session(request)

    @transaction(db)
    async def insert(tx):
        row = await tx.execute_and_fetchone(
            '''
SELECT billing_projects.name as billing_project, user
FROM billing_projects
LEFT JOIN (SELECT * FROM billing_project_users
    WHERE billing_project = %s AND user = %s FOR UPDATE) AS t
  ON billing_projects.name = t.billing_project
WHERE billing_projects.name = %s;
''',
            (billing_project, user, billing_project))
        if row is None:
            set_message(session, f'No such billing project {billing_project}.', 'error')
            raise web.HTTPFound(deploy_config.external_url('batch', f'/billing_projects'))

        if row['user'] is not None:
            set_message(session, f'User {user} is already member of billing project {billing_project}.', 'info')
            raise web.HTTPFound(deploy_config.external_url('batch', f'/billing_projects'))

        await tx.execute_insertone(
            '''
INSERT INTO billing_project_users(billing_project, user)
VALUES (%s, %s);
''',
            (billing_project, user))
    await insert()  # pylint: disable=no-value-for-parameter
    set_message(session, f'Added user {user} to billing project {billing_project}.', 'info')
    return web.HTTPFound(deploy_config.external_url('batch', f'/billing_projects'))


@routes.post('/billing_projects/create')
@prom_async_time(REQUEST_TIME_POST_CREATE_BILLING_PROJECT_UI)
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
async def post_create_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db = request.app['db']
    post = await request.post()
    billing_project = post['billing_project']

    session = await aiohttp_session.get_session(request)

    @transaction(db)
    async def insert(tx):
        row = await tx.execute_and_fetchone(
            '''
SELECT 1 FROM billing_projects
WHERE name = %s
FOR UPDATE;
''',
            (billing_project))
        if row is not None:
            set_message(session, f'Billing project {billing_project} already exists.', 'error')
            raise web.HTTPFound(deploy_config.external_url('batch', f'/billing_projects'))

        await tx.execute_insertone(
            '''
INSERT INTO billing_projects(name)
VALUES (%s);
''',
            (billing_project,))
    await insert()  # pylint: disable=no-value-for-parameter
    set_message(session, f'Added billing project {billing_project}.', 'info')
    return web.HTTPFound(deploy_config.external_url('batch', f'/billing_projects'))


@routes.get('')
@routes.get('/')
@web_authenticated_users_only()
async def index(request, userdata):
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


async def cancel_batch_loop_body(app):
    async with ssl_client_session(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
        await request_retry_transient_errors(
            session, 'POST',
            deploy_config.url('batch-driver', f'/api/v1alpha/batches/cancel'),
            headers=app['driver_headers'])

    should_wait = True
    return should_wait


async def delete_batch_loop_body(app):
    async with ssl_client_session(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
        await request_retry_transient_errors(
            session, 'POST',
            deploy_config.url('batch-driver', f'/api/v1alpha/batches/delete'),
            headers=app['driver_headers'])

    should_wait = True
    return should_wait


async def on_startup(app):
    pool = concurrent.futures.ThreadPoolExecutor()
    app['blocking_pool'] = pool

    db = Database()
    await db.async_init()
    app['db'] = db

    row = await db.select_and_fetchone(
        '''
SELECT worker_type, worker_cores, worker_disk_size_gb,
  instance_id, internal_token, n_tokens FROM globals;
''')

    app['worker_type'] = row['worker_type']
    app['worker_cores'] = row['worker_cores']
    app['worker_disk_size_gb'] = row['worker_disk_size_gb']
    app['n_tokens'] = row['n_tokens']

    instance_id = row['instance_id']
    log.info(f'instance_id {instance_id}')
    app['instance_id'] = instance_id

    app['driver_headers'] = {
        'Authorization': f'Bearer {row["internal_token"]}'
    }

    credentials = google.oauth2.service_account.Credentials.from_service_account_file(
        '/gsa-key/key.json')
    app['log_store'] = LogStore(BATCH_BUCKET_NAME, WORKER_LOGS_BUCKET_NAME, instance_id, pool, credentials=credentials)

    cancel_batch_state_changed = asyncio.Event()
    app['cancel_batch_state_changed'] = cancel_batch_state_changed

    asyncio.ensure_future(retry_long_running(
        'cancel_batch_loop',
        run_if_changed, cancel_batch_state_changed, cancel_batch_loop_body, app))

    delete_batch_state_changed = asyncio.Event()
    app['delete_batch_state_changed'] = delete_batch_state_changed

    asyncio.ensure_future(retry_long_running(
        'delete_batch_loop',
        run_if_changed, delete_batch_state_changed, delete_batch_loop_body, app))


async def on_cleanup(app):
    blocking_pool = app['blocking_pool']
    blocking_pool.shutdown()


def run():
    app = web.Application(client_max_size=HTTP_CLIENT_MAX_SIZE)
    setup_aiohttp_session(app)

    setup_aiohttp_jinja2(app, 'batch.front_end')
    setup_common_static_routes(routes)
    app.add_routes(routes)
    app.router.add_get("/metrics", server_stats)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    web.run_app(deploy_config.prefix_application(app,
                                                 'batch',
                                                 client_max_size=HTTP_CLIENT_MAX_SIZE),
                host='0.0.0.0',
                port=5000,
                access_log_class=AccessLogger,
                ssl_context=get_server_ssl_context())
