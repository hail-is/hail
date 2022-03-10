import asyncio
import collections
import datetime
import json
import logging
import os
import random
import signal
import traceback
from functools import wraps
from numbers import Number
from typing import Dict, Optional, Union

import aiohttp
import aiohttp_session
import humanize
import pandas as pd
import plotly
import plotly.express as px
import pymysql
from aiohttp import web
from prometheus_async.aio.web import server_stats  # type: ignore

from gear import (
    Database,
    check_csrf_token,
    monitor_endpoints_middleware,
    rest_authenticated_users_only,
    setup_aiohttp_session,
    transaction,
    web_authenticated_developers_only,
    web_authenticated_users_only,
)
from gear.clients import get_cloud_async_fs
from gear.database import CallError
from hailtop import aiotools, dictfix, httpx
from hailtop.batch_client.parse import parse_cpu_in_mcpu, parse_memory_in_bytes, parse_storage_in_bytes
from hailtop.config import get_deploy_config
from hailtop.hail_logging import AccessLogger
from hailtop.tls import internal_server_ssl_context
from hailtop.utils import (
    LoggingTimer,
    cost_str,
    dump_all_stacktraces,
    humanize_timedelta_msecs,
    periodically_call,
    request_retry_transient_errors,
    retry_long_running,
    run_if_changed,
    time_msecs,
    time_msecs_str,
)
from web_common import render_template, set_message, setup_aiohttp_jinja2, setup_common_static_routes

from ..batch import batch_record_to_dict, cancel_batch_in_db, job_record_to_dict
from ..batch_configuration import BATCH_STORAGE_URI, CLOUD, DEFAULT_NAMESPACE, SCOPE
from ..batch_format_version import BatchFormatVersion
from ..cloud.resource_utils import (
    cores_mcpu_to_memory_bytes,
    cost_from_msec_mcpu,
    is_valid_cores_mcpu,
    memory_to_worker_type,
    valid_machine_types,
)
from ..exceptions import (
    BatchOperationAlreadyCompletedError,
    BatchUserError,
    ClosedBillingProjectError,
    InvalidBillingLimitError,
    NonExistentBillingProjectError,
)
from ..file_store import FileStore
from ..globals import BATCH_FORMAT_VERSION, HTTP_CLIENT_MAX_SIZE
from ..inst_coll_config import InstanceCollectionConfigs
from ..spec_writer import SpecWriter
from ..utils import accrued_cost_from_cost_and_msec_mcpu, coalesce, query_billing_projects
from .validate import ValidationError, validate_and_clean_jobs, validate_batch

# import uvloop


# uvloop.install()

log = logging.getLogger('batch.front_end')

routes = web.RouteTableDef()

deploy_config = get_deploy_config()

BATCH_JOB_DEFAULT_CPU = os.environ.get('HAIL_BATCH_JOB_DEFAULT_CPU', '1')
BATCH_JOB_DEFAULT_MEMORY = os.environ.get('HAIL_BATCH_JOB_DEFAULT_MEMORY', 'standard')
BATCH_JOB_DEFAULT_STORAGE = os.environ.get('HAIL_BATCH_JOB_DEFAULT_STORAGE', '0Gi')
BATCH_JOB_DEFAULT_PREEMPTIBLE = True


def rest_authenticated_developers_or_auth_only(fun):
    @rest_authenticated_users_only
    @wraps(fun)
    async def wrapped(request, userdata, *args, **kwargs):
        if userdata['is_developer'] == 1 or userdata['username'] == 'auth':
            return await fun(request, userdata, *args, **kwargs)
        raise web.HTTPUnauthorized()

    return wrapped


def catch_ui_error_in_dev(fun):
    @wraps(fun)
    async def wrapped(request, userdata, *args, **kwargs):
        try:
            return await fun(request, userdata, *args, **kwargs)
        except asyncio.CancelledError:
            raise
        except aiohttp.web_exceptions.HTTPFound as e:
            raise e
        except Exception as e:
            if SCOPE == 'dev':
                log.exception('error while populating ui page')
                raise web.HTTPInternalServerError(text=traceback.format_exc()) from e
            raise

    return wrapped


async def _user_can_access(db: Database, batch_id: int, user: str):
    record = await db.select_and_fetchone(
        '''
SELECT id
FROM batches
LEFT JOIN billing_project_users ON batches.billing_project = billing_project_users.billing_project
WHERE id = %s AND billing_project_users.`user` = %s;
''',
        (batch_id, user),
    )

    return record is not None


def rest_billing_project_users_only(fun):
    @rest_authenticated_users_only
    @wraps(fun)
    async def wrapped(request, userdata, *args, **kwargs):
        db = request.app['db']
        batch_id = int(request.match_info['batch_id'])
        user = userdata['username']
        permitted_user = await _user_can_access(db, batch_id, user)
        if not permitted_user:
            raise web.HTTPNotFound()
        return await fun(request, userdata, batch_id, *args, **kwargs)

    return wrapped


def web_billing_project_users_only(redirect=True):
    def wrap(fun):
        @web_authenticated_users_only(redirect)
        @wraps(fun)
        async def wrapped(request, userdata, *args, **kwargs):
            db = request.app['db']
            batch_id = int(request.match_info['batch_id'])
            user = userdata['username']
            permitted_user = await _user_can_access(db, batch_id, user)
            if not permitted_user:
                raise web.HTTPNotFound()
            return await fun(request, userdata, batch_id, *args, **kwargs)

        return wrapped

    return wrap


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


async def _handle_ui_error(session, f, *args, **kwargs):
    try:
        await f(*args, **kwargs)
    except KeyError as e:
        set_message(session, str(e), 'error')
        log.info(f'ui error: KeyError {e}')
        return True
    except BatchOperationAlreadyCompletedError as e:
        set_message(session, e.message, e.ui_error_type)
        log.info(f'ui error: BatchOperationAlreadyCompletedError {e.message}')
        return True
    except BatchUserError as e:
        set_message(session, e.message, e.ui_error_type)
        log.info(f'ui error: BatchUserError {e.message}')
        return True
    else:
        return False


async def _handle_api_error(f, *args, **kwargs):
    try:
        await f(*args, **kwargs)
    except BatchOperationAlreadyCompletedError as e:
        log.info(e.message)
        return
    except BatchUserError as e:
        raise e.http_response()


async def _query_batch_jobs(request, batch_id):
    state_query_values = {
        'pending': ['Pending'],
        'ready': ['Ready'],
        'creating': ['Creating'],
        'running': ['Running'],
        'live': ['Ready', 'Creating', 'Running'],
        'cancelled': ['Cancelled'],
        'error': ['Error'],
        'failed': ['Failed'],
        'bad': ['Error', 'Failed'],
        'success': ['Success'],
        'done': ['Cancelled', 'Error', 'Failed', 'Success'],
    }

    db = request.app['db']

    # batch has already been validated
    where_conditions = ['(jobs.batch_id = %s)']
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
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` = %s AND `value` = %s))
'''
            args = [k, v]
        elif t.startswith('has:'):
            k = t[4:]
            condition = '''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` = %s))
'''
            args = [k]
        elif t in state_query_values:
            values = state_query_values[t]
            condition = ' OR '.join(['(jobs.state = %s)' for v in values])
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
SELECT jobs.*, batches.user, batches.billing_project,  batches.format_version,
  job_attributes.value AS name, COALESCE(SUM(`usage` * rate), 0) AS cost
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

    jobs = [job_record_to_dict(record, record['name']) async for record in db.select_and_fetchall(sql, sql_args)]

    if len(jobs) == 50:
        last_job_id = jobs[-1]['job_id']
    else:
        last_job_id = None

    return (jobs, last_job_id)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs')
@rest_billing_project_users_only
async def get_jobs(request, userdata, batch_id):  # pylint: disable=unused-argument
    db = request.app['db']
    record = await db.select_and_fetchone(
        '''
SELECT * FROM batches
WHERE id = %s AND NOT deleted;
''',
        (batch_id,),
    )
    if not record:
        raise web.HTTPNotFound()

    jobs, last_job_id = await _query_batch_jobs(request, batch_id)
    resp = {'jobs': jobs}
    if last_job_id is not None:
        resp['last_job_id'] = last_job_id
    return web.json_response(resp)


async def _get_job_log_from_record(app, batch_id, job_id, record):
    client_session: httpx.ClientSession = app['client_session']
    batch_format_version = BatchFormatVersion(record['format_version'])

    state = record['state']
    ip_address = record['ip_address']

    spec = json.loads(record['spec'])
    tasks = []

    has_input_files = batch_format_version.get_spec_has_input_files(spec)
    if has_input_files:
        tasks.append('input')

    tasks.append('main')

    has_output_files = batch_format_version.get_spec_has_output_files(spec)
    if has_output_files:
        tasks.append('output')

    if state == 'Running':
        try:
            resp = await request_retry_transient_errors(
                client_session, 'GET', f'http://{ip_address}:5000/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log'
            )
            return await resp.json()
        except aiohttp.ClientResponseError:
            log.exception(f'while getting log for {(batch_id, job_id)}')
            return {task: 'ERROR: encountered a problem while fetching the log' for task in tasks}

    if state in ('Pending', 'Ready', 'Creating'):
        return None

    if state == 'Cancelled' and record['last_cancelled_attempt_id'] is None:
        return None

    assert state in ('Error', 'Failed', 'Success', 'Cancelled')

    attempt_id = record['attempt_id'] or record['last_cancelled_attempt_id']
    assert attempt_id is not None

    file_store: FileStore = app['file_store']
    batch_format_version = BatchFormatVersion(record['format_version'])

    async def _read_log_from_cloud_storage(task):
        try:
            data = await file_store.read_log_file(batch_format_version, batch_id, job_id, attempt_id, task)
        except FileNotFoundError:
            id = (batch_id, job_id)
            log.exception(f'missing log file for {id} and task {task}')
            data = 'ERROR: could not find log file'
        return task, data

    return dict(await asyncio.gather(*[_read_log_from_cloud_storage(task) for task in tasks]))


async def _get_job_log(app, batch_id, job_id):
    db: Database = app['db']

    record = await db.select_and_fetchone(
        '''
SELECT jobs.state, jobs.spec, ip_address, format_version, jobs.attempt_id, t.attempt_id AS last_cancelled_attempt_id
FROM jobs
INNER JOIN batches
  ON jobs.batch_id = batches.id
LEFT JOIN attempts
  ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id AND jobs.attempt_id = attempts.attempt_id
LEFT JOIN instances
  ON attempts.instance_name = instances.name
LEFT JOIN (
  SELECT batch_id, job_id, attempt_id
  FROM attempts
  WHERE reason = "cancelled" AND batch_id = %s AND job_id = %s
  ORDER BY end_time DESC
  LIMIT 1
) AS t
  ON jobs.batch_id = t.batch_id AND jobs.job_id = t.job_id
WHERE jobs.batch_id = %s AND NOT deleted AND jobs.job_id = %s;
''',
        (batch_id, job_id, batch_id, job_id),
    )
    if not record:
        raise web.HTTPNotFound()
    return await _get_job_log_from_record(app, batch_id, job_id, record)


async def _get_attributes(app, record):
    db: Database = app['db']

    batch_id = record['batch_id']
    job_id = record['job_id']
    format_version = BatchFormatVersion(record['format_version'])

    if not format_version.has_full_spec_in_cloud():
        spec = json.loads(record['spec'])
        return spec.get('attributes')

    records = db.select_and_fetchall(
        '''
SELECT `key`, `value`
FROM job_attributes
WHERE batch_id = %s AND job_id = %s;
''',
        (batch_id, job_id),
        query_name='get_attributes',
    )
    return {record['key']: record['value'] async for record in records}


async def _get_full_job_spec(app, record):
    db: Database = app['db']
    file_store: FileStore = app['file_store']

    batch_id = record['batch_id']
    job_id = record['job_id']
    format_version = BatchFormatVersion(record['format_version'])

    if not format_version.has_full_spec_in_cloud():
        return json.loads(record['spec'])

    token, start_job_id = await SpecWriter.get_token_start_id(db, batch_id, job_id)

    try:
        spec = await file_store.read_spec_file(batch_id, token, start_job_id, job_id)
        return json.loads(spec)
    except FileNotFoundError:
        id = (batch_id, job_id)
        log.exception(f'missing spec file for {id}')
        return None


async def _get_full_job_status(app, record):
    client_session: httpx.ClientSession = app['client_session']
    file_store: FileStore = app['file_store']

    batch_id = record['batch_id']
    job_id = record['job_id']
    state = record['state']
    format_version = BatchFormatVersion(record['format_version'])

    if state in ('Pending', 'Creating', 'Ready'):
        return None

    if state == 'Cancelled' and record['last_cancelled_attempt_id'] is None:
        return None

    attempt_id = record['attempt_id'] or record['last_cancelled_attempt_id']
    assert attempt_id is not None

    if state in ('Error', 'Failed', 'Success', 'Cancelled'):
        if not format_version.has_full_status_in_gcs():
            return json.loads(record['status'])

        try:
            status = await file_store.read_status_file(batch_id, job_id, attempt_id)
            return json.loads(status)
        except FileNotFoundError:
            id = (batch_id, job_id)
            log.exception(f'missing status file for {id}')
            return None

    assert state == 'Running'
    assert record['status'] is None

    ip_address = record['ip_address']
    try:
        resp = await request_retry_transient_errors(
            client_session, 'GET', f'http://{ip_address}:5000/api/v1alpha/batches/{batch_id}/jobs/{job_id}/status'
        )
        return await resp.json()
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            return None
        raise


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
@rest_billing_project_users_only
async def get_job_log(request, userdata, batch_id):  # pylint: disable=unused-argument
    job_id = int(request.match_info['job_id'])
    job_log = await _get_job_log(request.app, batch_id, job_id)
    return web.json_response(job_log)


async def _query_batches(request, user, q):
    db = request.app['db']

    where_conditions = [
        'EXISTS (SELECT * FROM billing_project_users WHERE billing_project_users.`user` = %s AND billing_project_users.billing_project = batches.billing_project)',
        'NOT deleted',
    ]
    where_args = [user]

    last_batch_id = request.query.get('last_batch_id')
    if last_batch_id is not None:
        last_batch_id = int(last_batch_id)
        where_conditions.append('(batches.id < %s)')
        where_args.append(last_batch_id)

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
((batches.id) IN
 (SELECT batch_id FROM batch_attributes
  WHERE `key` = %s AND `value` = %s))
'''
            args = [k, v]
        elif t.startswith('has:'):
            k = t[4:]
            condition = '''
((batches.id) IN
 (SELECT batch_id FROM batch_attributes
  WHERE `key` = %s))
'''
            args = [k]
        elif t.startswith('user:'):
            k = t[5:]
            condition = '''
(batches.`user` = %s)
'''
            args = [k]
        elif t.startswith('billing_project:'):
            k = t[16:]
            condition = '''
(batches.`billing_project` = %s)
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
            condition = '(batches_cancelled.id IS NOT NULL)'
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
SELECT batches.*, batches_cancelled.id IS NOT NULL AS cancelled, COALESCE(SUM(`usage` * rate), 0) AS cost, batches_n_jobs_in_complete_states.n_completed, batches_n_jobs_in_complete_states.n_succeeded, batches_n_jobs_in_complete_states.n_failed, batches_n_jobs_in_complete_states.n_cancelled
FROM batches
LEFT JOIN batches_n_jobs_in_complete_states
  ON batches.id = batches_n_jobs_in_complete_states.id
LEFT JOIN batches_cancelled
  ON batches.id = batches_cancelled.id
LEFT JOIN aggregated_batch_resources
  ON batches.id = aggregated_batch_resources.batch_id
LEFT JOIN resources
  ON aggregated_batch_resources.resource = resources.resource
WHERE {' AND '.join(where_conditions)}
GROUP BY batches.id
ORDER BY batches.id DESC
LIMIT 51;
'''
    sql_args = where_args

    batches = [
        batch_record_to_dict(batch) async for batch in db.select_and_fetchall(sql, sql_args, query_name='get_batches')
    ]

    if len(batches) == 51:
        batches.pop()
        last_batch_id = batches[-1]['id']
    else:
        last_batch_id = None

    return (batches, last_batch_id)


@routes.get('/api/v1alpha/batches')
@rest_authenticated_users_only
async def get_batches(request, userdata):  # pylint: disable=unused-argument
    user = userdata['username']
    q = request.query.get('q', f'user:{user}')
    batches, last_batch_id = await _query_batches(request, user, q)
    body = {'batches': batches}
    if last_batch_id is not None:
        body['last_batch_id'] = last_batch_id
    return web.json_response(body)


def check_service_account_permissions(user, sa):
    if sa is None:
        return
    if user == 'ci':
        if sa['name'] in ('ci-agent', 'admin') and DEFAULT_NAMESPACE in ('default', sa['namespace']):
            return
    elif user == 'test':
        if sa['namespace'] == DEFAULT_NAMESPACE and sa['name'] == 'test-batch-sa':
            return
    raise web.HTTPBadRequest(reason=f'unauthorized service account {(sa["namespace"], sa["name"])} for user {user}')


@routes.post('/api/v1alpha/batches/{batch_id}/jobs/create')
@rest_authenticated_users_only
async def create_jobs(request: aiohttp.web.Request, userdata: dict):
    app = request.app

    if app['frozen']:
        log.info('ignoring batch create request; batch is frozen')
        raise web.HTTPServiceUnavailable()

    batch_id = int(request.match_info['batch_id'])
    job_specs = await request.json()
    return await _create_jobs(userdata, job_specs, batch_id, app)


async def _create_jobs(userdata: dict, job_specs: dict, batch_id: int, app: aiohttp.web.Application):
    db: Database = app['db']
    file_store: FileStore = app['file_store']
    user = userdata['username']

    # restrict to what's necessary; in particular, drop the session
    # which is sensitive
    userdata = {
        'username': user,
        'hail_credentials_secret_name': userdata['hail_credentials_secret_name'],
        'tokens_secret_name': userdata['tokens_secret_name'],
    }

    async with LoggingTimer(f'batch {batch_id} create jobs') as timer:
        async with timer.step('fetch batch'):
            record = await db.select_and_fetchone(
                '''
SELECT `state`, format_version FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''',
                (user, batch_id),
            )

        if not record:
            raise web.HTTPNotFound()
        if record['state'] != 'open':
            raise web.HTTPBadRequest(reason=f'batch {batch_id} is not open')
        batch_format_version = BatchFormatVersion(record['format_version'])

        async with timer.step('validate job_specs'):
            try:
                validate_and_clean_jobs(job_specs)
            except ValidationError as e:
                raise web.HTTPBadRequest(reason=e.reason)

        async with timer.step('build db args'):
            spec_writer = SpecWriter(file_store, batch_id)

            jobs_args = []
            job_parents_args = []
            job_attributes_args = []

            inst_coll_resources: Dict[str, Dict[str, int]] = collections.defaultdict(
                lambda: {
                    'n_jobs': 0,
                    'n_ready_jobs': 0,
                    'ready_cores_mcpu': 0,
                    'n_ready_cancellable_jobs': 0,
                    'ready_cancellable_cores_mcpu': 0,
                }
            )

            prev_job_idx = None
            start_job_id = None

            for spec in job_specs:
                job_id = spec['job_id']
                parent_ids = spec.pop('parent_ids', [])
                always_run = spec.pop('always_run', False)

                cloud = spec.get('cloud', CLOUD)

                if batch_format_version.has_full_spec_in_cloud():
                    attributes = spec.pop('attributes', None)
                else:
                    attributes = spec.get('attributes')

                id = (batch_id, job_id)

                if start_job_id is None:
                    start_job_id = job_id

                if batch_format_version.has_full_spec_in_cloud() and prev_job_idx:
                    if job_id != prev_job_idx + 1:
                        raise web.HTTPBadRequest(
                            reason=f'noncontiguous job ids found in the spec: {prev_job_idx} -> {job_id}'
                        )
                prev_job_idx = job_id

                resources = spec.get('resources')
                if not resources:
                    resources = {}
                    spec['resources'] = resources

                worker_type = None
                machine_type = resources.get('machine_type')
                preemptible = resources.get('preemptible', BATCH_JOB_DEFAULT_PREEMPTIBLE)

                if machine_type and machine_type not in valid_machine_types(cloud):
                    raise web.HTTPBadRequest(reason=f'unknown machine type {machine_type} for cloud {cloud}')

                if machine_type and ('cpu' in resources or 'memory' in resources):
                    raise web.HTTPBadRequest(reason='cannot specify cpu and memory with machine_type')

                if spec['process']['type'] == 'jvm':
                    if 'cpu' in resources:
                        raise web.HTTPBadRequest(reason='jvm jobs may not specify cpu')
                    if 'memory' in resources:
                        raise web.HTTPBadRequest(reason='jvm jobs may not specify memory')
                    if 'storage' in resources:
                        raise web.HTTPBadRequest(reason='jvm jobs may not specify storage')
                    if machine_type is not None:
                        raise web.HTTPBadRequest(reason='jvm jobs may not specify machine_type')

                req_memory_bytes: Optional[int]
                if machine_type is None:
                    if 'cpu' not in resources:
                        resources['cpu'] = BATCH_JOB_DEFAULT_CPU
                    resources['req_cpu'] = resources['cpu']
                    del resources['cpu']
                    req_cores_mcpu = parse_cpu_in_mcpu(resources['req_cpu'])

                    if req_cores_mcpu is None or not is_valid_cores_mcpu(req_cores_mcpu):
                        raise web.HTTPBadRequest(
                            reason=f'bad resource request for job {id}: '
                            f'cpu must be a power of two with a min of 0.25; '
                            f'found {resources["req_cpu"]}.'
                        )

                    if 'memory' not in resources:
                        resources['memory'] = BATCH_JOB_DEFAULT_MEMORY
                    resources['req_memory'] = resources['memory']
                    del resources['memory']
                    req_memory = resources['req_memory']
                    memory_to_worker_types = memory_to_worker_type(cloud)
                    if req_memory in memory_to_worker_types:
                        worker_type = memory_to_worker_types[req_memory]
                        req_memory_bytes = cores_mcpu_to_memory_bytes(cloud, req_cores_mcpu, worker_type)
                    else:
                        req_memory_bytes = parse_memory_in_bytes(req_memory)
                else:
                    req_cores_mcpu = None
                    req_memory_bytes = None

                if 'storage' not in resources:
                    resources['storage'] = BATCH_JOB_DEFAULT_STORAGE
                resources['req_storage'] = resources['storage']
                del resources['storage']
                req_storage_bytes = parse_storage_in_bytes(resources['req_storage'])

                if req_storage_bytes is None:
                    raise web.HTTPBadRequest(
                        reason=f'bad resource request for job {id}: '
                        f'storage must be convertable to bytes; '
                        f'found {resources["req_storage"]}'
                    )

                inst_coll_configs: InstanceCollectionConfigs = app['inst_coll_configs']

                result, exc = inst_coll_configs.select_inst_coll(
                    cloud, machine_type, preemptible, worker_type, req_cores_mcpu, req_memory_bytes, req_storage_bytes
                )

                if exc:
                    raise web.HTTPBadRequest(reason=exc.message)

                if result is None:
                    raise web.HTTPBadRequest(
                        reason=f'resource requests for job {id} are unsatisfiable: '
                        f'cloud={cloud}, '
                        f'cpu={resources.get("req_cpu")}, '
                        f'memory={resources.get("req_memory")}, '
                        f'storage={resources["req_storage"]}, '
                        f'preemptible={preemptible}, '
                        f'machine_type={machine_type}'
                    )

                inst_coll_name, cores_mcpu, memory_bytes, storage_gib = result
                resources['cores_mcpu'] = cores_mcpu
                resources['memory_bytes'] = memory_bytes
                resources['storage_gib'] = storage_gib
                resources['preemptible'] = preemptible

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

                secrets.append(
                    {
                        'namespace': DEFAULT_NAMESPACE,
                        'name': userdata['hail_credentials_secret_name'],
                        'mount_path': '/gsa-key',
                        'mount_in_copy': True,
                    }
                )

                env = spec.get('env')
                if not env:
                    env = []
                    spec['env'] = env
                assert isinstance(spec['env'], list)

                if cloud == 'gcp' and all(envvar['name'] != 'GOOGLE_APPLICATION_CREDENTIALS' for envvar in spec['env']):
                    spec['env'].append({'name': 'GOOGLE_APPLICATION_CREDENTIALS', 'value': '/gsa-key/key.json'})

                if cloud == 'azure' and all(
                    envvar['name'] != 'AZURE_APPLICATION_CREDENTIALS' for envvar in spec['env']
                ):
                    spec['env'].append({'name': 'AZURE_APPLICATION_CREDENTIALS', 'value': '/gsa-key/key.json'})

                if spec.get('mount_tokens', False):
                    secrets.append(
                        {
                            'namespace': DEFAULT_NAMESPACE,
                            'name': userdata['tokens_secret_name'],
                            'mount_path': '/user-tokens',
                            'mount_in_copy': False,
                        }
                    )
                    secrets.append(
                        {
                            'namespace': DEFAULT_NAMESPACE,
                            'name': 'worker-deploy-config',
                            'mount_path': '/deploy-config',
                            'mount_in_copy': False,
                        }
                    )
                    secrets.append(
                        {
                            'namespace': DEFAULT_NAMESPACE,
                            'name': 'ssl-config-batch-user-code',
                            'mount_path': '/ssl-config',
                            'mount_in_copy': False,
                        }
                    )

                sa = spec.get('service_account')
                check_service_account_permissions(user, sa)

                icr = inst_coll_resources[inst_coll_name]
                icr['n_jobs'] += 1
                if len(parent_ids) == 0:
                    state = 'Ready'
                    icr['n_ready_jobs'] += 1
                    icr['ready_cores_mcpu'] += cores_mcpu
                    if not always_run:
                        icr['n_ready_cancellable_jobs'] += 1
                        icr['ready_cancellable_cores_mcpu'] += cores_mcpu
                else:
                    state = 'Pending'

                network = spec.get('network')
                if user != 'ci' and not (network is None or network == 'public'):
                    raise web.HTTPBadRequest(reason=f'unauthorized network {network}')

                unconfined = spec.get('unconfined')
                if user != 'ci' and unconfined:
                    raise web.HTTPBadRequest(reason=f'unauthorized use of unconfined={unconfined}')

                spec_writer.add(json.dumps(spec))
                db_spec = batch_format_version.db_spec(spec)

                jobs_args.append(
                    (
                        batch_id,
                        job_id,
                        state,
                        json.dumps(db_spec),
                        always_run,
                        cores_mcpu,
                        len(parent_ids),
                        inst_coll_name,
                    )
                )

                for parent_id in parent_ids:
                    job_parents_args.append((batch_id, job_id, parent_id))

                if attributes:
                    for k, v in attributes.items():
                        job_attributes_args.append((batch_id, job_id, k, v))

        rand_token = random.randint(0, app['n_tokens'] - 1)

        async def write_spec_to_cloud():
            if batch_format_version.has_full_spec_in_cloud():
                async with timer.step('write spec to cloud'):
                    await spec_writer.write()

        async def insert_jobs_into_db(tx):
            async with timer.step('insert jobs'):
                try:
                    try:
                        await tx.execute_many(
                            '''
INSERT INTO jobs (batch_id, job_id, state, spec, always_run, cores_mcpu, n_pending_parents, inst_coll)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
''',
                            jobs_args,
                        )
                    except pymysql.err.IntegrityError as err:
                        # 1062 ER_DUP_ENTRY https://dev.mysql.com/doc/refman/5.7/en/server-error-reference.html#error_er_dup_entry
                        if err.args[0] == 1062:
                            log.info(f'bunch containing job {(batch_id, jobs_args[0][1])} already inserted ({err})')
                            return
                        raise
                    try:
                        await tx.execute_many(
                            '''
INSERT INTO `job_parents` (batch_id, job_id, parent_id)
VALUES (%s, %s, %s);
''',
                            job_parents_args,
                        )
                    except pymysql.err.IntegrityError as err:
                        # 1062 ER_DUP_ENTRY https://dev.mysql.com/doc/refman/5.7/en/server-error-reference.html#error_er_dup_entry
                        if err.args[0] == 1062:
                            raise web.HTTPBadRequest(text=f'bunch contains job with duplicated parents ({err})')
                        raise
                    await tx.execute_many(
                        '''
INSERT INTO `job_attributes` (batch_id, job_id, `key`, `value`)
VALUES (%s, %s, %s, %s);
''',
                        job_attributes_args,
                    )

                    batches_inst_coll_staging_args = [
                        (
                            batch_id,
                            inst_coll,
                            rand_token,
                            resources['n_jobs'],
                            resources['n_ready_jobs'],
                            resources['ready_cores_mcpu'],
                        )
                        for inst_coll, resources in inst_coll_resources.items()
                    ]
                    await tx.execute_many(
                        '''
INSERT INTO batches_inst_coll_staging (batch_id, inst_coll, token, n_jobs, n_ready_jobs, ready_cores_mcpu)
VALUES (%s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  n_jobs = n_jobs + VALUES(n_jobs),
  n_ready_jobs = n_ready_jobs + VALUES(n_ready_jobs),
  ready_cores_mcpu = ready_cores_mcpu + VALUES(ready_cores_mcpu);
''',
                        batches_inst_coll_staging_args,
                    )

                    batch_inst_coll_cancellable_resources_args = [
                        (
                            batch_id,
                            inst_coll,
                            rand_token,
                            resources['n_ready_cancellable_jobs'],
                            resources['ready_cancellable_cores_mcpu'],
                        )
                        for inst_coll, resources in inst_coll_resources.items()
                    ]
                    await tx.execute_many(
                        '''
INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
VALUES (%s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  n_ready_cancellable_jobs = n_ready_cancellable_jobs + VALUES(n_ready_cancellable_jobs),
  ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + VALUES(ready_cancellable_cores_mcpu);
''',
                        batch_inst_coll_cancellable_resources_args,
                    )

                    if batch_format_version.has_full_spec_in_cloud():
                        await tx.execute_update(
                            '''
INSERT INTO batch_bunches (batch_id, token, start_job_id)
VALUES (%s, %s, %s);
''',
                            (batch_id, spec_writer.token, start_job_id),
                        )
                except asyncio.CancelledError:
                    raise
                except aiohttp.web.HTTPException:
                    raise
                except Exception as err:
                    raise ValueError(
                        f'encountered exception while inserting a bunch'
                        f'jobs_args={json.dumps(jobs_args)}'
                        f'job_parents_args={json.dumps(job_parents_args)}'
                    ) from err

        @transaction(db)
        async def write_and_insert(tx):
            # IMPORTANT: If cancellation or an error prevents writing the spec to the cloud, then we
            # must rollback. See https://github.com/hail-is/hail-production-issues/issues/9
            await asyncio.gather(write_spec_to_cloud(), insert_jobs_into_db(tx))

        await write_and_insert()  # pylint: disable=no-value-for-parameter
    return web.Response()


@routes.post('/api/v1alpha/batches/create-fast')
@rest_authenticated_users_only
async def create_batch_fast(request, userdata):
    app = request.app
    db: Database = app['db']

    if app['frozen']:
        log.info('ignoring batch create request; batch is frozen')
        raise web.HTTPServiceUnavailable()

    user = userdata['username']
    batch_and_bunch = await request.json()
    batch_spec = batch_and_bunch['batch']
    bunch = batch_and_bunch['bunch']
    batch_id = await _create_batch(batch_spec, userdata, db)
    try:
        await _create_jobs(userdata, bunch, batch_id, app)
    except web.HTTPBadRequest as e:
        if f'batch {batch_id} is not open' == e.reason:
            return web.json_response({'id': batch_id})
        raise
    await _close_batch(app, batch_id, user, db)
    return web.json_response({'id': batch_id})


@routes.post('/api/v1alpha/batches/create')
@rest_authenticated_users_only
async def create_batch(request, userdata):
    app = request.app
    db: Database = app['db']

    if app['frozen']:
        log.info('ignoring batch create jobs request; batch is frozen')
        raise web.HTTPServiceUnavailable()

    batch_spec = await request.json()
    id = await _create_batch(batch_spec, userdata, db)
    return web.json_response({'id': id})


async def _create_batch(batch_spec: dict, userdata: dict, db: Database):
    try:
        validate_batch(batch_spec)
    except ValidationError as e:
        raise web.HTTPBadRequest(reason=e.reason)

    user = userdata['username']

    # restrict to what's necessary; in particular, drop the session
    # which is sensitive
    userdata = {
        'username': user,
        'hail_credentials_secret_name': userdata['hail_credentials_secret_name'],
        'tokens_secret_name': userdata['tokens_secret_name'],
    }

    billing_project = batch_spec['billing_project']
    token = batch_spec['token']

    attributes = batch_spec.get('attributes')

    @transaction(db)
    async def insert(tx):
        bp = await tx.execute_and_fetchone(
            '''
SELECT billing_projects.status, billing_projects.limit
FROM billing_project_users
INNER JOIN billing_projects
  ON billing_projects.name = billing_project_users.billing_project
WHERE billing_project = %s AND user = %s
LOCK IN SHARE MODE''',
            (billing_project, user),
        )

        if bp is None:
            raise web.HTTPForbidden(reason=f'Unknown Hail Batch billing project {billing_project}.')
        if bp['status'] in {'closed', 'deleted'}:
            raise web.HTTPForbidden(reason=f'Billing project {billing_project} is closed or deleted.')

        bp_cost_record = await tx.execute_and_fetchone(
            '''
SELECT billing_projects.msec_mcpu, COALESCE(SUM(`usage` * rate), 0) AS cost
FROM billing_projects
INNER JOIN aggregated_billing_project_resources
  ON billing_projects.name = aggregated_billing_project_resources.billing_project
INNER JOIN resources
  ON resources.resource = aggregated_billing_project_resources.resource
WHERE billing_projects.name = %s
''',
            (billing_project,),
        )
        limit = bp['limit']
        accrued_cost = accrued_cost_from_cost_and_msec_mcpu(bp_cost_record)
        if limit is not None and accrued_cost >= limit:
            raise web.HTTPForbidden(
                reason=f'billing project {billing_project} has exceeded the budget; accrued={cost_str(accrued_cost)} limit={cost_str(limit)}'
            )

        maybe_batch = await tx.execute_and_fetchone(
            '''
SELECT * FROM batches
WHERE token = %s AND user = %s FOR UPDATE;
''',
            (token, user),
        )

        if maybe_batch is not None:
            return maybe_batch['id']

        now = time_msecs()
        id = await tx.execute_insertone(
            '''
INSERT INTO batches (userdata, user, billing_project, attributes, callback, n_jobs, time_created, token, state, format_version, cancel_after_n_failures)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
''',
            (
                json.dumps(userdata),
                user,
                billing_project,
                json.dumps(attributes),
                batch_spec.get('callback'),
                batch_spec['n_jobs'],
                now,
                token,
                'open',
                BATCH_FORMAT_VERSION,
                batch_spec.get('cancel_after_n_failures'),
            ),
        )
        await tx.execute_insertone(
            '''
INSERT INTO batches_n_jobs_in_complete_states (id) VALUES (%s);
''',
            (id,),
        )

        if attributes:
            await tx.execute_many(
                '''
INSERT INTO `batch_attributes` (batch_id, `key`, `value`)
VALUES (%s, %s, %s)
''',
                [(id, k, v) for k, v in attributes.items()],
            )
        return id

    return await insert()  # pylint: disable=no-value-for-parameter


async def _get_batch(app, batch_id):
    db: Database = app['db']

    record = await db.select_and_fetchone(
        '''
SELECT batches.*, batches_cancelled.id IS NOT NULL AS cancelled, COALESCE(SUM(`usage` * rate), 0) AS cost, batches_n_jobs_in_complete_states.n_completed, batches_n_jobs_in_complete_states.n_succeeded, batches_n_jobs_in_complete_states.n_failed, batches_n_jobs_in_complete_states.n_cancelled
FROM batches
LEFT JOIN batches_n_jobs_in_complete_states
       ON batches.id = batches_n_jobs_in_complete_states.id
LEFT JOIN batches_cancelled
       ON batches.id = batches_cancelled.id
LEFT JOIN aggregated_batch_resources
       ON batches.id = aggregated_batch_resources.batch_id
LEFT JOIN resources
       ON aggregated_batch_resources.resource = resources.resource
WHERE batches.id = %s AND NOT deleted
GROUP BY batches.id, batches_cancelled.id;
''',
        (batch_id),
    )
    if not record:
        raise web.HTTPNotFound()

    return batch_record_to_dict(record)


async def _cancel_batch(app, batch_id):
    await cancel_batch_in_db(app['db'], batch_id)
    app['cancel_batch_state_changed'].set()
    return web.Response()


async def _delete_batch(app, batch_id):
    db: Database = app['db']

    record = await db.select_and_fetchone(
        '''
SELECT `state` FROM batches
WHERE id = %s AND NOT deleted;
''',
        (batch_id,),
    )
    if not record:
        raise web.HTTPNotFound()

    await db.just_execute('CALL cancel_batch(%s);', (batch_id,))
    await db.execute_update('UPDATE batches SET deleted = 1 WHERE id = %s;', (batch_id,))

    if record['state'] == 'running':
        app['delete_batch_state_changed'].set()


@routes.get('/api/v1alpha/batches/{batch_id}')
@rest_billing_project_users_only
async def get_batch(request, userdata, batch_id):  # pylint: disable=unused-argument
    return web.json_response(await _get_batch(request.app, batch_id))


@routes.patch('/api/v1alpha/batches/{batch_id}/cancel')
@rest_billing_project_users_only
async def cancel_batch(request, userdata, batch_id):  # pylint: disable=unused-argument
    await _handle_api_error(_cancel_batch, request.app, batch_id)
    return web.Response()


@routes.patch('/api/v1alpha/batches/{batch_id}/close')
@rest_authenticated_users_only
async def close_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']

    app = request.app
    db: Database = app['db']

    if app['frozen']:
        log.info('ignoring batch close request; batch is frozen')
        raise web.HTTPServiceUnavailable()

    record = await db.select_and_fetchone(
        '''
SELECT 1 FROM batches
WHERE user = %s AND id = %s AND NOT deleted;
''',
        (user, batch_id),
    )
    if not record:
        raise web.HTTPNotFound()

    return await _close_batch(app, batch_id, user, db)


async def _close_batch(app: aiohttp.web.Application, batch_id: int, user: str, db: Database):
    client_session: httpx.ClientSession = app['client_session']
    try:
        now = time_msecs()
        await db.check_call_procedure('CALL close_batch(%s, %s);', (batch_id, now))
    except CallError as e:
        # 2: wrong number of jobs
        if e.rv['rc'] == 2:
            expected_n_jobs = e.rv['expected_n_jobs']
            actual_n_jobs = e.rv['actual_n_jobs']
            raise web.HTTPBadRequest(reason=f'wrong number of jobs: expected {expected_n_jobs}, actual {actual_n_jobs}')
        raise

    await request_retry_transient_errors(
        client_session,
        'PATCH',
        deploy_config.url('batch-driver', f'/api/v1alpha/batches/{user}/{batch_id}/close'),
        headers=app['batch_headers'],
    )

    return web.Response()


@routes.delete('/api/v1alpha/batches/{batch_id}')
@rest_billing_project_users_only
async def delete_batch(request, userdata, batch_id):  # pylint: disable=unused-argument
    await _delete_batch(request.app, batch_id)
    return web.Response()


@routes.get('/batches/{batch_id}')
@web_billing_project_users_only()
@catch_ui_error_in_dev
async def ui_batch(request, userdata, batch_id):
    app = request.app
    batch = await _get_batch(app, batch_id)

    jobs, last_job_id = await _query_batch_jobs(request, batch_id)
    for j in jobs:
        j['duration'] = humanize_timedelta_msecs(j['duration'])
        j['cost'] = cost_str(j['cost'])
    batch['jobs'] = jobs

    batch['cost'] = cost_str(batch['cost'])

    page_context = {'batch': batch, 'q': request.query.get('q'), 'last_job_id': last_job_id}
    return await render_template('batch', request, userdata, 'batch.html', page_context)


@routes.post('/batches/{batch_id}/cancel')
@check_csrf_token
@web_billing_project_users_only(redirect=False)
@catch_ui_error_in_dev
async def ui_cancel_batch(request, userdata, batch_id):  # pylint: disable=unused-argument
    post = await request.post()
    q = post.get('q')
    params = {}
    if q is not None:
        params['q'] = q
    session = await aiohttp_session.get_session(request)
    errored = await _handle_ui_error(session, _cancel_batch, request.app, batch_id)
    if not errored:
        set_message(session, f'Batch {batch_id} cancelled.', 'info')
    location = request.app.router['batches'].url_for().with_query(params)
    return web.HTTPFound(location=location)


@routes.post('/batches/{batch_id}/delete')
@check_csrf_token
@web_billing_project_users_only(redirect=False)
@catch_ui_error_in_dev
async def ui_delete_batch(request, userdata, batch_id):  # pylint: disable=unused-argument
    post = await request.post()
    q = post.get('q')
    params = {}
    if q is not None:
        params['q'] = q
    await _delete_batch(request.app, batch_id)
    session = await aiohttp_session.get_session(request)
    set_message(session, f'Batch {batch_id} deleted.', 'info')
    location = request.app.router['batches'].url_for().with_query(params)
    return web.HTTPFound(location=location)


@routes.get('/batches', name='batches')
@web_authenticated_users_only()
@catch_ui_error_in_dev
async def ui_batches(request, userdata):
    user = userdata['username']
    q = request.query.get('q', f'user:{user}')
    batches, last_batch_id = await _query_batches(request, user, q)
    for batch in batches:
        batch['cost'] = cost_str(batch['cost'])
    page_context = {'batches': batches, 'q': q, 'last_batch_id': last_batch_id}
    return await render_template('batch', request, userdata, 'batches.html', page_context)


async def _get_job(app, batch_id, job_id):
    db: Database = app['db']

    record = await db.select_and_fetchone(
        '''
SELECT jobs.*, user, billing_project, ip_address, format_version, COALESCE(SUM(`usage` * rate), 0) AS cost,
  t.attempt_id AS last_cancelled_attempt_id
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
LEFT JOIN (
  SELECT batch_id, job_id, attempt_id
  FROM attempts
  WHERE reason = "cancelled" AND batch_id = %s AND job_id = %s
  ORDER BY end_time DESC
  LIMIT 1
) AS t
  ON jobs.batch_id = t.batch_id AND jobs.job_id = t.job_id
WHERE jobs.batch_id = %s AND NOT deleted AND jobs.job_id = %s
GROUP BY jobs.batch_id, jobs.job_id, t.attempt_id;
''',
        (batch_id, job_id, batch_id, job_id),
    )
    if not record:
        raise web.HTTPNotFound()

    full_status, full_spec, attributes = await asyncio.gather(
        _get_full_job_status(app, record), _get_full_job_spec(app, record), _get_attributes(app, record)
    )

    job = job_record_to_dict(record, attributes.get('name'))
    job['status'] = full_status
    job['spec'] = full_spec
    if attributes:
        job['attributes'] = attributes
    return job


async def _get_attempts(app, batch_id, job_id):
    db: Database = app['db']

    attempts = db.select_and_fetchall(
        '''
SELECT attempts.*
FROM jobs
INNER JOIN batches ON jobs.batch_id = batches.id
LEFT JOIN attempts ON jobs.batch_id = attempts.batch_id and jobs.job_id = attempts.job_id
WHERE jobs.batch_id = %s AND NOT deleted AND jobs.job_id = %s;
''',
        (batch_id, job_id),
        query_name='get_attempts',
    )

    attempts = [attempt async for attempt in attempts]
    if len(attempts) == 0:
        raise web.HTTPNotFound()
    if len(attempts) == 1 and attempts[0]['attempt_id'] is None:
        return None

    attempts.sort(key=lambda x: x['start_time'])

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
@rest_billing_project_users_only
async def get_attempts(request, userdata, batch_id):  # pylint: disable=unused-argument
    job_id = int(request.match_info['job_id'])
    attempts = await _get_attempts(request.app, batch_id, job_id)
    return web.json_response(attempts)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}')
@rest_billing_project_users_only
async def get_job(request, userdata, batch_id):  # pylint: disable=unused-argument
    job_id = int(request.match_info['job_id'])
    status = await _get_job(request.app, batch_id, job_id)
    return web.json_response(status)


@routes.get('/batches/{batch_id}/jobs/{job_id}')
@web_billing_project_users_only()
@catch_ui_error_in_dev
async def ui_get_job(request, userdata, batch_id):
    app = request.app
    job_id = int(request.match_info['job_id'])

    job, attempts, job_log = await asyncio.gather(
        _get_job(app, batch_id, job_id), _get_attempts(app, batch_id, job_id), _get_job_log(app, batch_id, job_id)
    )

    job['duration'] = humanize_timedelta_msecs(job['duration'])
    job['cost'] = cost_str(job['cost'])

    job_status = job['status']
    container_status_spec = dictfix.NoneOr(
        {
            'name': str,
            'timing': {
                'pulling': dictfix.NoneOr({'duration': dictfix.NoneOr(Number)}),
                'running': dictfix.NoneOr({'duration': dictfix.NoneOr(Number)}),
            },
            'short_error': dictfix.NoneOr(str),
            'error': dictfix.NoneOr(str),
            'container_status': {'out_of_memory': dictfix.NoneOr(bool)},
            'state': str,
        }
    )
    job_status_spec = {
        'container_statuses': {
            'input': container_status_spec,
            'main': container_status_spec,
            'output': container_status_spec,
        }
    }
    job_status = dictfix.dictfix(job_status, job_status_spec)
    container_statuses = job_status['container_statuses']
    step_statuses = [container_statuses['input'], container_statuses['main'], container_statuses['output']]
    step_errors = {step: status['error'] for step, status in container_statuses.items() if status is not None}

    for status in step_statuses:
        # backwards compatibility
        if status and status['short_error'] is None and status['container_status']['out_of_memory']:
            status['short_error'] = 'out of memory'

    job_specification = job['spec']
    if job_specification:
        if 'process' in job_specification:
            process_specification = job_specification['process']
            process_type = process_specification['type']
            assert process_specification['type'] in {'docker', 'jvm'}
            job_specification['image'] = process_specification['image'] if process_type == 'docker' else '[jvm]'
            job_specification['command'] = process_specification['command']
        job_specification = dictfix.dictfix(
            job_specification, dictfix.NoneOr({'image': str, 'command': list, 'resources': {}, 'env': list})
        )

    resources = job_specification['resources']
    if 'memory_bytes' in resources:
        resources['actual_memory'] = humanize.naturalsize(resources['memory_bytes'], binary=True)
        del resources['memory_bytes']
    if 'storage_gib' in resources:
        resources['actual_storage'] = humanize.naturalsize(resources['storage_gib'] * 1024**3, binary=True)
        del resources['storage_gib']
    if 'cores_mcpu' in resources:
        resources['actual_cpu'] = resources['cores_mcpu'] / 1000
        del resources['cores_mcpu']

    data = []
    for step in ['input', 'main', 'output']:
        if container_statuses[step]:
            for timing_name, timing_data in container_statuses[step]['timing'].items():
                if timing_data is not None:
                    plot_dict = {
                        'Title': f'{(batch_id, job_id)}',
                        'Step': step,
                        'Task': timing_name,
                    }

                    if timing_data.get('start_time') is not None:
                        plot_dict['Start'] = datetime.datetime.fromtimestamp(timing_data['start_time'] / 1000)

                        finish_time = timing_data.get('finish_time')
                        if finish_time is None:
                            finish_time = time_msecs()
                        plot_dict['Finish'] = datetime.datetime.fromtimestamp(finish_time / 1000)

                    data.append(plot_dict)

    if data:
        df = pd.DataFrame(data)

        fig = px.timeline(
            df,
            x_start='Start',
            x_end='Finish',
            y='Step',
            color='Task',
            hover_data=['Step'],
            color_discrete_sequence=px.colors.sequential.dense,
            category_orders={
                'Step': ['input', 'main', 'output'],
                'Task': ['pulling', 'setting up overlay', 'setting up network', 'running', 'uploading_log'],
            },
        )

        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        plot_json = None

    page_context = {
        'batch_id': batch_id,
        'job_id': job_id,
        'job': job,
        'job_log': job_log,
        'attempts': attempts,
        'step_statuses': step_statuses,
        'job_specification': job_specification,
        'job_status_str': json.dumps(job, indent=2),
        'step_errors': step_errors,
        'error': job_status.get('error'),
        'plot_json': plot_json,
    }

    return await render_template('batch', request, userdata, 'job.html', page_context)


@routes.get('/billing_limits')
@web_authenticated_users_only()
@catch_ui_error_in_dev
async def ui_get_billing_limits(request, userdata):
    app = request.app
    db: Database = app['db']

    if not userdata['is_developer']:
        user = userdata['username']
    else:
        user = None

    billing_projects = await query_billing_projects(db, user=user)

    page_context = {'billing_projects': billing_projects, 'is_developer': userdata['is_developer']}
    return await render_template('batch', request, userdata, 'billing_limits.html', page_context)


def _parse_billing_limit(limit: Optional[Union[str, float, int]]) -> Optional[float]:
    assert isinstance(limit, (str, float, int)) or limit is None, (limit, type(limit))

    if limit == 'None' or limit is None:
        return None
    try:
        parsed_limit = float(limit)
        assert parsed_limit >= 0
        return parsed_limit
    except (AssertionError, ValueError) as e:
        raise InvalidBillingLimitError(limit) from e


async def _edit_billing_limit(db, billing_project, limit):
    limit = _parse_billing_limit(limit)

    @transaction(db)
    async def insert(tx):
        row = await tx.execute_and_fetchone(
            '''
SELECT billing_projects.name as billing_project,
    billing_projects.`status` as `status`
FROM billing_projects
WHERE billing_projects.name = %s AND billing_projects.`status` != 'deleted'
FOR UPDATE;
        ''',
            (billing_project,),
        )
        if row is None:
            raise NonExistentBillingProjectError(billing_project)

        if row['status'] == 'closed':
            raise ClosedBillingProjectError(billing_project)

        await tx.execute_update(
            '''
UPDATE billing_projects SET `limit` = %s WHERE name = %s;
''',
            (limit, billing_project),
        )

    await insert()  # pylint: disable=no-value-for-parameter


@routes.post('/api/v1alpha/billing_limits/{billing_project}/edit')
@rest_authenticated_developers_or_auth_only
async def post_edit_billing_limits(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    data = await request.json()
    limit = data['limit']
    await _handle_api_error(_edit_billing_limit, db, billing_project, limit)
    return web.json_response({'billing_project': billing_project, 'limit': limit})


@routes.post('/billing_limits/{billing_project}/edit')
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_edit_billing_limits_ui(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    post = await request.post()
    limit = post['limit']
    session = await aiohttp_session.get_session(request)
    errored = await _handle_ui_error(session, _edit_billing_limit, db, billing_project, limit)
    if not errored:
        set_message(session, f'Modified limit {limit} for billing project {billing_project}.', 'info')
    return web.HTTPFound(deploy_config.external_url('batch', '/billing_limits'))


async def _query_billing(request, user=None):
    db: Database = request.app['db']

    date_format = '%m/%d/%Y'

    default_start = datetime.datetime.now().replace(day=1)
    default_start = datetime.datetime.strftime(default_start, date_format)

    default_end = None

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
        if end_query is not None and end_query != '':
            end = datetime.datetime.strptime(end_query, date_format)
            end = (end + datetime.timedelta(days=1)).timestamp() * 1000
        else:
            end = None
    except ValueError:
        return await parse_error(f"Invalid value for end '{end_query}'; must be in the format of MM/DD/YYYY.")

    if end is not None and start > end:
        return await parse_error('Invalid search; start must be earlier than end.')

    where_conditions = ["billing_projects.`status` != 'deleted'"]
    where_args = []

    if end is not None:
        where_conditions.append("`time_completed` IS NOT NULL")
        where_conditions.append("`time_completed` >= %s")
        where_args.append(start)
        where_conditions.append("`time_completed` <= %s")
        where_args.append(end)
    else:
        where_conditions.append(
            "((`time_completed` IS NOT NULL AND `time_completed` >= %s) OR "
            "(`time_closed` IS NOT NULL AND `time_completed` IS NULL))"
        )
        where_args.append(start)

    if user is not None:
        where_conditions.append("`user` = %s")
        where_args.append(user)

    sql = f'''
SELECT
  billing_project,
  `user`,
  CAST(SUM(IF(format_version < 3, batches.msec_mcpu, 0)) AS SIGNED) as msec_mcpu,
  SUM(IF(format_version >= 3, `usage` * rate, NULL)) as cost
FROM batches
LEFT JOIN aggregated_batch_resources
  ON aggregated_batch_resources.batch_id = batches.id
LEFT JOIN resources
  ON resources.resource = aggregated_batch_resources.resource
LEFT JOIN billing_projects
  ON billing_projects.name = batches.billing_project
WHERE {' AND '.join(where_conditions)}
GROUP BY billing_project, `user`;
'''

    sql_args = where_args

    def billing_record_to_dict(record):
        cost_msec_mcpu = cost_from_msec_mcpu(record['msec_mcpu'])
        cost_resources = record['cost']
        record['cost'] = coalesce(cost_msec_mcpu, 0) + coalesce(cost_resources, 0)
        del record['msec_mcpu']
        return record

    billing = [billing_record_to_dict(record) async for record in db.select_and_fetchall(sql, sql_args)]

    return (billing, start_query, end_query)


@routes.get('/billing')
@web_authenticated_users_only()
@catch_ui_error_in_dev
async def ui_get_billing(request, userdata):
    is_developer = userdata['is_developer'] == 1
    user = userdata['username'] if not is_developer else None
    billing, start, end = await _query_billing(request, user=user)

    billing_by_user = {}
    billing_by_project = {}
    for record in billing:
        billing_project = record['billing_project']
        user = record['user']
        cost = record['cost']
        billing_by_user[user] = billing_by_user.get(user, 0) + cost
        billing_by_project[billing_project] = billing_by_project.get(billing_project, 0) + cost

    billing_by_project = [
        {'billing_project': billing_project, 'cost': cost_str(cost)}
        for billing_project, cost in billing_by_project.items()
    ]
    billing_by_project.sort(key=lambda record: record['billing_project'])

    billing_by_user = [{'user': user, 'cost': cost_str(cost)} for user, cost in billing_by_user.items()]
    billing_by_user.sort(key=lambda record: record['user'])

    billing_by_project_user = [
        {'billing_project': record['billing_project'], 'user': record['user'], 'cost': cost_str(record['cost'])}
        for record in billing
    ]
    billing_by_project_user.sort(key=lambda record: (record['billing_project'], record['user']))

    page_context = {
        'billing_by_project': billing_by_project,
        'billing_by_user': billing_by_user,
        'billing_by_project_user': billing_by_project_user,
        'start': start,
        'end': end,
        'is_developer': is_developer,
        'user': userdata['username'],
    }
    return await render_template('batch', request, userdata, 'billing.html', page_context)


@routes.get('/billing_projects')
@web_authenticated_developers_only()
@catch_ui_error_in_dev
async def ui_get_billing_projects(request, userdata):
    db: Database = request.app['db']
    billing_projects = await query_billing_projects(db)
    page_context = {
        'billing_projects': [{**p, 'size': len(p['users'])} for p in billing_projects if p['status'] == 'open'],
        'closed_projects': [p for p in billing_projects if p['status'] == 'closed'],
    }
    return await render_template('batch', request, userdata, 'billing_projects.html', page_context)


@routes.get('/api/v1alpha/billing_projects')
@rest_authenticated_users_only
async def get_billing_projects(request, userdata):
    db: Database = request.app['db']

    if not userdata['is_developer'] and userdata['username'] != 'auth':
        user = userdata['username']
    else:
        user = None

    billing_projects = await query_billing_projects(db, user=user)

    return web.json_response(data=billing_projects)


@routes.get('/api/v1alpha/billing_projects/{billing_project}')
@rest_authenticated_users_only
async def get_billing_project(request, userdata):
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']

    if not userdata['is_developer'] and userdata['username'] != 'auth':
        user = userdata['username']
    else:
        user = None

    billing_projects = await query_billing_projects(db, user=user, billing_project=billing_project)

    if not billing_projects:
        raise web.HTTPForbidden(reason=f'Unknown Hail Batch billing project {billing_project}.')

    assert len(billing_projects) == 1
    return web.json_response(data=billing_projects[0])


async def _remove_user_from_billing_project(db, billing_project, user):
    @transaction(db)
    async def delete(tx):
        row = await tx.execute_and_fetchone(
            '''
SELECT billing_projects.name as billing_project,
billing_projects.`status` as `status`,
user FROM billing_projects
LEFT JOIN (SELECT * FROM billing_project_users
    WHERE billing_project = %s AND user = %s FOR UPDATE) AS t
  ON billing_projects.name = t.billing_project
WHERE billing_projects.name = %s;
''',
            (billing_project, user, billing_project),
        )
        if not row:
            raise NonExistentBillingProjectError(billing_project)
        assert row['billing_project'] == billing_project

        if row['status'] in {'closed', 'deleted'}:
            raise BatchUserError(
                f'Billing project {billing_project} has been closed or deleted and cannot be modified.', 'error'
            )

        if row['user'] is None:
            raise BatchOperationAlreadyCompletedError(
                f'User {user} is not in billing project {billing_project}.', 'info'
            )

        await tx.just_execute(
            '''
DELETE FROM billing_project_users
WHERE billing_project = %s AND user = %s;
''',
            (billing_project, user),
        )

    await delete()  # pylint: disable=no-value-for-parameter


@routes.post('/billing_projects/{billing_project}/users/{user}/remove')
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_billing_projects_remove_user(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    user = request.match_info['user']

    session = await aiohttp_session.get_session(request)
    errored = await _handle_ui_error(session, _remove_user_from_billing_project, db, billing_project, user)
    if not errored:
        set_message(session, f'Removed user {user} from billing project {billing_project}.', 'info')
    return web.HTTPFound(deploy_config.external_url('batch', '/billing_projects'))


@routes.post('/api/v1alpha/billing_projects/{billing_project}/users/{user}/remove')
@rest_authenticated_developers_or_auth_only
async def api_get_billing_projects_remove_user(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    user = request.match_info['user']
    await _handle_api_error(_remove_user_from_billing_project, db, billing_project, user)
    return web.json_response({'billing_project': billing_project, 'user': user})


async def _add_user_to_billing_project(db, billing_project, user):
    @transaction(db)
    async def insert(tx):
        row = await tx.execute_and_fetchone(
            '''
SELECT billing_projects.name as billing_project,
    billing_projects.`status` as `status`,
    user
FROM billing_projects
LEFT JOIN (SELECT * FROM billing_project_users
WHERE billing_project = %s AND user = %s FOR UPDATE) AS t
ON billing_projects.name = t.billing_project
WHERE billing_projects.name = %s AND billing_projects.`status` != 'deleted' LOCK IN SHARE MODE;
        ''',
            (billing_project, user, billing_project),
        )
        if row is None:
            raise NonExistentBillingProjectError(billing_project)

        if row['status'] == 'closed':
            raise ClosedBillingProjectError(billing_project)

        if row['user'] is not None:
            raise BatchOperationAlreadyCompletedError(
                f'User {user} is already member of billing project {billing_project}.', 'info'
            )
        await tx.execute_insertone(
            '''
INSERT INTO billing_project_users(billing_project, user)
VALUES (%s, %s);
        ''',
            (billing_project, user),
        )

    await insert()  # pylint: disable=no-value-for-parameter


@routes.post('/billing_projects/{billing_project}/users/add')
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_billing_projects_add_user(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    post = await request.post()
    user = post['user']
    billing_project = request.match_info['billing_project']

    session = await aiohttp_session.get_session(request)

    errored = await _handle_ui_error(session, _add_user_to_billing_project, db, billing_project, user)
    if not errored:
        set_message(session, f'Added user {user} to billing project {billing_project}.', 'info')
    return web.HTTPFound(deploy_config.external_url('batch', '/billing_projects'))


@routes.post('/api/v1alpha/billing_projects/{billing_project}/users/{user}/add')
@rest_authenticated_developers_or_auth_only
async def api_billing_projects_add_user(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    user = request.match_info['user']
    billing_project = request.match_info['billing_project']

    await _handle_api_error(_add_user_to_billing_project, db, billing_project, user)
    return web.json_response({'billing_project': billing_project, 'user': user})


async def _create_billing_project(db, billing_project):
    @transaction(db)
    async def insert(tx):
        row = await tx.execute_and_fetchone(
            '''
SELECT `status` FROM billing_projects
WHERE name = %s
FOR UPDATE;
''',
            (billing_project),
        )
        if row is not None:
            raise BatchOperationAlreadyCompletedError(f'Billing project {billing_project} already exists.', 'info')

        await tx.execute_insertone(
            '''
INSERT INTO billing_projects(name)
VALUES (%s);
''',
            (billing_project,),
        )

    await insert()  # pylint: disable=no-value-for-parameter


@routes.post('/billing_projects/create')
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_create_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    post = await request.post()
    billing_project = post['billing_project']

    session = await aiohttp_session.get_session(request)
    errored = await _handle_ui_error(session, _create_billing_project, db, billing_project)
    if not errored:
        set_message(session, f'Added billing project {billing_project}.', 'info')

    return web.HTTPFound(deploy_config.external_url('batch', '/billing_projects'))


@routes.post('/api/v1alpha/billing_projects/{billing_project}/create')
@rest_authenticated_developers_or_auth_only
async def api_get_create_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    await _handle_api_error(_create_billing_project, db, billing_project)
    return web.json_response(billing_project)


async def _close_billing_project(db, billing_project):
    @transaction(db)
    async def close_project(tx):
        row = await tx.execute_and_fetchone(
            '''
SELECT name, `status`, batches.id as batch_id
FROM billing_projects
LEFT JOIN batches
ON billing_projects.name = batches.billing_project
AND billing_projects.`status` != 'deleted'
AND batches.time_completed IS NULL
AND NOT batches.deleted
WHERE name = %s
LIMIT 1
FOR UPDATE;
    ''',
            (billing_project,),
        )
        if not row:
            raise NonExistentBillingProjectError(billing_project)
        assert row['name'] == billing_project
        if row['status'] == 'closed':
            raise BatchOperationAlreadyCompletedError(
                f'Billing project {billing_project} is already closed or deleted.', 'info'
            )
        if row['batch_id'] is not None:
            raise BatchUserError(f'Billing project {billing_project} has open or running batches.', 'error')

        await tx.execute_update("UPDATE billing_projects SET `status` = 'closed' WHERE name = %s;", (billing_project,))

    await close_project()  # pylint: disable=no-value-for-parameter


@routes.post('/billing_projects/{billing_project}/close')
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_close_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']

    session = await aiohttp_session.get_session(request)
    errored = await _handle_ui_error(session, _close_billing_project, db, billing_project)
    if not errored:
        set_message(session, f'Closed billing project {billing_project}.', 'info')
    return web.HTTPFound(deploy_config.external_url('batch', '/billing_projects'))


@routes.post('/api/v1alpha/billing_projects/{billing_project}/close')
@rest_authenticated_developers_or_auth_only
async def api_close_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']

    await _handle_api_error(_close_billing_project, db, billing_project)
    return web.json_response(billing_project)


async def _reopen_billing_project(db, billing_project):
    @transaction(db)
    async def open_project(tx):
        row = await tx.execute_and_fetchone(
            "SELECT name, `status` FROM billing_projects WHERE name = %s FOR UPDATE;", (billing_project,)
        )
        if not row:
            raise NonExistentBillingProjectError(billing_project)
        assert row['name'] == billing_project
        if row['status'] == 'deleted':
            raise BatchUserError(f'Billing project {billing_project} has been deleted and cannot be reopened.', 'error')
        if row['status'] == 'open':
            raise BatchOperationAlreadyCompletedError(f'Billing project {billing_project} is already open.', 'info')

        await tx.execute_update("UPDATE billing_projects SET `status` = 'open' WHERE name = %s;", (billing_project,))

    await open_project()  # pylint: disable=no-value-for-parameter


@routes.post('/billing_projects/{billing_project}/reopen')
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_reopen_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']

    session = await aiohttp_session.get_session(request)
    errored = await _handle_ui_error(session, _reopen_billing_project, db, billing_project)
    if not errored:
        set_message(session, f'Re-opened billing project {billing_project}.', 'info')
    return web.HTTPFound(deploy_config.external_url('batch', '/billing_projects'))


@routes.post('/api/v1alpha/billing_projects/{billing_project}/reopen')
@rest_authenticated_developers_or_auth_only
async def api_reopen_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    await _handle_api_error(_reopen_billing_project, db, billing_project)
    return web.json_response(billing_project)


async def _delete_billing_project(db, billing_project):
    @transaction(db)
    async def delete_project(tx):
        row = await tx.execute_and_fetchone(
            'SELECT name, `status` FROM billing_projects WHERE name = %s FOR UPDATE;', (billing_project,)
        )
        if not row:
            raise NonExistentBillingProjectError(billing_project)
        assert row['name'] == billing_project
        if row['status'] == 'deleted':
            raise BatchOperationAlreadyCompletedError(f'Billing project {billing_project} is already deleted.', 'info')
        if row['status'] == 'open':
            raise BatchUserError(f'Billing project {billing_project} is open and cannot be deleted.', 'error')

        await tx.execute_update("UPDATE billing_projects SET `status` = 'deleted' WHERE name = %s;", (billing_project,))

    await delete_project()  # pylint: disable=no-value-for-parameter


@routes.post('/api/v1alpha/billing_projects/{billing_project}/delete')
@rest_authenticated_developers_or_auth_only
async def api_delete_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']

    await _handle_api_error(_delete_billing_project, db, billing_project)
    return web.json_response(billing_project)


async def _refresh(app):
    db: Database = app['db']
    inst_coll_configs: InstanceCollectionConfigs = app['inst_coll_configs']
    await inst_coll_configs.refresh(db)
    row = await db.select_and_fetchone(
        '''
SELECT frozen FROM globals;
'''
    )
    app['frozen'] = row['frozen']


@routes.get('')
@routes.get('/')
@web_authenticated_users_only()
@catch_ui_error_in_dev
async def index(request, userdata):  # pylint: disable=unused-argument
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


async def cancel_batch_loop_body(app):
    client_session: httpx.ClientSession = app['client_session']
    await request_retry_transient_errors(
        client_session,
        'POST',
        deploy_config.url('batch-driver', '/api/v1alpha/batches/cancel'),
        headers=app['batch_headers'],
    )

    should_wait = True
    return should_wait


async def delete_batch_loop_body(app):
    client_session: httpx.ClientSession = app['client_session']
    await request_retry_transient_errors(
        client_session,
        'POST',
        deploy_config.url('batch-driver', '/api/v1alpha/batches/delete'),
        headers=app['batch_headers'],
    )

    should_wait = True
    return should_wait


async def on_startup(app):
    app['task_manager'] = aiotools.BackgroundTaskManager()
    app['client_session'] = httpx.client_session()

    db = Database()
    await db.async_init()
    app['db'] = db

    row = await db.select_and_fetchone(
        '''
SELECT instance_id, internal_token, n_tokens, frozen FROM globals;
'''
    )

    app['n_tokens'] = row['n_tokens']

    instance_id = row['instance_id']
    log.info(f'instance_id {instance_id}')
    app['instance_id'] = instance_id

    app['internal_token'] = row['internal_token']

    app['batch_headers'] = {'Authorization': f'Bearer {row["internal_token"]}'}

    app['frozen'] = row['frozen']

    fs = get_cloud_async_fs(credentials_file='/gsa-key/key.json')
    app['file_store'] = FileStore(fs, BATCH_STORAGE_URI, instance_id)

    app['inst_coll_configs'] = await InstanceCollectionConfigs.create(db)

    cancel_batch_state_changed = asyncio.Event()
    app['cancel_batch_state_changed'] = cancel_batch_state_changed

    app['task_manager'].ensure_future(
        retry_long_running('cancel_batch_loop', run_if_changed, cancel_batch_state_changed, cancel_batch_loop_body, app)
    )

    delete_batch_state_changed = asyncio.Event()
    app['delete_batch_state_changed'] = delete_batch_state_changed

    app['task_manager'].ensure_future(
        retry_long_running('delete_batch_loop', run_if_changed, delete_batch_state_changed, delete_batch_loop_body, app)
    )

    app['task_manager'].ensure_future(periodically_call(5, _refresh, app))


async def on_cleanup(app):
    try:
        app['task_manager'].shutdown()
    finally:
        try:
            await app['client_session'].close()
        finally:
            try:
                await app['file_store'].close()
            finally:
                await app['db'].async_close()


def run():
    app = web.Application(client_max_size=HTTP_CLIENT_MAX_SIZE, middlewares=[monitor_endpoints_middleware])
    setup_aiohttp_session(app)

    setup_aiohttp_jinja2(app, 'batch.front_end')
    setup_common_static_routes(routes)
    app.add_routes(routes)
    app.router.add_get("/metrics", server_stats)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    asyncio.get_event_loop().add_signal_handler(signal.SIGUSR1, dump_all_stacktraces)

    web.run_app(
        deploy_config.prefix_application(app, 'batch', client_max_size=HTTP_CLIENT_MAX_SIZE),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=internal_server_ssl_context(),
    )
