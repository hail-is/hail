import asyncio
import base64
import collections
import datetime
import json
import logging
import os
import random
import re
import signal
import traceback
from functools import wraps
from numbers import Number
from typing import Awaitable, Callable, Dict, Optional, Tuple, TypeVar, Union

import aiohttp
import aiohttp_session
import humanize
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pymysql
import uvloop
from aiohttp import web
from plotly.subplots import make_subplots
from prometheus_async.aio.web import server_stats  # type: ignore
from typing_extensions import ParamSpec

from gear import (
    AuthClient,
    Database,
    Transaction,
    check_csrf_token,
    json_request,
    json_response,
    monitor_endpoints_middleware,
    setup_aiohttp_session,
    transaction,
)
from gear.clients import get_cloud_async_fs
from gear.database import CallError
from gear.profiling import install_profiler_if_requested
from hailtop import aiotools, dictfix, httpx, version
from hailtop.batch_client.parse import parse_cpu_in_mcpu, parse_memory_in_bytes, parse_storage_in_bytes
from hailtop.config import get_deploy_config
from hailtop.hail_logging import AccessLogger
from hailtop.tls import internal_server_ssl_context
from hailtop.utils import (
    cost_str,
    dump_all_stacktraces,
    humanize_timedelta_msecs,
    periodically_call,
    retry_long_running,
    retry_transient_errors,
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
    is_valid_cores_mcpu,
    memory_to_worker_type,
    valid_machine_types,
)
from ..cloud.utils import ACCEPTABLE_QUERY_JAR_URL_PREFIX
from ..exceptions import (
    BatchOperationAlreadyCompletedError,
    BatchUserError,
    ClosedBillingProjectError,
    InvalidBillingLimitError,
    NonExistentBillingProjectError,
    QueryError,
)
from ..file_store import FileStore
from ..globals import BATCH_FORMAT_VERSION, HTTP_CLIENT_MAX_SIZE, RESERVED_STORAGE_GB_PER_CORE, complete_states
from ..inst_coll_config import InstanceCollectionConfigs
from ..resource_usage import ResourceUsageMonitor
from ..spec_writer import SpecWriter
from ..utils import (
    add_metadata_to_request,
    query_billing_projects_with_cost,
    query_billing_projects_without_cost,
    regions_to_bits_rep,
    unavailable_if_frozen,
)
from .query import (
    CURRENT_QUERY_VERSION,
    parse_batch_jobs_query_v1,
    parse_batch_jobs_query_v2,
    parse_list_batches_query_v1,
    parse_list_batches_query_v2,
)
from .validate import ValidationError, validate_and_clean_jobs, validate_batch, validate_batch_update

uvloop.install()

log = logging.getLogger('batch.front_end')

routes = web.RouteTableDef()

deploy_config = get_deploy_config()

auth = AuthClient()

BATCH_JOB_DEFAULT_CPU = os.environ.get('HAIL_BATCH_JOB_DEFAULT_CPU', '1')
BATCH_JOB_DEFAULT_MEMORY = os.environ.get('HAIL_BATCH_JOB_DEFAULT_MEMORY', 'standard')
BATCH_JOB_DEFAULT_STORAGE = os.environ.get('HAIL_BATCH_JOB_DEFAULT_STORAGE', '0Gi')
BATCH_JOB_DEFAULT_PREEMPTIBLE = True


T = TypeVar('T')
P = ParamSpec('P')


def rest_authenticated_developers_or_auth_only(fun):
    @auth.rest_authenticated_users_only
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
WHERE id = %s AND billing_project_users.`user_cs` = %s;
''',
        (batch_id, user),
    )

    return record is not None


def rest_billing_project_users_only(fun):
    @auth.rest_authenticated_users_only
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
        @auth.web_authenticated_users_only(redirect)
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


def cast_query_param_to_int(param: Optional[str]) -> Optional[int]:
    if param is not None:
        return int(param)
    return None


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


@routes.get('/api/v1alpha/version')
async def rest_get_version(request):  # pylint: disable=W0613
    return web.Response(text=version())


@routes.get('/api/v1alpha/cloud')
async def rest_cloud(request):  # pylint: disable=W0613
    return web.Response(text=CLOUD)


@routes.get('/api/v1alpha/supported_regions')
@auth.rest_authenticated_users_only
async def rest_get_supported_regions(request, userdata):  # pylint: disable=unused-argument
    return json_response(list(request.app['regions'].keys()))


async def _handle_ui_error(
    session: aiohttp_session.Session, f: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs
) -> T:
    try:
        return await f(*args, **kwargs)
    except KeyError as e:
        set_message(session, str(e), 'error')
        log.info(f'ui error: KeyError {e}')
        raise
    except BatchOperationAlreadyCompletedError as e:
        set_message(session, e.message, e.ui_error_type)
        log.info(f'ui error: BatchOperationAlreadyCompletedError {e.message}')
        raise
    except BatchUserError as e:
        set_message(session, e.message, e.ui_error_type)
        log.info(f'ui error: BatchUserError {e.message}')
        raise


async def _handle_api_error(f: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> Optional[T]:
    try:
        return await f(*args, **kwargs)
    except BatchOperationAlreadyCompletedError as e:
        log.info(e.message)
        return None
    except BatchUserError as e:
        raise e.http_response()


async def _query_batch_jobs(request: web.Request, batch_id: int, version: int, q: str, last_job_id: Optional[int]):
    db: Database = request.app['db']
    if version == 1:
        sql, sql_args = parse_batch_jobs_query_v1(batch_id, q, last_job_id)
    else:
        assert version == 2, version
        sql, sql_args = parse_batch_jobs_query_v2(batch_id, q, last_job_id)

    jobs = [job_record_to_dict(record, record['name']) async for record in db.select_and_fetchall(sql, sql_args)]

    if len(jobs) == 50:
        last_job_id = jobs[-1]['job_id']
    else:
        last_job_id = None

    return (jobs, last_job_id)


async def _get_jobs(request, batch_id: int, version: int, q: str, last_job_id: Optional[int]):
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

    jobs, last_job_id = await _query_batch_jobs(request, batch_id, version, q, last_job_id)

    resp = {'jobs': jobs}
    if last_job_id is not None:
        resp['last_job_id'] = last_job_id
    return resp


@routes.get('/api/v1alpha/batches/{batch_id}/jobs')
@rest_billing_project_users_only
@add_metadata_to_request
async def get_jobs_v1(request: web.Request, userdata: dict, batch_id: int):  # pylint: disable=unused-argument
    q = request.query.get('q', '')
    last_job_id = cast_query_param_to_int(request.query.get('last_job_id'))
    resp = await _handle_api_error(_get_jobs, request, batch_id, 1, q, last_job_id)
    assert resp is not None
    return json_response(resp)


@routes.get('/api/v2alpha/batches/{batch_id}/jobs')
@rest_billing_project_users_only
@add_metadata_to_request
async def get_jobs_v2(request: web.Request, userdata: dict, batch_id: int):  # pylint: disable=unused-argument
    q = request.query.get('q', '')
    last_job_id = cast_query_param_to_int(request.query.get('last_job_id'))
    resp = await _handle_api_error(_get_jobs, request, batch_id, 2, q, last_job_id)
    assert resp is not None
    return json_response(resp)


async def _get_job_record(app, batch_id, job_id):
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
    return record


def job_tasks_from_spec(record):
    batch_format_version = BatchFormatVersion(record['format_version'])
    spec = json.loads(record['spec'])
    tasks = []

    has_input_files = batch_format_version.get_spec_has_input_files(spec)
    if has_input_files:
        tasks.append('input')

    tasks.append('main')

    has_output_files = batch_format_version.get_spec_has_output_files(spec)
    if has_output_files:
        tasks.append('output')

    return tasks


def has_resource_available(record):
    state = record['state']
    if state in ('Pending', 'Ready', 'Creating'):
        return False
    if state == 'Cancelled' and record['last_cancelled_attempt_id'] is None:
        return False
    if state == 'Running':
        return True
    assert state in complete_states, state
    return True


def attempt_id_from_spec(record) -> Optional[str]:
    return record['attempt_id'] or record['last_cancelled_attempt_id']


async def _get_job_container_log_from_worker(client_session, batch_id, job_id, container, ip_address) -> bytes:
    try:
        return await retry_transient_errors(
            client_session.get_read,
            f'http://{ip_address}:5000/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log/{container}',
        )
    except aiohttp.ClientResponseError:
        log.exception(f'while getting log for {(batch_id, job_id)}')
        return b'ERROR: encountered a problem while fetching the log'


async def _read_job_container_log_from_cloud_storage(
    file_store: FileStore, batch_format_version: BatchFormatVersion, batch_id, job_id, container, attempt_id
) -> bytes:
    try:
        return await file_store.read_log_file(batch_format_version, batch_id, job_id, attempt_id, container)
    except FileNotFoundError:
        id = (batch_id, job_id)
        log.exception(f'missing log file for {id} and container {container}')
        return b'ERROR: could not find log file'


async def _get_job_container_log(app, batch_id, job_id, container, job_record) -> Optional[bytes]:
    if not has_resource_available(job_record):
        return None

    state = job_record['state']
    if state == 'Running':
        return await _get_job_container_log_from_worker(
            app['client_session'], batch_id, job_id, container, job_record['ip_address']
        )

    attempt_id = attempt_id_from_spec(job_record)
    assert attempt_id is not None and state in complete_states
    return await _read_job_container_log_from_cloud_storage(
        app['file_store'],
        BatchFormatVersion(job_record['format_version']),
        batch_id,
        job_id,
        container,
        attempt_id,
    )


async def _get_job_log(app, batch_id, job_id) -> Dict[str, Optional[bytes]]:
    record = await _get_job_record(app, batch_id, job_id)
    containers = job_tasks_from_spec(record)
    logs = await asyncio.gather(*[_get_job_container_log(app, batch_id, job_id, c, record) for c in containers])
    return dict(zip(containers, logs))


async def _get_job_resource_usage(app, batch_id, job_id):
    record = await _get_job_record(app, batch_id, job_id)

    client_session: httpx.ClientSession = app['client_session']
    file_store: FileStore = app['file_store']
    batch_format_version = BatchFormatVersion(record['format_version'])

    state = record['state']
    ip_address = record['ip_address']
    tasks = job_tasks_from_spec(record)
    attempt_id = attempt_id_from_spec(record)

    if not has_resource_available(record):
        return None

    if state == 'Running':
        try:
            data = await retry_transient_errors(
                client_session.get_read_json,
                f'http://{ip_address}:5000/api/v1alpha/batches/{batch_id}/jobs/{job_id}/resource_usage',
            )
            return {
                task: ResourceUsageMonitor.decode_to_df(base64.b64decode(encoded_df))
                for task, encoded_df in data.items()
            }
        except aiohttp.ClientResponseError:
            log.exception(f'while getting resource usage for {(batch_id, job_id)}')
            return {task: None for task in tasks}

    assert attempt_id is not None and state in complete_states

    async def _read_resource_usage_from_cloud_storage(task):
        try:
            df = await file_store.read_resource_usage_file(batch_format_version, batch_id, job_id, attempt_id, task)
        except FileNotFoundError:
            id = (batch_id, job_id)
            log.exception(f'missing resource usage file for {id} and task {task}')
            df = None
        return task, df

    return dict(await asyncio.gather(*[_read_resource_usage_from_cloud_storage(task) for task in tasks]))


async def _get_jvm_profile(app, batch_id, job_id):
    record = await _get_job_record(app, batch_id, job_id)

    file_store: FileStore = app['file_store']
    batch_format_version = BatchFormatVersion(record['format_version'])

    state = record['state']
    attempt_id = attempt_id_from_spec(record)

    if not has_resource_available(record):
        return None

    if state == 'Running':
        return None

    assert attempt_id is not None and state in complete_states

    try:
        data = await file_store.read_jvm_profile(batch_format_version, batch_id, job_id, attempt_id, 'main')
        return data.decode('utf-8')
    except FileNotFoundError:
        return None


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
        return await retry_transient_errors(
            client_session.get_read_json,
            f'http://{ip_address}:5000/api/v1alpha/batches/{batch_id}/jobs/{job_id}/status',
        )
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            return None
        raise


# deprecated
@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
@rest_billing_project_users_only
@add_metadata_to_request
async def get_job_log(request, userdata, batch_id):  # pylint: disable=unused-argument
    job_id = int(request.match_info['job_id'])
    job_log_bytes = await _get_job_log(request.app, batch_id, job_id)
    job_log_strings: Dict[str, Optional[str]] = {}
    for container, log in job_log_bytes.items():
        try:
            job_log_strings[container] = log.decode('utf-8') if log is not None else None
        except UnicodeDecodeError as e:
            raise web.HTTPBadRequest(
                reason=f'log for container {container} is not valid UTF-8, upgrade your hail version to download the log'
            ) from e
    return json_response(job_log_strings)


async def get_job_container_log(request, batch_id):
    app = request.app
    job_id = int(request.match_info['job_id'])
    container = request.match_info['container']
    record = await _get_job_record(app, batch_id, job_id)
    containers = job_tasks_from_spec(record)
    if container not in containers:
        raise web.HTTPBadRequest(reason=f'unknown container {container}')
    job_log = await _get_job_container_log(app, batch_id, job_id, container, record)
    return web.Response(body=job_log)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log/{container}')
@rest_billing_project_users_only
@add_metadata_to_request
async def rest_get_job_container_log(request, userdata, batch_id):  # pylint: disable=unused-argument
    return await get_job_container_log(request, batch_id)


async def _query_batches(request, user: str, q: str, version: int, last_batch_id: Optional[int]):
    db: Database = request.app['db']
    if version == 1:
        sql, sql_args = parse_list_batches_query_v1(user, q, last_batch_id)
    else:
        assert version == 2, version
        sql, sql_args = parse_list_batches_query_v2(user, q, last_batch_id)

    batches = [batch_record_to_dict(record) async for record in db.select_and_fetchall(sql, sql_args)]

    if len(batches) == 51:
        batches.pop()
        last_batch_id = batches[-1]['id']
    else:
        last_batch_id = None

    return (batches, last_batch_id)


@routes.get('/api/v1alpha/batches')
@auth.rest_authenticated_users_only
@add_metadata_to_request
async def get_batches_v1(request, userdata):  # pylint: disable=unused-argument
    user = userdata['username']
    q = request.query.get('q', f'user:{user}')
    last_batch_id = cast_query_param_to_int(request.query.get('last_batch_id'))
    result = await _handle_api_error(_query_batches, request, user, q, 1, last_batch_id)
    assert result is not None
    batches, last_batch_id = result
    body = {'batches': batches}
    if last_batch_id is not None:
        body['last_batch_id'] = last_batch_id
    return json_response(body)


@routes.get('/api/v2alpha/batches')
@auth.rest_authenticated_users_only
@add_metadata_to_request
async def get_batches_v2(request, userdata):  # pylint: disable=unused-argument
    user = userdata['username']
    q = request.query.get('q', f'user = {user}')
    last_batch_id = cast_query_param_to_int(request.query.get('last_batch_id'))
    result = await _handle_api_error(_query_batches, request, user, q, 2, last_batch_id)
    assert result is not None
    batches, last_batch_id = result
    body = {'batches': batches}
    if last_batch_id is not None:
        body['last_batch_id'] = last_batch_id
    return json_response(body)


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


# Deprecated. Use create_jobs_for_update instead
@routes.post('/api/v1alpha/batches/{batch_id}/jobs/create')
@auth.rest_authenticated_users_only
@add_metadata_to_request
async def create_jobs(request: aiohttp.web.Request, userdata: dict):
    app = request.app
    batch_id = int(request.match_info['batch_id'])
    job_specs = await json_request(request)
    return await _create_jobs(userdata, job_specs, batch_id, 1, app)


@routes.post('/api/v1alpha/batches/{batch_id}/updates/{update_id}/jobs/create')
@auth.rest_authenticated_users_only
@add_metadata_to_request
async def create_jobs_for_update(request: aiohttp.web.Request, userdata: dict):
    app = request.app

    if app['frozen']:
        log.info('ignoring batch create request; batch is frozen')
        raise web.HTTPServiceUnavailable()

    batch_id = int(request.match_info['batch_id'])
    update_id = int(request.match_info['update_id'])
    job_specs = await json_request(request)
    return await _create_jobs(userdata, job_specs, batch_id, update_id, app)


NON_HEX_DIGIT = re.compile('[^A-Fa-f0-9]')


def assert_is_sha_1_hex_string(revision: str):
    if len(revision) != 40 or NON_HEX_DIGIT.search(revision):
        raise web.HTTPBadRequest(reason=f'revision must be 40 character hexadecimal encoded SHA-1, got: {revision}')


async def _create_jobs(userdata: dict, job_specs: dict, batch_id: int, update_id: int, app: aiohttp.web.Application):
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

    record = await db.select_and_fetchone(
        '''
SELECT `state`, format_version, `committed`, start_job_id
FROM batch_updates
INNER JOIN batches ON batch_updates.batch_id = batches.id
WHERE batch_updates.batch_id = %s AND batch_updates.update_id = %s AND user = %s AND NOT deleted;
''',
        (batch_id, update_id, user),
    )

    if not record:
        raise web.HTTPNotFound()
    if record['committed']:
        raise web.HTTPBadRequest(reason=f'update {update_id} is already committed')
    batch_format_version = BatchFormatVersion(record['format_version'])
    update_start_job_id = int(record['start_job_id'])

    try:
        validate_and_clean_jobs(job_specs)
    except ValidationError as e:
        raise web.HTTPBadRequest(reason=e.reason)

    spec_writer = SpecWriter(file_store, batch_id)

    jobs_args = []
    job_parents_args = []
    job_attributes_args = []
    jobs_telemetry_args = []

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
    bunch_start_job_id = None

    for spec in job_specs:
        job_id = spec['job_id'] + update_start_job_id - 1
        spec['job_id'] = job_id

        absolute_parent_ids = spec.pop('absolute_parent_ids', [])
        in_update_parent_ids = spec.pop('in_update_parent_ids', [])
        parent_ids = absolute_parent_ids + [update_start_job_id + parent_id - 1 for parent_id in in_update_parent_ids]

        always_run = spec.pop('always_run', False)

        cloud = spec.get('cloud', CLOUD)

        if batch_format_version.has_full_spec_in_cloud():
            attributes = spec.pop('attributes', None)
        else:
            attributes = spec.get('attributes')

        id = (batch_id, job_id)

        if bunch_start_job_id is None:
            bunch_start_job_id = job_id

        if batch_format_version.has_full_spec_in_cloud() and prev_job_idx:
            if job_id != prev_job_idx + 1:
                raise web.HTTPBadRequest(reason=f'noncontiguous job ids found in the spec: {prev_job_idx} -> {job_id}')
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
            jvm_requested_cpu = parse_cpu_in_mcpu(resources.get('cpu', BATCH_JOB_DEFAULT_CPU))
            if 'cpu' in resources and jvm_requested_cpu not in (1000, 2000, 4000, 8000):
                raise web.HTTPBadRequest(reason='invalid cpu for jvm jobs. must be 1, 2, 4, or 8')
            if 'memory' in resources and resources['memory'] == 'lowmem':
                raise web.HTTPBadRequest(reason='jvm jobs cannot be on lowmem machines')
            if machine_type is not None:
                raise web.HTTPBadRequest(reason='jvm jobs may not specify machine_type')
            if spec['process']['jar_spec']['type'] == 'git_revision':
                revision = spec['process']['jar_spec']['value']
                assert_is_sha_1_hex_string(revision)
                spec['process']['jar_spec']['type'] = 'jar_url'
                spec['process']['jar_spec']['value'] = ACCEPTABLE_QUERY_JAR_URL_PREFIX + '/' + revision + '.jar'
            else:
                assert spec['process']['jar_spec']['type'] == 'jar_url'
                jar_url = spec['process']['jar_spec']['value']
                if not jar_url.startswith(ACCEPTABLE_QUERY_JAR_URL_PREFIX):
                    raise web.HTTPBadRequest(reason=f'unacceptable JAR url: {jar_url}')

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

        regions = spec.get('regions')
        if regions is not None:
            valid_regions = set(app['regions'].keys())
            invalid_user_regions = set(regions).difference(valid_regions)
            if invalid_user_regions:
                raise web.HTTPBadRequest(
                    reason=f'invalid regions specified: {invalid_user_regions}. Choose from {valid_regions}'
                )
            if len(regions) == 0:
                raise web.HTTPBadRequest(reason='regions must not be an empty array')
            n_regions = len(regions)
            regions_bits_rep = regions_to_bits_rep(regions, app['regions'])
        else:
            n_regions = None
            regions_bits_rep = None

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

        if cloud == 'azure' and all(envvar['name'] != 'AZURE_APPLICATION_CREDENTIALS' for envvar in spec['env']):
            spec['env'].append({'name': 'AZURE_APPLICATION_CREDENTIALS', 'value': '/gsa-key/key.json'})

        cloudfuse = spec.get('gcsfuse') or spec.get('cloudfuse')
        if cloudfuse:
            for config in cloudfuse:
                if not config['read_only']:
                    raise web.HTTPBadRequest(reason=f'Only read-only cloudfuse requests are supported. Found {config}')
                if config['mount_path'] == '/io':
                    raise web.HTTPBadRequest(
                        reason=f'Cloudfuse requests with mount_path=/io are not supported. Found {config}'
                    )

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
                    'name': 'ssl-config-batch-user-code',
                    'mount_path': '/ssl-config',
                    'mount_in_copy': False,
                }
            )

        sa = spec.get('service_account')
        check_service_account_permissions(user, sa)

        icr = inst_coll_resources[inst_coll_name]
        icr['n_jobs'] += 1

        # jobs in non-initial updates of a batch always start out as pending
        # because they may have currently running parents in previous updates
        # and we dont take those into account here when calculating the number
        # of pending parents
        if update_id == 1 and len(parent_ids) == 0:
            state = 'Ready'
            time_ready = time_msecs()
            icr['n_ready_jobs'] += 1
            icr['ready_cores_mcpu'] += cores_mcpu
            if not always_run:
                icr['n_ready_cancellable_jobs'] += 1
                icr['ready_cancellable_cores_mcpu'] += cores_mcpu
        else:
            state = 'Pending'
            time_ready = None

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
                update_id,
                state,
                json.dumps(db_spec),
                always_run,
                cores_mcpu,
                len(parent_ids),
                inst_coll_name,
                n_regions,
                regions_bits_rep,
            )
        )

        jobs_telemetry_args.append((batch_id, job_id, time_ready))

        for parent_id in parent_ids:
            job_parents_args.append((batch_id, job_id, parent_id))

        if attributes:
            for k, v in attributes.items():
                job_attributes_args.append((batch_id, job_id, k, v))

    rand_token = random.randint(0, app['n_tokens'] - 1)

    async def write_spec_to_cloud():
        if batch_format_version.has_full_spec_in_cloud():
            await spec_writer.write()

    async def insert_jobs_into_db(tx):
        try:
            try:
                await tx.execute_many(
                    '''
INSERT INTO jobs (batch_id, job_id, update_id, state, spec, always_run, cores_mcpu, n_pending_parents, inst_coll, n_regions, regions_bits_rep)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
''',
                    jobs_args,
                    query_name='insert_jobs',
                )
            except pymysql.err.IntegrityError as err:
                # 1062 ER_DUP_ENTRY https://dev.mysql.com/doc/refman/5.7/en/server-error-reference.html#error_er_dup_entry
                if err.args[0] == 1062:
                    log.info(f'bunch containing job {(batch_id, jobs_args[0][1])} already inserted')
                    return
                raise
            try:
                await tx.execute_many(
                    '''
INSERT INTO `job_parents` (batch_id, job_id, parent_id)
VALUES (%s, %s, %s);
''',
                    job_parents_args,
                    query_name='insert_job_parents',
                )
            except pymysql.err.IntegrityError as err:
                # 1062 ER_DUP_ENTRY https://dev.mysql.com/doc/refman/5.7/en/server-error-reference.html#error_er_dup_entry
                if err.args[0] == 1062:
                    raise web.HTTPBadRequest(text=f'bunch contains job with duplicated parents ({job_parents_args})')
                raise

            await tx.execute_many(
                '''
INSERT INTO `job_attributes` (batch_id, job_id, `key`, `value`)
VALUES (%s, %s, %s, %s);
''',
                job_attributes_args,
                query_name='insert_job_attributes',
            )

            await tx.execute_many(
                '''
INSERT INTO jobs_telemetry (batch_id, job_id, time_ready)
VALUES (%s, %s, %s);
''',
                jobs_telemetry_args,
                query_name='insert_jobs_telemetry',
            )

            batches_inst_coll_staging_args = [
                (
                    batch_id,
                    update_id,
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
INSERT INTO batches_inst_coll_staging (batch_id, update_id, inst_coll, token, n_jobs, n_ready_jobs, ready_cores_mcpu)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  n_jobs = n_jobs + VALUES(n_jobs),
  n_ready_jobs = n_ready_jobs + VALUES(n_ready_jobs),
  ready_cores_mcpu = ready_cores_mcpu + VALUES(ready_cores_mcpu);
''',
                batches_inst_coll_staging_args,
                query_name='insert_batches_inst_coll_staging',
            )

            batch_inst_coll_cancellable_resources_args = [
                (
                    batch_id,
                    update_id,
                    inst_coll,
                    rand_token,
                    resources['n_ready_cancellable_jobs'],
                    resources['ready_cancellable_cores_mcpu'],
                )
                for inst_coll, resources in inst_coll_resources.items()
            ]
            await tx.execute_many(
                '''
INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, inst_coll, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
VALUES (%s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  n_ready_cancellable_jobs = n_ready_cancellable_jobs + VALUES(n_ready_cancellable_jobs),
  ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + VALUES(ready_cancellable_cores_mcpu);
''',
                batch_inst_coll_cancellable_resources_args,
                query_name='insert_inst_coll_cancellable_resources',
            )

            if batch_format_version.has_full_spec_in_cloud():
                await tx.execute_update(
                    '''
INSERT INTO batch_bunches (batch_id, token, start_job_id)
VALUES (%s, %s, %s);
''',
                    (batch_id, spec_writer.token, bunch_start_job_id),
                    query_name='insert_batch_bunches',
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
@auth.rest_authenticated_users_only
@add_metadata_to_request
async def create_batch_fast(request, userdata):
    app = request.app
    db: Database = app['db']

    user = userdata['username']
    batch_and_bunch = await json_request(request)
    batch_spec = batch_and_bunch['batch']
    bunch = batch_and_bunch['bunch']
    batch_id = await _create_batch(batch_spec, userdata, db)
    update_id, _ = await _create_batch_update(batch_id, batch_spec['token'], batch_spec['n_jobs'], user, db)
    try:
        await _create_jobs(userdata, bunch, batch_id, update_id, app)
    except web.HTTPBadRequest as e:
        if f'update {update_id} is already committed' == e.reason:
            return json_response({'id': batch_id})
        raise
    await _commit_update(app, batch_id, update_id, user, db)
    request['batch_telemetry']['batch_id'] = str(batch_id)
    return json_response({'id': batch_id})


@routes.post('/api/v1alpha/batches/create')
@auth.rest_authenticated_users_only
@add_metadata_to_request
async def create_batch(request, userdata):
    app = request.app
    db: Database = app['db']

    batch_spec = await json_request(request)
    id = await _create_batch(batch_spec, userdata, db)
    n_jobs = batch_spec['n_jobs']
    if n_jobs > 0:
        update_id, _ = await _create_batch_update(
            id, batch_spec['token'], batch_spec['n_jobs'], userdata['username'], db
        )
    else:
        update_id = None
    request['batch_telemetry']['batch_id'] = str(id)
    return json_response({'id': id, 'update_id': update_id})


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
WHERE billing_projects.name_cs = %s AND user_cs = %s
LOCK IN SHARE MODE''',
            (billing_project, user),
        )

        if bp is None:
            raise web.HTTPForbidden(reason=f'Unknown Hail Batch billing project {billing_project}.')
        if bp['status'] in {'closed', 'deleted'}:
            raise web.HTTPForbidden(reason=f'Billing project {billing_project} is closed or deleted.')

        bp_cost_record = await tx.execute_and_fetchone(
            '''
SELECT COALESCE(SUM(t.`usage` * rate), 0) AS cost
FROM (
  SELECT resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM aggregated_billing_project_user_resources_v2
  WHERE billing_project = %s
  GROUP BY resource_id
) AS t
LEFT JOIN resources on resources.resource_id = t.resource_id;
''',
            (billing_project,),
        )
        limit = bp['limit']
        accrued_cost = bp_cost_record['cost']
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
INSERT INTO batches (userdata, user, billing_project, attributes, callback, n_jobs, time_created, time_completed, token, state, format_version, cancel_after_n_failures)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
''',
            (
                json.dumps(userdata),
                user,
                billing_project,
                json.dumps(attributes),
                batch_spec.get('callback'),
                0,
                now,
                now,
                token,
                'complete',
                BATCH_FORMAT_VERSION,
                batch_spec.get('cancel_after_n_failures'),
            ),
            query_name='insert_batches',
        )
        await tx.execute_insertone(
            '''
INSERT INTO batches_n_jobs_in_complete_states (id) VALUES (%s);
''',
            (id,),
            query_name='insert_batches_n_jobs_in_complete_states',
        )

        if attributes:
            await tx.execute_many(
                '''
INSERT INTO `batch_attributes` (batch_id, `key`, `value`)
VALUES (%s, %s, %s)
''',
                [(id, k, v) for k, v in attributes.items()],
                query_name='insert_batch_attributes',
            )
        return id

    return await insert()  # pylint: disable=no-value-for-parameter


@routes.post('/api/v1alpha/batches/{batch_id}/update-fast')
@auth.rest_authenticated_users_only
@add_metadata_to_request
async def update_batch_fast(request, userdata):
    app = request.app
    db: Database = app['db']

    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    update_and_bunch = await json_request(request)
    update_spec = update_and_bunch['update']
    bunch = update_and_bunch['bunch']

    try:
        validate_batch_update(update_spec)
    except ValidationError as e:
        raise web.HTTPBadRequest(reason=e.reason)

    update_id, start_job_id = await _create_batch_update(
        batch_id, update_spec['token'], update_spec['n_jobs'], user, db
    )

    try:
        await _create_jobs(userdata, bunch, batch_id, update_id, app)
    except web.HTTPBadRequest as e:
        if f'update {update_id} is already committed' == e.reason:
            return json_response({'update_id': update_id, 'start_job_id': start_job_id})
        raise
    await _commit_update(app, batch_id, update_id, user, db)
    request['batch_telemetry']['batch_id'] = str(batch_id)
    return json_response({'update_id': update_id, 'start_job_id': start_job_id})


@routes.post('/api/v1alpha/batches/{batch_id}/updates/create')
@auth.rest_authenticated_users_only
@add_metadata_to_request
async def create_update(request, userdata):
    app = request.app
    db: Database = app['db']

    if app['frozen']:
        log.info('ignoring batch create request; batch is frozen')
        raise web.HTTPServiceUnavailable()

    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    update_spec = await json_request(request)

    try:
        validate_batch_update(update_spec)
    except ValidationError as e:
        raise web.HTTPBadRequest(reason=e.reason)

    update_id, _ = await _create_batch_update(batch_id, update_spec['token'], update_spec['n_jobs'], user, db)
    return json_response({'update_id': update_id})


async def _create_batch_update(
    batch_id: int, update_token: str, n_jobs: int, user: str, db: Database
) -> Tuple[int, int]:
    @transaction(db)
    async def update(tx: Transaction):
        assert n_jobs > 0
        record = await tx.execute_and_fetchone(
            '''
SELECT update_id, start_job_id FROM batch_updates
WHERE batch_id = %s AND token = %s;
''',
            (batch_id, update_token),
        )

        if record:
            return record['update_id'], record['start_job_id']

        # We use FOR UPDATE so that we serialize batch update insertions
        # This is necessary to reserve job id ranges.
        # We don't allow updates to batches that have been cancelled
        # but do allow updates to batches with jobs that have been cancelled.
        record = await tx.execute_and_fetchone(
            '''
SELECT batches_cancelled.id IS NOT NULL AS cancelled
FROM batches
LEFT JOIN batches_cancelled ON batches.id = batches_cancelled.id
WHERE batches.id = %s AND user = %s AND NOT deleted
FOR UPDATE;
''',
            (batch_id, user),
        )
        if not record:
            raise web.HTTPNotFound()
        if record['cancelled']:
            raise web.HTTPBadRequest(reason='Cannot submit new jobs to a cancelled batch')

        now = time_msecs()

        record = await tx.execute_and_fetchone(
            '''
SELECT update_id, start_job_id, n_jobs FROM batch_updates
WHERE batch_id = %s
ORDER BY update_id DESC
LIMIT 1;
''',
            (batch_id,),
        )
        if record:
            update_id = int(record['update_id']) + 1
            update_start_job_id = int(record['start_job_id']) + int(record['n_jobs'])
        else:
            update_id = 1
            update_start_job_id = 1

        await tx.execute_insertone(
            '''
INSERT INTO batch_updates
(batch_id, update_id, token, start_job_id, n_jobs, committed, time_created)
VALUES (%s, %s, %s, %s, %s, %s, %s);
''',
            (batch_id, update_id, update_token, update_start_job_id, n_jobs, False, now),
            query_name='insert_batch_update',
        )

        return (update_id, update_start_job_id)

    return await update()  # pylint: disable=no-value-for-parameter


async def _get_batch(app, batch_id):
    db: Database = app['db']

    record = await db.select_and_fetchone(
        '''
WITH base_t AS (
SELECT batches.*,
  batches_cancelled.id IS NOT NULL AS cancelled,
  batches_n_jobs_in_complete_states.n_completed,
  batches_n_jobs_in_complete_states.n_succeeded,
  batches_n_jobs_in_complete_states.n_failed,
  batches_n_jobs_in_complete_states.n_cancelled
FROM batches
LEFT JOIN batches_n_jobs_in_complete_states
       ON batches.id = batches_n_jobs_in_complete_states.id
LEFT JOIN batches_cancelled
       ON batches.id = batches_cancelled.id
WHERE batches.id = %s AND NOT deleted
)
SELECT base_t.*, COALESCE(SUM(`usage` * rate), 0) AS cost
FROM base_t
LEFT JOIN (
  SELECT aggregated_batch_resources_v2.batch_id, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM base_t
  LEFT JOIN aggregated_batch_resources_v2 ON base_t.id = aggregated_batch_resources_v2.batch_id
  GROUP BY aggregated_batch_resources_v2.batch_id, aggregated_batch_resources_v2.resource_id
) AS usage_t ON base_t.id = usage_t.batch_id
LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
GROUP BY base_t.id;
''',
        (batch_id,),
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
@add_metadata_to_request
async def get_batch(request, userdata, batch_id):  # pylint: disable=unused-argument
    return json_response(await _get_batch(request.app, batch_id))


@routes.patch('/api/v1alpha/batches/{batch_id}/cancel')
@rest_billing_project_users_only
@add_metadata_to_request
async def cancel_batch(request, userdata, batch_id):  # pylint: disable=unused-argument
    await _handle_api_error(_cancel_batch, request.app, batch_id)
    return web.Response()


# deprecated
@routes.patch('/api/v1alpha/batches/{batch_id}/close')
@auth.rest_authenticated_users_only
@add_metadata_to_request
async def close_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']

    app = request.app
    db: Database = app['db']

    record = await db.select_and_fetchone(
        '''
SELECT batches_cancelled.id IS NOT NULL AS cancelled
FROM batches
LEFT JOIN batches_cancelled ON batches.id = batches_cancelled.id
WHERE user = %s AND batches.id = %s AND NOT deleted;
''',
        (user, batch_id),
    )
    if not record:
        raise web.HTTPNotFound()
    if record['cancelled']:
        raise web.HTTPBadRequest(reason='Cannot close a previously cancelled batch.')

    record = await db.select_and_fetchone(
        '''
SELECT 1 FROM batch_updates
WHERE batch_id = %s AND update_id = 1;
''',
        (batch_id,),
    )
    if record:
        await _commit_update(app, batch_id, 1, user, db)
    return web.Response()


@routes.patch('/api/v1alpha/batches/{batch_id}/updates/{update_id}/commit')
@auth.rest_authenticated_users_only
@add_metadata_to_request
async def commit_update(request: web.Request, userdata):
    app = request.app
    db: Database = app['db']
    user = userdata['username']

    batch_id = int(request.match_info['batch_id'])
    update_id = int(request.match_info['update_id'])

    record = await db.select_and_fetchone(
        '''
SELECT start_job_id, batches_cancelled.id IS NOT NULL AS cancelled
FROM batches
LEFT JOIN batch_updates ON batches.id = batch_updates.batch_id
LEFT JOIN batches_cancelled ON batches.id = batches_cancelled.id
WHERE user = %s AND batches.id = %s AND batch_updates.update_id = %s AND NOT deleted;
''',
        (user, batch_id, update_id),
    )
    if not record:
        raise web.HTTPNotFound()
    if record['cancelled']:
        raise web.HTTPBadRequest(reason='Cannot commit an update to a cancelled batch')

    await _commit_update(app, batch_id, update_id, user, db)
    return json_response({'start_job_id': record['start_job_id']})


async def _commit_update(app: web.Application, batch_id: int, update_id: int, user: str, db: Database):
    client_session: httpx.ClientSession = app['client_session']

    try:
        now = time_msecs()
        await db.check_call_procedure(
            'CALL commit_batch_update(%s, %s, %s);', (batch_id, update_id, now), 'commit_batch_update'
        )
    except CallError as e:
        # 2: wrong number of jobs
        if e.rv['rc'] == 2:
            expected_n_jobs = e.rv['expected_n_jobs']
            actual_n_jobs = e.rv['actual_n_jobs']
            raise web.HTTPBadRequest(reason=f'wrong number of jobs: expected {expected_n_jobs}, actual {actual_n_jobs}')
        raise

    app['task_manager'].ensure_future(
        retry_transient_errors(
            client_session.patch,
            deploy_config.url('batch-driver', f'/api/v1alpha/batches/{user}/{batch_id}/update'),
            headers=app['batch_headers'],
        )
    )


@routes.delete('/api/v1alpha/batches/{batch_id}')
@rest_billing_project_users_only
@add_metadata_to_request
async def delete_batch(request, userdata, batch_id):  # pylint: disable=unused-argument
    await _delete_batch(request.app, batch_id)
    return web.Response()


@routes.get('/batches/{batch_id}')
@web_billing_project_users_only()
@catch_ui_error_in_dev
async def ui_batch(request, userdata, batch_id):
    app = request.app
    batch = await _get_batch(app, batch_id)

    q = request.query.get('q', '')
    last_job_id = cast_query_param_to_int(request.query.get('last_job_id'))

    try:
        jobs, last_job_id = await _query_batch_jobs(request, batch_id, CURRENT_QUERY_VERSION, q, last_job_id)
    except QueryError as e:
        session = await aiohttp_session.get_session(request)
        set_message(session, e.message, 'error')
        jobs = []
        last_job_id = None

    for j in jobs:
        j['duration'] = humanize_timedelta_msecs(j['duration'])
        j['cost'] = cost_str(j['cost'])
    batch['jobs'] = jobs

    batch['cost'] = cost_str(batch['cost'])

    page_context = {
        'batch': batch,
        'q': q,
        'last_job_id': last_job_id,
    }
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
    try:
        await _handle_ui_error(session, _cancel_batch, request.app, batch_id)
        set_message(session, f'Batch {batch_id} cancelled.', 'info')
    finally:
        location = request.app.router['batches'].url_for().with_query(params)
        return web.HTTPFound(location=location)  # pylint: disable=lost-exception


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
@auth.web_authenticated_users_only()
@catch_ui_error_in_dev
async def ui_batches(request: web.Request, userdata: dict):
    session = await aiohttp_session.get_session(request)
    user = userdata['username']
    q = request.query.get('q', f'user:{user}')
    last_batch_id = cast_query_param_to_int(request.query.get('last_batch_id'))
    try:
        result = await _handle_ui_error(session, _query_batches, request, user, q, CURRENT_QUERY_VERSION, last_batch_id)
        assert result is not None
        batches, last_batch_id = result
    except asyncio.CancelledError:
        raise
    except Exception:
        batches = []

    for batch in batches:
        batch['cost'] = cost_str(batch['cost'])

    page_context = {'batches': batches, 'q': q, 'last_batch_id': last_batch_id}
    return await render_template('batch', request, userdata, 'batches.html', page_context)


async def _get_job(app, batch_id, job_id):
    db: Database = app['db']

    record = await db.select_and_fetchone(
        '''
WITH base_t AS (
SELECT jobs.*, user, billing_project, ip_address, format_version, t.attempt_id AS last_cancelled_attempt_id
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
) AS t ON jobs.batch_id = t.batch_id AND jobs.job_id = t.job_id
LEFT JOIN batch_updates
  ON jobs.batch_id = batch_updates.batch_id AND jobs.update_id = batch_updates.update_id
WHERE jobs.batch_id = %s AND NOT deleted AND jobs.job_id = %s AND batch_updates.committed
)
SELECT base_t.*, COALESCE(SUM(`usage` * rate), 0) AS cost
FROM base_t
LEFT JOIN (
  SELECT aggregated_job_resources_v2.batch_id,
    aggregated_job_resources_v2.job_id,
    aggregated_job_resources_v2.resource_id,
    CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM base_t
  LEFT JOIN aggregated_job_resources_v2
    ON aggregated_job_resources_v2.batch_id = base_t.batch_id AND
       aggregated_job_resources_v2.job_id = base_t.job_id
  GROUP BY aggregated_job_resources_v2.batch_id, aggregated_job_resources_v2.job_id, aggregated_job_resources_v2.resource_id
) AS usage_t ON usage_t.batch_id = base_t.batch_id AND usage_t.job_id = base_t.job_id
LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
GROUP BY base_t.batch_id, base_t.job_id, base_t.last_cancelled_attempt_id;
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

    attempts.sort(key=lambda x: x['start_time'] or x['end_time'])

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
    return json_response(attempts)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}')
@rest_billing_project_users_only
@add_metadata_to_request
async def get_job(request, userdata, batch_id):  # pylint: disable=unused-argument
    job_id = int(request.match_info['job_id'])
    status = await _get_job(request.app, batch_id, job_id)
    return json_response(status)


def plot_job_durations(container_statuses: dict, batch_id: int, job_id: int):
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

    if not data:
        return None

    df = pd.DataFrame(data)

    fig = px.timeline(
        df,
        x_start='Start',
        x_end='Finish',
        y='Step',
        color='Task',
        hover_data=['Step'],
        color_discrete_sequence=px.colors.qualitative.Prism,
        category_orders={
            'Step': ['input', 'main', 'output'],
            'Task': [
                'pulling',
                'setting up overlay',
                'setting up network',
                'running',
                'uploading_log',
                'uploading_resource_usage',
            ],
        },
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def plot_resource_usage(
    resource_usage: Optional[Dict[str, Optional[pd.DataFrame]]],
    memory_limit_bytes: Optional[int],
    io_storage_limit_bytes: Optional[int],
    non_io_storage_limit_bytes: Optional[int],
) -> Optional[str]:
    if resource_usage is None:
        return None

    if io_storage_limit_bytes is not None:
        if io_storage_limit_bytes == 0:
            io_storage_title = ''
        else:
            io_storage_title = (
                f'Storage (Mounted Drive at /io) - {humanize.naturalsize(io_storage_limit_bytes, binary=False)} max'
            )
    else:
        io_storage_title = 'Storage (Mounted Drive at /io)'

    if non_io_storage_limit_bytes is not None:
        if io_storage_limit_bytes != 0:
            non_io_storage_title = (
                f'Storage (Container Overlay) - {humanize.naturalsize(non_io_storage_limit_bytes, binary=False)} max'
            )
        else:
            non_io_storage_title = f'Storage - {humanize.naturalsize(non_io_storage_limit_bytes, binary=False)} max'
    else:
        non_io_storage_title = 'Storage (Container Overlay)'

    if memory_limit_bytes is not None:
        memory_title = f'Memory - {humanize.naturalsize(memory_limit_bytes, binary=False)} max'
    else:
        memory_title = 'Memory'

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            'CPU Usage',
            memory_title,
            'Network Download Bandwidth (MB/sec)',
            'Network Upload Bandwidth (MB/sec)',
            non_io_storage_title,
            io_storage_title,
        ),
    )
    fig.update_layout(height=800, width=800)

    colors = {'input': 'red', 'main': 'green', 'output': 'blue'}

    max_cpu_value = 1
    max_memory_value = 1024 * 1024
    max_download_network_bandwidth_value = 500
    max_upload_network_bandwidth_value = 500
    max_io_storage_value = 1024 * 1024 * 1024
    max_non_io_storage_value = 1024 * 1024 * 1024
    n_total_rows = 0

    for container_name, df in resource_usage.items():
        if df is None:
            continue

        n_rows = df.shape[0]
        n_total_rows += n_rows

        time_df = pd.to_datetime(df['time_msecs'], unit='ms')
        mem_df = df['memory_in_bytes']
        cpu_df = df['cpu_usage']

        def get_df(df, colname):
            if colname not in df:
                df[colname] = ResourceUsageMonitor.missing_value
            return df[colname]

        network_download_df = get_df(df, 'network_bandwidth_download_in_bytes_per_second')
        network_upload_df = get_df(df, 'network_bandwidth_upload_in_bytes_per_second')
        non_io_storage_df = get_df(df, 'non_io_storage_in_bytes')
        io_storage_df = get_df(df, 'io_storage_in_bytes')

        if n_rows != 0:
            max_cpu_value = max(max_cpu_value, cpu_df.max())
            max_memory_value = max(max_memory_value, mem_df.max())
            max_download_network_bandwidth_value = max(max_download_network_bandwidth_value, network_download_df.max())
            max_upload_network_bandwidth_value = max(max_upload_network_bandwidth_value, network_upload_df.max())
            max_io_storage_value = max(max_io_storage_value, io_storage_df.max())
            max_non_io_storage_value = max(max_non_io_storage_value, non_io_storage_df.max())

        def add_trace(time, measurement, row, col, container_name, show_legend):
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=measurement,
                    showlegend=show_legend,
                    legendgroup=container_name,
                    name=container_name,
                    mode='markers+lines',
                    line={'color': colors[container_name]},
                ),
                row=row,
                col=col,
            )

        add_trace(time_df, cpu_df, 1, 1, container_name, True)
        add_trace(time_df, mem_df, 1, 2, container_name, False)
        add_trace(time_df, network_download_df, 2, 1, container_name, False)
        add_trace(time_df, network_upload_df, 2, 2, container_name, False)
        add_trace(time_df, non_io_storage_df, 3, 1, container_name, False)
        if io_storage_limit_bytes != 0:
            add_trace(time_df, io_storage_df, 3, 2, container_name, False)

        limit_props = {'color': 'black', 'width': 2}
        if memory_limit_bytes is not None:
            fig.add_hline(memory_limit_bytes, row=1, col=2, line=limit_props)
        if non_io_storage_limit_bytes is not None:
            fig.add_hline(non_io_storage_limit_bytes, row=3, col=1, line=limit_props)
        if io_storage_limit_bytes is not None:
            fig.add_hline(io_storage_limit_bytes, row=3, col=2, line=limit_props)

    fig.update_layout(
        showlegend=True,
        yaxis1_tickformat='%',
        yaxis2_tickformat='s',
        yaxis5_tickformat='s',
        yaxis6_tickformat='s',
        yaxis1_range=[0, 1.25 * max_cpu_value],
        yaxis2_range=[0, 1.25 * max_memory_value],
        yaxis3_range=[0, 1.25 * max_download_network_bandwidth_value],
        yaxis4_range=[0, 1.25 * max_upload_network_bandwidth_value],
        yaxis5_range=[0, 1.25 * max_non_io_storage_value],
        yaxis6_range=[0, 1.25 * max_io_storage_value],
    )

    if n_total_rows == 0:
        return None

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@routes.get('/batches/{batch_id}/jobs/{job_id}/jvm_profile')
@web_billing_project_users_only()
@catch_ui_error_in_dev
async def ui_get_jvm_profile(request, userdata, batch_id):  # pylint: disable=unused-argument
    app = request.app
    job_id = int(request.match_info['job_id'])
    profile = await _get_jvm_profile(app, batch_id, job_id)
    if profile is None:
        raise web.HTTPNotFound()
    return web.Response(text=profile, content_type='text/html')


@routes.get('/batches/{batch_id}/jobs/{job_id}')
@web_billing_project_users_only()
@catch_ui_error_in_dev
async def ui_get_job(request, userdata, batch_id):
    app = request.app
    job_id = int(request.match_info['job_id'])

    job, attempts, job_log_bytes, resource_usage = await asyncio.gather(
        _get_job(app, batch_id, job_id),
        _get_attempts(app, batch_id, job_id),
        _get_job_log(app, batch_id, job_id),
        _get_job_resource_usage(app, batch_id, job_id),
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
                'uploading_resource_usage': dictfix.NoneOr({'duration': dictfix.NoneOr(Number)}),
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
    step_errors = {step: status['error'] for step, status in container_statuses.items() if status is not None}

    for status in container_statuses.values():
        # backwards compatibility
        if status and status['short_error'] is None and status['container_status']['out_of_memory']:
            status['short_error'] = 'out of memory'

    has_jvm_profile = False

    job_specification = job['spec']
    if job_specification:
        if 'process' in job_specification:
            process_specification = job_specification['process']
            process_type = process_specification['type']
            assert process_specification['type'] in {'docker', 'jvm'}
            job_specification['image'] = process_specification['image'] if process_type == 'docker' else '[jvm]'
            job_specification['command'] = process_specification['command']
            has_jvm_profile = job['state'] in complete_states and process_specification.get('profile', False)

        job_specification = dictfix.dictfix(
            job_specification, dictfix.NoneOr({'image': str, 'command': list, 'resources': {}, 'env': list})
        )

    io_storage_limit_bytes = None
    non_io_storage_limit_bytes = None
    memory_limit_bytes = None

    resources = job_specification['resources']
    if 'memory_bytes' in resources:
        memory_limit_bytes = resources['memory_bytes']
        resources['actual_memory'] = humanize.naturalsize(memory_limit_bytes, binary=True)
        del resources['memory_bytes']
    if 'storage_gib' in resources:
        io_storage_limit_bytes = resources['storage_gib'] * 1024**3
        resources['actual_storage'] = humanize.naturalsize(io_storage_limit_bytes, binary=True)
        del resources['storage_gib']
    if 'cores_mcpu' in resources:
        cores = resources['cores_mcpu'] / 1000
        non_io_storage_limit_gb = min(cores * RESERVED_STORAGE_GB_PER_CORE, RESERVED_STORAGE_GB_PER_CORE)
        non_io_storage_limit_bytes = int(non_io_storage_limit_gb * 1024**3 + 1)
        resources['actual_cpu'] = cores
        del resources['cores_mcpu']

    # Not all logs will be proper utf-8 but we attempt to show them as
    # str or else Jinja will present them surrounded by b''
    job_log_strings_or_bytes = {}
    for container, log in job_log_bytes.items():
        try:
            job_log_strings_or_bytes[container] = log.decode('utf-8') if log is not None else None
        except UnicodeDecodeError:
            job_log_strings_or_bytes[container] = log

    page_context = {
        'batch_id': batch_id,
        'job_id': job_id,
        'job': job,
        'job_log': job_log_strings_or_bytes,
        'attempts': attempts,
        'container_statuses': container_statuses,
        'job_specification': job_specification,
        'job_status_str': json.dumps(job, indent=2),
        'step_errors': step_errors,
        'error': job_status.get('error'),
        'plot_job_durations': plot_job_durations(container_statuses, batch_id, job_id),
        'plot_resource_usage': plot_resource_usage(
            resource_usage, memory_limit_bytes, io_storage_limit_bytes, non_io_storage_limit_bytes
        ),
        'has_jvm_profile': has_jvm_profile,
    }

    return await render_template('batch', request, userdata, 'job.html', page_context)


# This should really be the exact same thing as the REST endpoint
@routes.get('/batches/{batch_id}/jobs/{job_id}/log/{container}')
@web_billing_project_users_only()
@catch_ui_error_in_dev
async def ui_get_job_log(request, userdata, batch_id):  # pylint: disable=unused-argument
    return await get_job_container_log(request, batch_id)


@routes.get('/billing_limits')
@auth.web_authenticated_users_only()
@catch_ui_error_in_dev
async def ui_get_billing_limits(request, userdata):
    app = request.app
    db: Database = app['db']

    if not userdata['is_developer']:
        user = userdata['username']
    else:
        user = None

    billing_projects = await query_billing_projects_with_cost(db, user=user)

    open_billing_projects = [bp for bp in billing_projects if bp['status'] == 'open']
    closed_billing_projects = [bp for bp in billing_projects if bp['status'] == 'closed']

    page_context = {
        'open_billing_projects': open_billing_projects,
        'closed_billing_projects': closed_billing_projects,
        'is_developer': userdata['is_developer'],
    }
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
WHERE billing_projects.name_cs = %s AND billing_projects.`status` != 'deleted'
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
UPDATE billing_projects SET `limit` = %s WHERE name_cs = %s;
''',
            (limit, billing_project),
        )

    await insert()  # pylint: disable=no-value-for-parameter


@routes.post('/api/v1alpha/billing_limits/{billing_project}/edit')
@rest_authenticated_developers_or_auth_only
async def post_edit_billing_limits(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    data = await json_request(request)
    limit = data['limit']
    await _handle_api_error(_edit_billing_limit, db, billing_project, limit)
    return json_response({'billing_project': billing_project, 'limit': limit})


@routes.post('/billing_limits/{billing_project}/edit')
@check_csrf_token
@auth.web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_edit_billing_limits_ui(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    post = await request.post()
    limit = post['limit']
    session = await aiohttp_session.get_session(request)
    try:
        await _handle_ui_error(session, _edit_billing_limit, db, billing_project, limit)
        set_message(session, f'Modified limit {limit} for billing project {billing_project}.', 'info')
    finally:
        return web.HTTPFound(deploy_config.external_url('batch', '/billing_limits'))  # pylint: disable=lost-exception


async def _query_billing(request, user=None):
    db: Database = request.app['db']

    date_format = '%m/%d/%Y'

    default_start = datetime.datetime.now().replace(day=1)
    default_start_str = datetime.datetime.strftime(default_start, date_format)

    default_end = None

    async def parse_error(msg):
        session = await aiohttp_session.get_session(request)
        set_message(session, msg, 'error')
        return ([], default_start_str, default_end)

    start_query = request.query.get('start', default_start_str)
    try:
        start = datetime.datetime.strptime(start_query, date_format)
    except ValueError:
        return await parse_error(f"Invalid value for start '{start_query}'; must be in the format of MM/DD/YYYY.")

    end_query = request.query.get('end', default_end)
    try:
        if end_query is not None and end_query != '':
            end = datetime.datetime.strptime(end_query, date_format)
        else:
            end = None
    except ValueError:
        return await parse_error(f"Invalid value for end '{end_query}'; must be in the format of MM/DD/YYYY.")

    if end is not None and start > end:
        return await parse_error('Invalid search; start must be earlier than end.')

    where_conditions = [
        "billing_projects.`status` != 'deleted'",
        "billing_date >= %s",
    ]
    where_args = [start]

    if end is not None:
        where_conditions.append("billing_date <= %s")
        where_args.append(end)

    if user is not None:
        where_conditions.append("`user` = %s")
        where_args.append(user)

    sql = f'''
SELECT
  billing_project,
  `user`,
  COALESCE(SUM(`usage` * rate), 0) AS cost
FROM (
  SELECT billing_project, `user`, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM aggregated_billing_project_user_resources_by_date_v2
  LEFT JOIN billing_projects ON billing_projects.name = aggregated_billing_project_user_resources_by_date_v2.billing_project
  WHERE {' AND '.join(where_conditions)}
  GROUP BY billing_project, `user`, resource_id
) AS t
LEFT JOIN resources ON resources.resource_id = t.resource_id
GROUP BY billing_project, `user`;
'''

    sql_args = where_args

    billing = [record async for record in db.select_and_fetchall(sql, sql_args)]

    return (billing, start_query, end_query)


@routes.get('/billing')
@auth.web_authenticated_users_only()
@catch_ui_error_in_dev
async def ui_get_billing(request, userdata):
    is_developer = userdata['is_developer'] == 1
    user = userdata['username'] if not is_developer else None
    billing, start, end = await _query_billing(request, user=user)

    billing_by_user: Dict[str, int] = {}
    billing_by_project: Dict[str, int] = {}
    for record in billing:
        billing_project = record['billing_project']
        user = record['user']
        cost = record['cost']
        billing_by_user[user] = billing_by_user.get(user, 0) + cost
        billing_by_project[billing_project] = billing_by_project.get(billing_project, 0) + cost

    billing_by_project_list = [
        {'billing_project': billing_project, 'cost': cost_str(cost) or '$0'}
        for billing_project, cost in billing_by_project.items()
    ]
    billing_by_project_list.sort(key=lambda record: record['billing_project'])

    billing_by_user_list = [{'user': user, 'cost': cost_str(cost) or '$0'} for user, cost in billing_by_user.items()]
    billing_by_user_list.sort(key=lambda record: record['user'])

    billing_by_project_user = [
        {'billing_project': record['billing_project'], 'user': record['user'], 'cost': cost_str(record['cost']) or '$0'}
        for record in billing
    ]
    billing_by_project_user.sort(key=lambda record: (record['billing_project'], record['user']))

    total_cost = cost_str(sum(record['cost'] for record in billing))

    page_context = {
        'billing_by_project': billing_by_project_list,
        'billing_by_user': billing_by_user_list,
        'billing_by_project_user': billing_by_project_user,
        'start': start,
        'end': end,
        'is_developer': is_developer,
        'user': userdata['username'],
        'total_cost': total_cost,
    }
    return await render_template('batch', request, userdata, 'billing.html', page_context)


@routes.get('/billing_projects')
@auth.web_authenticated_developers_only()
@catch_ui_error_in_dev
async def ui_get_billing_projects(request, userdata):
    db: Database = request.app['db']
    billing_projects = await query_billing_projects_without_cost(db)
    page_context = {
        'billing_projects': [{**p, 'size': len(p['users'])} for p in billing_projects if p['status'] == 'open'],
        'closed_projects': [p for p in billing_projects if p['status'] == 'closed'],
    }
    return await render_template('batch', request, userdata, 'billing_projects.html', page_context)


@routes.get('/api/v1alpha/billing_projects')
@auth.rest_authenticated_users_only
async def get_billing_projects(request, userdata):
    db: Database = request.app['db']

    if not userdata['is_developer'] and userdata['username'] != 'auth':
        user = userdata['username']
    else:
        user = None

    billing_projects = await query_billing_projects_with_cost(db, user=user)
    return json_response(billing_projects)


@routes.get('/api/v1alpha/billing_projects/{billing_project}')
@auth.rest_authenticated_users_only
async def get_billing_project(request, userdata):
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']

    if not userdata['is_developer'] and userdata['username'] != 'auth':
        user = userdata['username']
    else:
        user = None

    billing_projects = await query_billing_projects_with_cost(db, user=user, billing_project=billing_project)

    if not billing_projects:
        raise web.HTTPForbidden(reason=f'Unknown Hail Batch billing project {billing_project}.')

    assert len(billing_projects) == 1
    return json_response(billing_projects[0])


async def _remove_user_from_billing_project(db, billing_project, user):
    @transaction(db)
    async def delete(tx):
        row = await tx.execute_and_fetchone(
            '''
SELECT billing_projects.name_cs as billing_project,
  billing_projects.`status` as `status`,
  `user`
FROM billing_projects
LEFT JOIN (
  SELECT billing_project_users.* FROM billing_project_users
  LEFT JOIN billing_projects ON billing_projects.name = billing_project_users.billing_project
  WHERE billing_projects.name_cs = %s AND user_cs = %s
  FOR UPDATE
) AS t ON billing_projects.name = t.billing_project
WHERE billing_projects.name_cs = %s;
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
DELETE billing_project_users FROM billing_project_users
LEFT JOIN billing_projects ON billing_projects.name = billing_project_users.billing_project
WHERE billing_projects.name_cs = %s AND user_cs = %s;
''',
            (billing_project, user),
        )

    await delete()  # pylint: disable=no-value-for-parameter


@routes.post('/billing_projects/{billing_project}/users/{user}/remove')
@check_csrf_token
@auth.web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_billing_projects_remove_user(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    user = request.match_info['user']

    session = await aiohttp_session.get_session(request)
    try:
        await _handle_ui_error(session, _remove_user_from_billing_project, db, billing_project, user)
        set_message(session, f'Removed user {user} from billing project {billing_project}.', 'info')
    finally:
        return web.HTTPFound(deploy_config.external_url('batch', '/billing_projects'))  # pylint: disable=lost-exception


@routes.post('/api/v1alpha/billing_projects/{billing_project}/users/{user}/remove')
@rest_authenticated_developers_or_auth_only
async def api_get_billing_projects_remove_user(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    user = request.match_info['user']
    await _handle_api_error(_remove_user_from_billing_project, db, billing_project, user)
    return json_response({'billing_project': billing_project, 'user': user})


async def _add_user_to_billing_project(db, billing_project, user):
    @transaction(db)
    async def insert(tx):
        # we want to be case-insensitive here to avoid duplicates with existing records
        row = await tx.execute_and_fetchone(
            '''
SELECT billing_projects.name as billing_project,
    billing_projects.`status` as `status`,
    user
FROM billing_projects
LEFT JOIN (
  SELECT *
  FROM billing_project_users
  LEFT JOIN billing_projects ON billing_projects.name = billing_project_users.billing_project
  WHERE billing_projects.name_cs = %s AND user = %s
  FOR UPDATE
) AS t
ON billing_projects.name = t.billing_project
WHERE billing_projects.name_cs = %s AND billing_projects.`status` != 'deleted' LOCK IN SHARE MODE;
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
INSERT INTO billing_project_users(billing_project, user, user_cs)
VALUES (%s, %s, %s);
        ''',
            (billing_project, user, user),
        )

    await insert()  # pylint: disable=no-value-for-parameter


@routes.post('/billing_projects/{billing_project}/users/add')
@check_csrf_token
@auth.web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_billing_projects_add_user(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    post = await request.post()
    user = post['user']
    billing_project = request.match_info['billing_project']

    session = await aiohttp_session.get_session(request)

    try:
        await _handle_ui_error(session, _add_user_to_billing_project, db, billing_project, user)
        set_message(session, f'Added user {user} to billing project {billing_project}.', 'info')
    finally:
        return web.HTTPFound(deploy_config.external_url('batch', '/billing_projects'))  # pylint: disable=lost-exception


@routes.post('/api/v1alpha/billing_projects/{billing_project}/users/{user}/add')
@rest_authenticated_developers_or_auth_only
async def api_billing_projects_add_user(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    user = request.match_info['user']
    billing_project = request.match_info['billing_project']

    await _handle_api_error(_add_user_to_billing_project, db, billing_project, user)
    return json_response({'billing_project': billing_project, 'user': user})


async def _create_billing_project(db, billing_project):
    @transaction(db)
    async def insert(tx):
        # we want to avoid having billing projects with different cases but the same name
        row = await tx.execute_and_fetchone(
            '''
SELECT name_cs, `status`
FROM billing_projects
WHERE name = %s
FOR UPDATE;
''',
            (billing_project),
        )
        if row is not None:
            billing_project_cs = row['name_cs']
            raise BatchOperationAlreadyCompletedError(f'Billing project {billing_project_cs} already exists.', 'info')

        await tx.execute_insertone(
            '''
INSERT INTO billing_projects(name, name_cs)
VALUES (%s, %s);
''',
            (billing_project, billing_project),
        )

    await insert()  # pylint: disable=no-value-for-parameter


@routes.post('/billing_projects/create')
@check_csrf_token
@auth.web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_create_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    post = await request.post()
    billing_project = post['billing_project']

    session = await aiohttp_session.get_session(request)
    try:
        await _handle_ui_error(session, _create_billing_project, db, billing_project)
        set_message(session, f'Added billing project {billing_project}.', 'info')
    finally:
        return web.HTTPFound(deploy_config.external_url('batch', '/billing_projects'))  # pylint: disable=lost-exception


@routes.post('/api/v1alpha/billing_projects/{billing_project}/create')
@rest_authenticated_developers_or_auth_only
async def api_get_create_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    await _handle_api_error(_create_billing_project, db, billing_project)
    return json_response(billing_project)


async def _close_billing_project(db, billing_project):
    @transaction(db)
    async def close_project(tx):
        row = await tx.execute_and_fetchone(
            '''
SELECT name_cs, `status`, batches.id as batch_id
FROM billing_projects
LEFT JOIN batches
ON billing_projects.name = batches.billing_project
AND billing_projects.`status` != 'deleted'
AND batches.time_completed IS NULL
AND NOT batches.deleted
WHERE name_cs = %s
LIMIT 1
FOR UPDATE;
    ''',
            (billing_project,),
        )
        if not row:
            raise NonExistentBillingProjectError(billing_project)
        assert row['name_cs'] == billing_project
        if row['status'] == 'closed':
            raise BatchOperationAlreadyCompletedError(
                f'Billing project {billing_project} is already closed or deleted.', 'info'
            )
        if row['batch_id'] is not None:
            raise BatchUserError(f'Billing project {billing_project} has running batches.', 'error')

        await tx.execute_update(
            "UPDATE billing_projects SET `status` = 'closed' WHERE name_cs = %s;", (billing_project,)
        )

    await close_project()  # pylint: disable=no-value-for-parameter


@routes.post('/billing_projects/{billing_project}/close')
@check_csrf_token
@auth.web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_close_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']

    session = await aiohttp_session.get_session(request)
    try:
        await _handle_ui_error(session, _close_billing_project, db, billing_project)
        set_message(session, f'Closed billing project {billing_project}.', 'info')
    finally:
        return web.HTTPFound(deploy_config.external_url('batch', '/billing_projects'))  # pylint: disable=lost-exception


@routes.post('/api/v1alpha/billing_projects/{billing_project}/close')
@rest_authenticated_developers_or_auth_only
async def api_close_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']

    await _handle_api_error(_close_billing_project, db, billing_project)
    return json_response(billing_project)


async def _reopen_billing_project(db, billing_project):
    @transaction(db)
    async def open_project(tx):
        row = await tx.execute_and_fetchone(
            "SELECT name_cs, `status` FROM billing_projects WHERE name_cs = %s FOR UPDATE;", (billing_project,)
        )
        if not row:
            raise NonExistentBillingProjectError(billing_project)
        assert row['name_cs'] == billing_project
        if row['status'] == 'deleted':
            raise BatchUserError(f'Billing project {billing_project} has been deleted and cannot be reopened.', 'error')
        if row['status'] == 'open':
            raise BatchOperationAlreadyCompletedError(f'Billing project {billing_project} is already open.', 'info')

        await tx.execute_update("UPDATE billing_projects SET `status` = 'open' WHERE name_cs = %s;", (billing_project,))

    await open_project()  # pylint: disable=no-value-for-parameter


@routes.post('/billing_projects/{billing_project}/reopen')
@check_csrf_token
@auth.web_authenticated_developers_only(redirect=False)
@catch_ui_error_in_dev
async def post_reopen_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']

    session = await aiohttp_session.get_session(request)
    try:
        await _handle_ui_error(session, _reopen_billing_project, db, billing_project)
        set_message(session, f'Re-opened billing project {billing_project}.', 'info')
    finally:
        return web.HTTPFound(deploy_config.external_url('batch', '/billing_projects'))  # pylint: disable=lost-exception


@routes.post('/api/v1alpha/billing_projects/{billing_project}/reopen')
@rest_authenticated_developers_or_auth_only
async def api_reopen_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']
    await _handle_api_error(_reopen_billing_project, db, billing_project)
    return json_response(billing_project)


async def _delete_billing_project(db, billing_project):
    @transaction(db)
    async def delete_project(tx):
        row = await tx.execute_and_fetchone(
            'SELECT name_cs, `status` FROM billing_projects WHERE name_cs = %s FOR UPDATE;', (billing_project,)
        )
        if not row:
            raise NonExistentBillingProjectError(billing_project)
        assert row['name_cs'] == billing_project, row
        if row['status'] == 'deleted':
            raise BatchOperationAlreadyCompletedError(f'Billing project {billing_project} is already deleted.', 'info')
        if row['status'] == 'open':
            raise BatchUserError(f'Billing project {billing_project} is open and cannot be deleted.', 'error')

        await tx.execute_update(
            "UPDATE billing_projects SET `status` = 'deleted' WHERE name_cs = %s;", (billing_project,)
        )

    await delete_project()  # pylint: disable=no-value-for-parameter


@routes.post('/api/v1alpha/billing_projects/{billing_project}/delete')
@rest_authenticated_developers_or_auth_only
async def api_delete_billing_projects(request, userdata):  # pylint: disable=unused-argument
    db: Database = request.app['db']
    billing_project = request.match_info['billing_project']

    await _handle_api_error(_delete_billing_project, db, billing_project)
    return json_response(billing_project)


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

    regions = {
        record['region']: record['region_id']
        async for record in db.select_and_fetchall('SELECT region_id, region from regions')
    }
    app['regions'] = regions


@routes.get('')
@routes.get('/')
@auth.web_authenticated_users_only()
@catch_ui_error_in_dev
async def index(request, userdata):  # pylint: disable=unused-argument
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


async def cancel_batch_loop_body(app):
    client_session: httpx.ClientSession = app['client_session']
    await retry_transient_errors(
        client_session.post,
        deploy_config.url('batch-driver', '/api/v1alpha/batches/cancel'),
        headers=app['batch_headers'],
    )

    should_wait = True
    return should_wait


async def delete_batch_loop_body(app):
    client_session: httpx.ClientSession = app['client_session']
    await retry_transient_errors(
        client_session.post,
        deploy_config.url('batch-driver', '/api/v1alpha/batches/delete'),
        headers=app['batch_headers'],
    )

    should_wait = True
    return should_wait


class BatchFrontEndAccessLogger(AccessLogger):
    def __init__(self, logger: logging.Logger, log_format: str):
        super().__init__(logger, log_format)
        if DEFAULT_NAMESPACE == 'default':
            self.exclude = [
                (endpoint[0], re.compile(deploy_config.base_path('batch') + endpoint[1]))
                for endpoint in [
                    ('POST', '/api/v1alpha/batches/\\d*/jobs/create'),
                    ('GET', '/api/v1alpha/batches/\\d*'),
                    ('GET', '/metrics'),
                ]
            ]
        else:
            self.exclude = []

    def log(self, request, response, time):
        for method, path_expr in self.exclude:
            if path_expr.fullmatch(request.path) and method == request.method:
                return

        super().log(request, response, time)


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

    regions = {
        record['region']: record['region_id']
        async for record in db.select_and_fetchall('SELECT region_id, region from regions')
    }

    if len(regions) != 0:
        assert max(regions.values()) < 64, str(regions)
    app['regions'] = regions

    fs = get_cloud_async_fs()
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
    install_profiler_if_requested('batch')

    app = web.Application(
        client_max_size=HTTP_CLIENT_MAX_SIZE, middlewares=[unavailable_if_frozen, monitor_endpoints_middleware]
    )
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
        port=int(os.environ['PORT']),
        access_log_class=BatchFrontEndAccessLogger,
        ssl_context=internal_server_ssl_context(),
    )
