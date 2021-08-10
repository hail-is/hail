import logging
import json
from functools import wraps
import concurrent
import copy
import asyncio
import signal
import dictdiffer
from aiohttp import web
import aiohttp_session
import kubernetes_asyncio as kube
import google.oauth2.service_account
from prometheus_async.aio.web import server_stats
from gear import (
    Database,
    setup_aiohttp_session,
    rest_authenticated_developers_only,
    web_authenticated_developers_only,
    check_csrf_token,
    transaction,
    monitor_endpoint,
)
from hailtop.hail_logging import AccessLogger
from hailtop.config import get_deploy_config
from hailtop.httpx import client_session
from hailtop.utils import (
    time_msecs,
    RateLimit,
    serialization,
    Notice,
    periodically_call,
    AsyncWorkerPool,
    request_retry_transient_errors,
    dump_all_stacktraces,
)
from hailtop.tls import internal_server_ssl_context
from hailtop import aiogoogle, aiotools
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, render_template, set_message
import googlecloudprofiler
import uvloop

from ..log_store import LogStore
from ..batch import cancel_batch_in_db
from ..batch_configuration import (
    REFRESH_INTERVAL_IN_SECONDS,
    DEFAULT_NAMESPACE,
    BATCH_BUCKET_NAME,
    HAIL_SHA,
    HAIL_SHOULD_PROFILE,
    HAIL_SHOULD_CHECK_INVARIANTS,
    PROJECT,
    MACHINE_NAME_PREFIX,
)
from ..globals import HTTP_CLIENT_MAX_SIZE
from ..inst_coll_config import InstanceCollectionConfigs

from .zone_monitor import ZoneMonitor
from .gce import GCEEventMonitor
from .canceller import Canceller
from .instance_collection_manager import InstanceCollectionManager
from .job import mark_job_complete, mark_job_started
from .k8s_cache import K8sCache
from .pool import Pool
from ..utils import query_billing_projects, unreserved_worker_data_disk_size_gib, batch_only, authorization_token
from ..exceptions import BatchUserError

uvloop.install()

log = logging.getLogger('batch')

log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')

routes = web.RouteTableDef()

deploy_config = get_deploy_config()


def instance_name_from_request(request):
    instance_name = request.headers.get('X-Hail-Instance-Name')
    if instance_name is None:
        raise ValueError(f'request is missing required header X-Hail-Instance-Name: {request}')
    return instance_name


def instance_from_request(request):
    instance_name = instance_name_from_request(request)
    inst_coll_manager = request.app['inst_coll_manager']
    return inst_coll_manager.get_instance(instance_name)


def activating_instances_only(fun):
    @wraps(fun)
    async def wrapped(request):
        instance = instance_from_request(request)
        if not instance:
            instance_name = instance_name_from_request(request)
            log.info(f'instance {instance_name} not found')
            raise web.HTTPUnauthorized()

        if instance.state != 'pending':
            log.info(f'instance {instance.name} not pending')
            raise web.HTTPUnauthorized()

        activation_token = authorization_token(request)
        if not activation_token:
            log.info(f'activation token not found for instance {instance.name}')
            raise web.HTTPUnauthorized()

        db = request.app['db']
        record = await db.select_and_fetchone(
            'SELECT state FROM instances WHERE name = %s AND activation_token = %s;', (instance.name, activation_token)
        )
        if not record:
            log.info(f'instance {instance.name}, activation token not found in database')
            raise web.HTTPUnauthorized()

        resp = await fun(request, instance)

        return resp

    return wrapped


def active_instances_only(fun):
    @wraps(fun)
    async def wrapped(request):
        instance = instance_from_request(request)
        if not instance:
            instance_name = instance_name_from_request(request)
            log.info(f'instance not found {instance_name}')
            raise web.HTTPUnauthorized()

        if instance.state != 'active':
            log.info(f'instance not active {instance.name}')
            raise web.HTTPUnauthorized()

        token = authorization_token(request)
        if not token:
            log.info(f'token not found for instance {instance.name}')
            raise web.HTTPUnauthorized()

        db = request.app['db']
        record = await db.select_and_fetchone(
            'SELECT state FROM instances WHERE name = %s AND token = %s;', (instance.name, token)
        )
        if not record:
            log.info(f'instance {instance.name}, token not found in database')
            raise web.HTTPUnauthorized()

        await instance.mark_healthy()

        return await fun(request, instance)

    return wrapped


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


@routes.get('/check_invariants')
@monitor_endpoint
@rest_authenticated_developers_only
async def get_check_invariants(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    data = {
        'check_incremental_error': app['check_incremental_error'],
        'check_resource_aggregation_error': app['check_resource_aggregation_error'],
    }
    return web.json_response(data=data)


@routes.patch('/api/v1alpha/batches/{user}/{batch_id}/close')
@monitor_endpoint
@batch_only
async def close_batch(request):
    db = request.app['db']

    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])

    record = await db.select_and_fetchone(
        '''
SELECT state FROM batches WHERE user = %s AND id = %s;
''',
        (user, batch_id),
    )
    if not record:
        raise web.HTTPNotFound()

    request.app['scheduler_state_changed'].notify()

    return web.Response()


def set_cancel_state_changed(app):
    app['cancel_running_state_changed'].set()
    app['cancel_creating_state_changed'].set()
    app['cancel_ready_state_changed'].set()


@routes.post('/api/v1alpha/batches/cancel')
@monitor_endpoint
@batch_only
async def cancel_batch(request):
    set_cancel_state_changed(request.app)
    return web.Response()


@routes.post('/api/v1alpha/batches/delete')
@monitor_endpoint
@batch_only
async def delete_batch(request):
    set_cancel_state_changed(request.app)
    return web.Response()


async def get_gsa_key_1(instance):
    log.info(f'returning gsa-key to activating instance {instance}')
    with open('/gsa-key/key.json', 'r') as f:
        key = json.loads(f.read())
    return web.json_response({'key': key})


async def activate_instance_1(request, instance):
    body = await request.json()
    ip_address = body['ip_address']

    log.info(f'activating {instance}')
    timestamp = time_msecs()
    token = await instance.activate(ip_address, timestamp)
    await instance.mark_healthy()

    return web.json_response({'token': token})


@routes.get('/api/v1alpha/instances/gsa_key')
@monitor_endpoint
@activating_instances_only
async def get_gsa_key(request, instance):  # pylint: disable=unused-argument
    return await asyncio.shield(get_gsa_key_1(instance))


@routes.post('/api/v1alpha/instances/activate')
@monitor_endpoint
@activating_instances_only
async def activate_instance(request, instance):
    return await asyncio.shield(activate_instance_1(request, instance))


async def deactivate_instance_1(instance):
    log.info(f'deactivating {instance}')
    await instance.deactivate('deactivated')
    await instance.mark_healthy()
    return web.Response()


@routes.post('/api/v1alpha/instances/deactivate')
@monitor_endpoint
@active_instances_only
async def deactivate_instance(request, instance):  # pylint: disable=unused-argument
    return await asyncio.shield(deactivate_instance_1(instance))


async def job_complete_1(request, instance):
    body = await request.json()
    job_status = body['status']

    batch_id = job_status['batch_id']
    job_id = job_status['job_id']
    attempt_id = job_status['attempt_id']

    state = job_status['state']
    if state == 'succeeded':
        new_state = 'Success'
    elif state == 'error':
        new_state = 'Error'
    else:
        assert state == 'failed', state
        new_state = 'Failed'

    start_time = job_status['start_time']
    end_time = job_status['end_time']
    status = job_status['status']
    resources = job_status.get('resources')

    await mark_job_complete(
        request.app,
        batch_id,
        job_id,
        attempt_id,
        instance.name,
        new_state,
        status,
        start_time,
        end_time,
        'completed',
        resources,
    )

    await instance.mark_healthy()

    return web.Response()


@routes.post('/api/v1alpha/instances/job_complete')
@monitor_endpoint
@active_instances_only
async def job_complete(request, instance):
    return await asyncio.shield(job_complete_1(request, instance))


async def job_started_1(request, instance):
    body = await request.json()
    job_status = body['status']

    batch_id = job_status['batch_id']
    job_id = job_status['job_id']
    attempt_id = job_status['attempt_id']
    start_time = job_status['start_time']
    resources = job_status.get('resources')

    await mark_job_started(request.app, batch_id, job_id, attempt_id, instance, start_time, resources)

    await instance.mark_healthy()

    return web.Response()


@routes.post('/api/v1alpha/instances/job_started')
@monitor_endpoint
@active_instances_only
async def job_started(request, instance):
    return await asyncio.shield(job_started_1(request, instance))


@routes.get('/')
@routes.get('')
@monitor_endpoint
@web_authenticated_developers_only()
async def get_index(request, userdata):
    app = request.app
    db: Database = app['db']
    inst_coll_manager: InstanceCollectionManager = app['inst_coll_manager']

    ready_cores = await db.select_and_fetchone(
        '''
SELECT CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu
FROM user_inst_coll_resources;
'''
    )
    ready_cores_mcpu = ready_cores['ready_cores_mcpu']

    page_context = {
        'pools': inst_coll_manager.pools.values(),
        'jpim': inst_coll_manager.job_private_inst_manager,
        'instance_id': app['instance_id'],
        'n_instances_by_state': inst_coll_manager.global_n_instances_by_state,
        'instances': inst_coll_manager.name_instance.values(),
        'ready_cores_mcpu': ready_cores_mcpu,
        'live_total_cores_mcpu': inst_coll_manager.global_live_total_cores_mcpu,
        'live_free_cores_mcpu': inst_coll_manager.global_live_free_cores_mcpu,
    }
    return await render_template('batch-driver', request, userdata, 'index.html', page_context)


def validate(session, url_path, name, value, predicate, description):
    if not predicate(value):
        set_message(session, f'{name} invalid: {value}.  Must be {description}.', 'error')
        raise web.HTTPFound(deploy_config.external_url('batch-driver', url_path))
    return value


def validate_int(session, url_path, name, value, predicate, description):
    try:
        i = int(value)
    except ValueError as e:
        set_message(session, f'{name} invalid: {value}.  Must be an integer.', 'error')
        raise web.HTTPFound(deploy_config.external_url('batch-driver', url_path)) from e
    return validate(session, url_path, name, i, predicate, description)


async def refresh_inst_colls_on_front_end(app):
    async with client_session() as session:
        await request_retry_transient_errors(
            session,
            'PATCH',
            deploy_config.url('batch', '/api/v1alpha/inst_colls/refresh'),
            headers=app['batch_headers'],
        )


@routes.post('/config-update/pool/{pool}')
@check_csrf_token
@monitor_endpoint
@web_authenticated_developers_only()
async def pool_config_update(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    inst_coll_manager: InstanceCollectionManager = app['inst_coll_manager']

    session = await aiohttp_session.get_session(request)

    pool_name = request.match_info['pool']
    pool = inst_coll_manager.get_inst_coll(pool_name)
    pool_url_path = f'/inst_coll/pool/{pool_name}'

    if not isinstance(pool, Pool):
        set_message(session, f'Unknown pool {pool_name}.', 'error')
        raise web.HTTPFound(deploy_config.external_url('batch-driver', pool_url_path))

    post = await request.post()

    worker_type = pool.worker_type

    if worker_type == 'standard':
        valid_worker_cores = (1, 2, 4, 8, 16, 32, 64, 96)
    else:
        valid_worker_cores = (2, 4, 8, 16, 32, 64, 96)
    worker_cores = validate_int(
        session,
        pool_url_path,
        f'{worker_type} worker cores',
        post['worker_cores'],
        lambda v: v in valid_worker_cores,
        f'one of {", ".join(str(v) for v in valid_worker_cores)}',
    )

    standing_worker_cores = validate_int(
        session,
        pool_url_path,
        f'{worker_type} standing worker cores',
        post['standing_worker_cores'],
        lambda v: v in valid_worker_cores,
        f'one of {", ".join(str(v) for v in valid_worker_cores)}',
    )

    boot_disk_size_gb = validate_int(
        session,
        pool_url_path,
        'Worker boot disk size',
        post['boot_disk_size_gb'],
        lambda v: v >= 10,
        'a positive integer greater than or equal to 10',
    )

    worker_local_ssd_data_disk = 'worker_local_ssd_data_disk' in post

    worker_pd_ssd_data_disk_size_gb = validate_int(
        session,
        pool_url_path,
        'Worker PD SSD data disk size (in GB)',
        post['worker_pd_ssd_data_disk_size_gb'],
        lambda v: v >= 0,
        'a nonnegative integer',
    )

    if not worker_local_ssd_data_disk and worker_pd_ssd_data_disk_size_gb == 0:
        set_message(session, 'Either the worker must use a local SSD or PD SSD data disk must be non-zero.', 'error')
        raise web.HTTPFound(deploy_config.external_url('batch-driver', pool_url_path))

    if worker_local_ssd_data_disk and worker_pd_ssd_data_disk_size_gb > 0:
        set_message(session, 'Worker cannot both use local SSD and have a non-zero PD SSD data disk.', 'error')
        raise web.HTTPFound(deploy_config.external_url('batch-driver', pool_url_path))

    if not worker_local_ssd_data_disk:
        unreserved_disk_storage_gb = unreserved_worker_data_disk_size_gib(
            worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb, worker_cores
        )
        if unreserved_disk_storage_gb < 0:
            min_disk_storage = worker_pd_ssd_data_disk_size_gb - unreserved_disk_storage_gb
            set_message(session, f'PD SSD must be at least {min_disk_storage} GB', 'error')
            raise web.HTTPFound(deploy_config.external_url('batch-driver', pool_url_path))

    max_instances = validate_int(
        session, pool_url_path, 'Max instances', post['max_instances'], lambda v: v > 0, 'a positive integer'
    )

    max_live_instances = validate_int(
        session, pool_url_path, 'Max live instances', post['max_live_instances'], lambda v: v > 0, 'a positive integer'
    )

    enable_standing_worker = 'enable_standing_worker' in post

    await pool.configure(
        worker_cores,
        boot_disk_size_gb,
        worker_local_ssd_data_disk,
        worker_pd_ssd_data_disk_size_gb,
        enable_standing_worker,
        standing_worker_cores,
        max_instances,
        max_live_instances,
    )

    await refresh_inst_colls_on_front_end(app)

    set_message(session, f'Updated configuration for {pool}.', 'info')

    return web.HTTPFound(deploy_config.external_url('batch-driver', pool_url_path))


@routes.post('/config-update/jpim')
@check_csrf_token
@monitor_endpoint
@web_authenticated_developers_only()
async def job_private_config_update(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    inst_coll_manager: InstanceCollectionManager = app['inst_coll_manager']

    session = await aiohttp_session.get_session(request)

    job_private_inst_manager = inst_coll_manager.job_private_inst_manager
    url_path = '/inst_coll/jpim'

    post = await request.post()

    boot_disk_size_gb = validate_int(
        session,
        url_path,
        'Worker boot disk size',
        post['boot_disk_size_gb'],
        lambda v: v >= 10,
        'a positive integer greater than or equal to 10',
    )

    max_instances = validate_int(
        session, url_path, 'Max instances', post['max_instances'], lambda v: v > 0, 'a positive integer'
    )

    max_live_instances = validate_int(
        session, url_path, 'Max live instances', post['max_live_instances'], lambda v: v > 0, 'a positive integer'
    )

    await job_private_inst_manager.configure(boot_disk_size_gb, max_instances, max_live_instances)

    await refresh_inst_colls_on_front_end(app)

    set_message(session, f'Updated configuration for {job_private_inst_manager}.', 'info')

    return web.HTTPFound(deploy_config.external_url('batch-driver', url_path))


@routes.get('/inst_coll/pool/{pool}')
@monitor_endpoint
@web_authenticated_developers_only()
async def get_pool(request, userdata):
    app = request.app
    inst_coll_manager: InstanceCollectionManager = app['inst_coll_manager']

    session = await aiohttp_session.get_session(request)

    pool_name = request.match_info['pool']
    pool = inst_coll_manager.get_inst_coll(pool_name)

    if not isinstance(pool, Pool):
        set_message(session, f'Unknown pool {pool_name}.', 'error')
        raise web.HTTPFound(deploy_config.external_url('batch-driver', '/'))

    user_resources = await pool.scheduler.compute_fair_share()
    user_resources = sorted(
        user_resources.values(),
        key=lambda record: record['ready_cores_mcpu'] + record['running_cores_mcpu'],
        reverse=True,
    )

    ready_cores_mcpu = sum([record['ready_cores_mcpu'] for record in user_resources])

    page_context = {
        'pool': pool,
        'instances': pool.name_instance.values(),
        'user_resources': user_resources,
        'ready_cores_mcpu': ready_cores_mcpu,
    }

    return await render_template('batch-driver', request, userdata, 'pool.html', page_context)


@routes.get('/inst_coll/jpim')
@monitor_endpoint
@web_authenticated_developers_only()
async def get_job_private_inst_manager(request, userdata):
    app = request.app
    inst_coll_manager: InstanceCollectionManager = app['inst_coll_manager']

    job_private_inst_manager = inst_coll_manager.job_private_inst_manager

    user_resources = await job_private_inst_manager.compute_fair_share()
    user_resources = sorted(
        user_resources.values(),
        key=lambda record: record['n_ready_jobs'] + record['n_creating_jobs'] + record['n_running_jobs'],
        reverse=True,
    )

    n_ready_jobs = sum([record['n_ready_jobs'] for record in user_resources])
    n_creating_jobs = sum([record['n_creating_jobs'] for record in user_resources])
    n_running_jobs = sum([record['n_running_jobs'] for record in user_resources])

    page_context = {
        'jpim': job_private_inst_manager,
        'instances': job_private_inst_manager.name_instance.values(),
        'user_resources': user_resources,
        'n_ready_jobs': n_ready_jobs,
        'n_creating_jobs': n_creating_jobs,
        'n_running_jobs': n_running_jobs,
    }

    return await render_template('batch-driver', request, userdata, 'job_private.html', page_context)


@routes.get('/user_resources')
@monitor_endpoint
@web_authenticated_developers_only()
async def get_user_resources(request, userdata):
    app = request.app
    db: Database = app['db']

    records = db.execute_and_fetchall(
        '''
SELECT user,
  CAST(COALESCE(SUM(n_ready_jobs), 0) AS SIGNED) AS n_ready_jobs,
  CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu,
  CAST(COALESCE(SUM(n_running_jobs), 0) AS SIGNED) AS n_running_jobs,
  CAST(COALESCE(SUM(running_cores_mcpu), 0) AS SIGNED) AS running_cores_mcpu
FROM user_inst_coll_resources
GROUP BY user
HAVING n_ready_jobs + n_running_jobs > 0;
'''
    )

    user_resources = sorted(
        [record async for record in records],
        key=lambda record: record['ready_cores_mcpu'] + record['running_cores_mcpu'],
        reverse=True,
    )

    page_context = {'user_resources': user_resources}
    return await render_template('batch-driver', request, userdata, 'user_resources.html', page_context)


async def check_incremental(app, db):
    @transaction(db, read_only=True)
    async def check(tx):
        user_inst_coll_with_broken_resources = tx.execute_and_fetchall(
            '''
SELECT
  t.*,
  u.*
FROM
(
  SELECT user, inst_coll,
    CAST(COALESCE(SUM(state = 'Ready' AND runnable), 0) AS SIGNED) AS actual_n_ready_jobs,
    CAST(COALESCE(SUM(cores_mcpu * (state = 'Ready' AND runnable)), 0) AS SIGNED) AS actual_ready_cores_mcpu,
    CAST(COALESCE(SUM(state = 'Running' AND (NOT cancelled)), 0) AS SIGNED) AS actual_n_running_jobs,
    CAST(COALESCE(SUM(cores_mcpu * (state = 'Running' AND (NOT cancelled))), 0) AS SIGNED) AS actual_running_cores_mcpu,
    CAST(COALESCE(SUM(state = 'Creating' AND (NOT cancelled)), 0) AS SIGNED) AS actual_n_creating_jobs,
    CAST(COALESCE(SUM(state = 'Ready' AND cancelled), 0) AS SIGNED) AS actual_n_cancelled_ready_jobs,
    CAST(COALESCE(SUM(state = 'Running' AND cancelled), 0) AS SIGNED) AS actual_n_cancelled_running_jobs,
    CAST(COALESCE(SUM(state = 'Creating' AND cancelled), 0) AS SIGNED) AS actual_n_cancelled_creating_jobs
  FROM
  (
    SELECT batches.user, jobs.state, jobs.cores_mcpu, jobs.inst_coll,
      (jobs.always_run OR NOT (jobs.cancelled OR batches.cancelled)) AS runnable,
      (NOT jobs.always_run AND (jobs.cancelled OR batches.cancelled)) AS cancelled
    FROM batches
    INNER JOIN jobs ON batches.id = jobs.batch_id
    WHERE batches.`state` = 'running'
  ) as v
  GROUP BY user, inst_coll
) as t
INNER JOIN
(
  SELECT user, inst_coll,
    CAST(COALESCE(SUM(n_ready_jobs), 0) AS SIGNED) AS expected_n_ready_jobs,
    CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS expected_ready_cores_mcpu,
    CAST(COALESCE(SUM(n_running_jobs), 0) AS SIGNED) AS expected_n_running_jobs,
    CAST(COALESCE(SUM(running_cores_mcpu), 0) AS SIGNED) AS expected_running_cores_mcpu,
    CAST(COALESCE(SUM(n_creating_jobs), 0) AS SIGNED) AS expected_n_creating_jobs,
    CAST(COALESCE(SUM(n_cancelled_ready_jobs), 0) AS SIGNED) AS expected_n_cancelled_ready_jobs,
    CAST(COALESCE(SUM(n_cancelled_running_jobs), 0) AS SIGNED) AS expected_n_cancelled_running_jobs,
    CAST(COALESCE(SUM(n_cancelled_creating_jobs), 0) AS SIGNED) AS expected_n_cancelled_creating_jobs
  FROM user_inst_coll_resources
  GROUP BY user, inst_coll
) AS u
ON t.user = u.user AND t.inst_coll = u.inst_coll
WHERE actual_n_ready_jobs != expected_n_ready_jobs
   OR actual_ready_cores_mcpu != expected_ready_cores_mcpu
   OR actual_n_running_jobs != expected_n_running_jobs
   OR actual_running_cores_mcpu != expected_running_cores_mcpu
   OR actual_n_creating_jobs != expected_n_creating_jobs
   OR actual_n_cancelled_ready_jobs != expected_n_cancelled_ready_jobs
   OR actual_n_cancelled_running_jobs != expected_n_cancelled_running_jobs
   OR actual_n_cancelled_creating_jobs != expected_n_cancelled_creating_jobs
LOCK IN SHARE MODE;
'''
        )

        async for record in user_inst_coll_with_broken_resources:
            log.error(f'user_inst_coll_resources corrupt: {record}')

    try:
        await check()  # pylint: disable=no-value-for-parameter
    except Exception as e:
        app['check_incremental_error'] = serialization.exception_to_dict(e)
        log.exception('while checking incremental')


async def check_resource_aggregation(app, db):
    def json_to_value(x):
        if x is None:
            return x
        return json.loads(x)

    def merge(r1, r2):
        if r1 is None:
            r1 = {}
        if r2 is None:
            r2 = {}

        result = {}

        def add_items(d):
            for k, v in d.items():
                if k not in result:
                    result[k] = v
                else:
                    result[k] += v

        add_items(r1)
        add_items(r2)
        return result

    def seqop(result, k, v):
        if k not in result:
            result[k] = v
        else:
            result[k] = merge(result[k], v)

    def fold(d, key_f):
        if d is None:
            d = {}
        d = copy.deepcopy(d)
        result = {}
        for k, v in d.items():
            seqop(result, key_f(k), v)
        return result

    @transaction(db, read_only=True)
    async def check(tx):
        attempt_resources = tx.execute_and_fetchall(
            '''
SELECT attempt_resources.batch_id, attempt_resources.job_id, attempt_resources.attempt_id,
  JSON_OBJECTAGG(resource, quantity * GREATEST(COALESCE(end_time - start_time, 0), 0)) as resources
FROM attempt_resources
INNER JOIN attempts
ON attempts.batch_id = attempt_resources.batch_id AND
  attempts.job_id = attempt_resources.job_id AND
  attempts.attempt_id = attempt_resources.attempt_id
GROUP BY batch_id, job_id, attempt_id
LOCK IN SHARE MODE;
'''
        )

        agg_job_resources = tx.execute_and_fetchall(
            '''
SELECT batch_id, job_id, JSON_OBJECTAGG(resource, `usage`) as resources
FROM aggregated_job_resources
GROUP BY batch_id, job_id
LOCK IN SHARE MODE;
'''
        )

        agg_batch_resources = tx.execute_and_fetchall(
            '''
SELECT batch_id, billing_project, JSON_OBJECTAGG(resource, `usage`) as resources
FROM (
  SELECT batch_id, resource, SUM(`usage`) AS `usage`
  FROM aggregated_batch_resources
  GROUP BY batch_id, resource) AS t
JOIN batches ON batches.id = t.batch_id
GROUP BY t.batch_id, billing_project
LOCK IN SHARE MODE;
'''
        )

        agg_billing_project_resources = tx.execute_and_fetchall(
            '''
SELECT billing_project, JSON_OBJECTAGG(resource, `usage`) as resources
FROM (
  SELECT billing_project, resource, SUM(`usage`) AS `usage`
  FROM aggregated_billing_project_resources
  GROUP BY billing_project, resource) AS t
GROUP BY t.billing_project
LOCK IN SHARE MODE;
'''
        )

        attempt_resources = {
            (record['batch_id'], record['job_id'], record['attempt_id']): json_to_value(record['resources'])
            async for record in attempt_resources
        }

        agg_job_resources = {
            (record['batch_id'], record['job_id']): json_to_value(record['resources'])
            async for record in agg_job_resources
        }

        agg_batch_resources = {
            (record['batch_id'], record['billing_project']): json_to_value(record['resources'])
            async for record in agg_batch_resources
        }

        agg_billing_project_resources = {
            record['billing_project']: json_to_value(record['resources'])
            async for record in agg_billing_project_resources
        }

        attempt_by_batch_resources = fold(attempt_resources, lambda k: k[0])
        attempt_by_job_resources = fold(attempt_resources, lambda k: (k[0], k[1]))
        job_by_batch_resources = fold(agg_job_resources, lambda k: k[0])
        batch_by_billing_project_resources = fold(agg_batch_resources, lambda k: k[1])

        agg_batch_resources_2 = {batch_id: resources for (batch_id, _), resources in agg_batch_resources.items()}

        assert attempt_by_batch_resources == agg_batch_resources_2, (
            dictdiffer.diff(attempt_by_batch_resources, agg_batch_resources_2),
            attempt_by_batch_resources,
            agg_batch_resources_2,
        )
        assert attempt_by_job_resources == agg_job_resources, (
            dictdiffer.diff(attempt_by_job_resources, agg_job_resources),
            attempt_by_job_resources,
            agg_job_resources,
        )
        assert job_by_batch_resources == agg_batch_resources_2, (
            dictdiffer.diff(job_by_batch_resources, agg_batch_resources_2),
            job_by_batch_resources,
            agg_batch_resources_2,
        )
        assert batch_by_billing_project_resources == agg_billing_project_resources, (
            dictdiffer.diff(batch_by_billing_project_resources, agg_billing_project_resources),
            batch_by_billing_project_resources,
            agg_billing_project_resources,
        )

    try:
        await check()  # pylint: disable=no-value-for-parameter
    except Exception as e:
        app['check_resource_aggregation_error'] = serialization.exception_to_dict(e)
        log.exception('while checking resource aggregation')


async def _cancel_batch(app, batch_id):
    try:
        await cancel_batch_in_db(app['db'], batch_id)
    except BatchUserError as exc:
        log.info(f'cannot cancel batch because {exc.message}')
        return
    set_cancel_state_changed(app)


async def monitor_billing_limits(app):
    db: Database = app['db']

    records = await query_billing_projects(db)
    for record in records:
        limit = record['limit']
        accrued_cost = record['accrued_cost']
        if limit is not None and accrued_cost >= limit:
            running_batches = db.execute_and_fetchall(
                '''
SELECT id
FROM batches
WHERE billing_project = %s AND state = 'running';
''',
                (record['billing_project'],),
            )
            async for batch in running_batches:
                await _cancel_batch(app, batch['id'])


async def cancel_fast_failing_batches(app):
    db: Database = app['db']

    records = db.select_and_fetchall(
        '''
SELECT id
FROM batches
WHERE state = 'running' AND cancel_after_n_failures IS NOT NULL AND n_failed >= cancel_after_n_failures
'''
    )
    async for batch in records:
        await _cancel_batch(app, batch['id'])


async def scheduling_cancelling_bump(app):
    log.info('scheduling cancelling bump loop')
    app['scheduler_state_changed'].notify()
    app['cancel_ready_state_changed'].set()
    app['cancel_creating_state_changed'].set()
    app['cancel_running_state_changed'].set()


async def on_startup(app):
    app['task_manager'] = aiotools.BackgroundTaskManager()
    pool = concurrent.futures.ThreadPoolExecutor()
    app['blocking_pool'] = pool

    kube.config.load_incluster_config()
    k8s_client = kube.client.CoreV1Api()
    k8s_cache = K8sCache(k8s_client, refresh_time=5)
    app['k8s_cache'] = k8s_cache

    db = Database()
    await db.async_init(maxsize=50)
    app['db'] = db

    row = await db.select_and_fetchone(
        '''
SELECT instance_id, internal_token FROM globals;
'''
    )

    instance_id = row['instance_id']
    log.info(f'instance_id {instance_id}')
    app['instance_id'] = instance_id

    app['internal_token'] = row['internal_token']

    app['batch_headers'] = {'Authorization': f'Bearer {row["internal_token"]}'}

    resources = db.select_and_fetchall('SELECT resource FROM resources;')

    app['resources'] = [record['resource'] async for record in resources]

    aiogoogle_credentials = aiogoogle.Credentials.from_file('/gsa-key/key.json')
    compute_client = aiogoogle.ComputeClient(PROJECT, credentials=aiogoogle_credentials)
    app['compute_client'] = compute_client

    logging_client = aiogoogle.LoggingClient(
        credentials=aiogoogle_credentials,
        # The project-wide logging quota is 60 request/m.  The event
        # loop sleeps 15s per iteration, so the max rate is 4
        # iterations/m.  Note, the event loop could make multiple
        # logging requests per iteration, so these numbers are not
        # quite comparable.  I didn't want to consume the entire quota
        # since there will be other users of the logging API (us at
        # the web console, test deployments, etc.)
        rate_limit=RateLimit(10, 60),
    )
    app['logging_client'] = logging_client

    scheduler_state_changed = Notice()
    app['scheduler_state_changed'] = scheduler_state_changed

    cancel_ready_state_changed = asyncio.Event()
    app['cancel_ready_state_changed'] = cancel_ready_state_changed

    cancel_creating_state_changed = asyncio.Event()
    app['cancel_creating_state_changed'] = cancel_creating_state_changed

    cancel_running_state_changed = asyncio.Event()
    app['cancel_running_state_changed'] = cancel_running_state_changed

    async_worker_pool = AsyncWorkerPool(100, queue_size=100)
    app['async_worker_pool'] = async_worker_pool

    credentials = google.oauth2.service_account.Credentials.from_service_account_file('/gsa-key/key.json')
    log_store = LogStore(BATCH_BUCKET_NAME, instance_id, pool, credentials=credentials)
    app['log_store'] = log_store

    zone_monitor = ZoneMonitor(app)
    app['zone_monitor'] = zone_monitor
    await zone_monitor.async_init()

    inst_coll_configs = InstanceCollectionConfigs(app)
    await inst_coll_configs.async_init()

    inst_coll_manager = InstanceCollectionManager(app, MACHINE_NAME_PREFIX)
    app['inst_coll_manager'] = inst_coll_manager
    await inst_coll_manager.async_init(inst_coll_configs)

    canceller = Canceller(app)
    app['canceller'] = canceller
    await canceller.async_init()

    gce_event_monitor = GCEEventMonitor(app, MACHINE_NAME_PREFIX)
    app['gce_event_monitor'] = gce_event_monitor
    await gce_event_monitor.async_init()

    app['check_incremental_error'] = None
    app['check_resource_aggregation_error'] = None

    if HAIL_SHOULD_CHECK_INVARIANTS:
        app['task_manager'].ensure_future(periodically_call(10, check_incremental, app, db))
        app['task_manager'].ensure_future(periodically_call(10, check_resource_aggregation, app, db))

    app['task_manager'].ensure_future(periodically_call(10, monitor_billing_limits, app))

    app['task_manager'].ensure_future(periodically_call(10, cancel_fast_failing_batches, app))

    app['task_manager'].ensure_future(periodically_call(60, scheduling_cancelling_bump, app))


async def on_cleanup(app):
    try:
        app['blocking_pool'].shutdown()
    finally:
        try:
            await app['db'].async_close()
        finally:
            try:
                app['zone_monitor'].shutdown()
            finally:
                try:
                    app['inst_coll_manager'].shutdown()
                finally:
                    try:
                        app['canceller'].shutdown()
                    finally:
                        try:
                            app['gce_event_monitor'].shutdown()
                        finally:
                            try:
                                app['task_manager'].shutdown()
                            finally:
                                del app['k8s_cache'].client
                                await asyncio.gather(
                                    *(t for t in asyncio.all_tasks() if t is not asyncio.current_task())
                                )


def run():
    if HAIL_SHOULD_PROFILE:
        profiler_tag = f'{DEFAULT_NAMESPACE}'
        if profiler_tag == 'default':
            profiler_tag = DEFAULT_NAMESPACE + f'-{HAIL_SHA[0:12]}'
        googlecloudprofiler.start(
            service='batch-driver',
            service_version=profiler_tag,
            # https://cloud.google.com/profiler/docs/profiling-python#agent_logging
            verbose=3,
        )

    app = web.Application(client_max_size=HTTP_CLIENT_MAX_SIZE)
    setup_aiohttp_session(app)

    setup_aiohttp_jinja2(app, 'batch.driver')
    setup_common_static_routes(routes)
    app.add_routes(routes)
    app.router.add_get("/metrics", server_stats)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    asyncio.get_event_loop().add_signal_handler(signal.SIGUSR1, dump_all_stacktraces)

    web.run_app(
        deploy_config.prefix_application(app, 'batch-driver', client_max_size=HTTP_CLIENT_MAX_SIZE),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=internal_server_ssl_context(),
    )
