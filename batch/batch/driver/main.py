import asyncio
import copy
import json
import logging
import os
import re
import signal
import warnings
from collections import defaultdict, namedtuple
from contextlib import AsyncExitStack
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, NoReturn, Set, Tuple

import aiohttp_session
import dictdiffer
import kubernetes_asyncio.client
import kubernetes_asyncio.config
import pandas as pd
import plotly
import plotly.graph_objects as go
import prometheus_client as pc  # type: ignore
import uvloop
from aiohttp import web
from plotly.subplots import make_subplots
from prometheus_async.aio.web import server_stats

from gear import (
    AuthServiceAuthenticator,
    Database,
    K8sCache,
    Transaction,
    check_csrf_token,
    json_request,
    json_response,
    monitor_endpoints_middleware,
    setup_aiohttp_session,
    transaction,
)
from gear.auth import AIOHTTPHandler, UserData
from gear.clients import get_cloud_async_fs
from gear.profiling import install_profiler_if_requested
from hailtop import aiotools, httpx
from hailtop.config import get_deploy_config
from hailtop.hail_logging import AccessLogger
from hailtop.utils import (
    AsyncWorkerPool,
    Notice,
    dump_all_stacktraces,
    flatten,
    periodically_call,
    time_msecs,
)
from web_common import render_template, set_message, setup_aiohttp_jinja2, setup_common_static_routes

from ..batch import cancel_batch_in_db
from ..batch_configuration import (
    BATCH_STORAGE_URI,
    CLOUD,
    DEFAULT_NAMESPACE,
    MACHINE_NAME_PREFIX,
    REFRESH_INTERVAL_IN_SECONDS,
)
from ..cloud.driver import get_cloud_driver
from ..cloud.resource_utils import local_ssd_size, possible_cores_from_worker_type, unreserved_worker_data_disk_size_gib
from ..exceptions import BatchUserError
from ..file_store import FileStore
from ..globals import HTTP_CLIENT_MAX_SIZE
from ..inst_coll_config import InstanceCollectionConfigs, PoolConfig
from ..utils import (
    add_metadata_to_request,
    authorization_token,
    json_to_value,
    query_billing_projects_with_cost,
)
from .canceller import Canceller
from .driver import CloudDriver
from .instance import Instance
from .instance_collection import InstanceCollectionManager, JobPrivateInstanceManager, Pool
from .job import mark_job_complete, mark_job_started

uvloop.install()

log = logging.getLogger('batch')

log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')

routes = web.RouteTableDef()

deploy_config = get_deploy_config()

auth = AuthServiceAuthenticator()

warnings.filterwarnings(
    'ignore',
    ".*Warning: Field or reference 'batch.billing_projects.name' of SELECT #. was resolved in SELECT #.",
    module='aiomysql.*',
)


def instance_name_from_request(request):
    instance_name = request.headers.get('X-Hail-Instance-Name')
    if instance_name is None:
        raise ValueError(f'request is missing required header X-Hail-Instance-Name: {request}')
    return instance_name


def instance_from_request(request):
    instance_name = instance_name_from_request(request)
    inst_coll_manager: InstanceCollectionManager = request.app['driver'].inst_coll_manager
    return inst_coll_manager.get_instance(instance_name)


# Old workers use the Authorization header for their identity token
# but that can conflict with bearer tokens used when the driver is behind another
# auth mechanism
def instance_token(request):
    return request.headers.get('X-Hail-Instance-Token') or authorization_token(request)


def batch_only(fun: AIOHTTPHandler):
    @wraps(fun)
    @auth.authenticated_users_only()
    async def wrapped(request: web.Request, userdata: UserData):
        if userdata['username'] != 'batch':
            raise web.HTTPUnauthorized()
        return await fun(request)

    return wrapped


def activating_instances_only(fun: Callable[[web.Request, Instance], Awaitable[web.StreamResponse]]) -> AIOHTTPHandler:
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

        activation_token = instance_token(request)
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

        return await fun(request, instance)

    return wrapped


def active_instances_only(fun: Callable[[web.Request, Instance], Awaitable[web.StreamResponse]]) -> AIOHTTPHandler:
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

        token = instance_token(request)
        if not token:
            log.info(f'token not found for instance {instance.name}')
            raise web.HTTPUnauthorized()

        inst_coll_manager: InstanceCollectionManager = request.app['driver'].inst_coll_manager
        retrieved_token: str = await inst_coll_manager.name_token_cache.lookup(instance.name)
        if token != retrieved_token:
            log.info('authorization token does not match')
            raise web.HTTPUnauthorized()

        await instance.mark_healthy()

        return await fun(request, instance)

    return wrapped


@routes.get('/healthcheck')
async def get_healthcheck(_) -> web.Response:
    return web.Response()


@routes.get('/check_invariants')
@auth.authenticated_developers_only()
async def get_check_invariants(request: web.Request, _) -> web.Response:
    db: Database = request.app['db']
    incremental_result, resource_agg_result = await asyncio.gather(
        check_incremental(db), check_resource_aggregation(db), return_exceptions=True
    )
    return json_response({
        'check_incremental_error': incremental_result,
        'check_resource_aggregation_error': resource_agg_result,
    })


@routes.patch('/api/v1alpha/batches/{user}/{batch_id}/update')
@batch_only
@add_metadata_to_request
async def update_batch(request):
    db = request.app['db']

    user = request.match_info['user']
    batch_id = int(request.match_info['batch_id'])

    record = await db.select_and_fetchone(
        """
SELECT state FROM batches WHERE user = %s AND id = %s;
""",
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
@batch_only
async def cancel_batch(request):
    set_cancel_state_changed(request.app)
    return web.Response()


@routes.post('/api/v1alpha/batches/delete')
@batch_only
async def delete_batch(request):
    set_cancel_state_changed(request.app)
    return web.Response()


async def activate_instance_1(request, instance):
    body = await json_request(request)
    ip_address = body['ip_address']

    log.info(f'activating {instance}')
    timestamp = time_msecs()
    token = await instance.activate(ip_address, timestamp)
    await instance.mark_healthy()

    return json_response({'token': token})


@routes.post('/api/v1alpha/instances/activate')
@activating_instances_only
@add_metadata_to_request
async def activate_instance(request, instance):
    return await asyncio.shield(activate_instance_1(request, instance))


async def deactivate_instance_1(instance):
    log.info(f'deactivating {instance}')
    await instance.deactivate('deactivated')
    await instance.mark_healthy()


@routes.post('/api/v1alpha/instances/deactivate')
@active_instances_only
@add_metadata_to_request
async def deactivate_instance(_, instance: Instance) -> web.Response:
    await asyncio.shield(deactivate_instance_1(instance))
    return web.Response()


@routes.post('/instances/{instance_name}/kill')
@auth.authenticated_developers_only()
async def kill_instance(request: web.Request, _) -> NoReturn:
    instance_name = request.match_info['instance_name']

    inst_coll_manager: InstanceCollectionManager = request.app['driver'].inst_coll_manager
    instance = inst_coll_manager.get_instance(instance_name)

    if instance is None:
        raise web.HTTPNotFound()

    session = await aiohttp_session.get_session(request)
    if instance.state == 'active':
        await asyncio.shield(instance.kill())
        set_message(session, f'Killed instance {instance_name}', 'info')
    else:
        set_message(session, 'Cannot kill a non-active instance', 'error')

    pool_name = instance.inst_coll.name
    pool_url_path = f'/inst_coll/pool/{pool_name}'
    raise web.HTTPFound(deploy_config.external_url('batch-driver', pool_url_path))


async def job_complete_1(request, instance):
    body = await json_request(request)
    job_status = body['status']
    marked_job_started = body.get('marked_job_started', False)

    batch_id = job_status['batch_id']
    job_id = job_status['job_id']
    attempt_id = job_status['attempt_id']

    request['batch_telemetry']['batch_id'] = str(batch_id)
    request['batch_telemetry']['job_id'] = str(job_id)

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
        marked_job_started=marked_job_started,
    )

    await instance.mark_healthy()

    return web.Response()


@routes.post('/api/v1alpha/instances/job_complete')
@active_instances_only
@add_metadata_to_request
async def job_complete(request, instance):
    return await asyncio.shield(job_complete_1(request, instance))


async def job_started_1(request, instance):
    body = await json_request(request)
    job_status = body['status']

    batch_id = job_status['batch_id']
    job_id = job_status['job_id']
    attempt_id = job_status['attempt_id']
    start_time = job_status['start_time']
    resources = job_status.get('resources')

    request['batch_telemetry']['batch_id'] = str(batch_id)
    request['batch_telemetry']['job_id'] = str(job_id)

    await mark_job_started(request.app, batch_id, job_id, attempt_id, instance, start_time, resources)

    await instance.mark_healthy()

    return web.Response()


@routes.post('/api/v1alpha/instances/job_started')
@active_instances_only
@add_metadata_to_request
async def job_started(request, instance):
    return await asyncio.shield(job_started_1(request, instance))


async def billing_update_1(request, instance):
    db: Database = request.app['db']

    body = await json_request(request)
    update_timestamp = body['timestamp']
    running_attempts = body['attempts']

    if running_attempts:
        where_attempt_query = []
        where_attempt_args = []
        for attempt in running_attempts:
            where_attempt_query.append('(batch_id = %s AND job_id = %s AND attempt_id = %s)')
            where_attempt_args.append([attempt['batch_id'], attempt['job_id'], attempt['attempt_id']])

        where_query = f'WHERE {" OR ".join(where_attempt_query)}'
        where_args = [update_timestamp, *flatten(where_attempt_args)]

        await db.execute_update(
            f"""
UPDATE attempts
SET rollup_time = %s
{where_query};
""",
            where_args,
        )

    await instance.mark_healthy()

    return web.Response()


@routes.post('/api/v1alpha/billing_update')
@active_instances_only
@add_metadata_to_request
async def billing_update(request, instance):
    return await asyncio.shield(billing_update_1(request, instance))


@routes.get('/')
@routes.get('')
@auth.authenticated_developers_only()
async def get_index(request, userdata):
    app = request.app
    db: Database = app['db']
    inst_coll_manager: InstanceCollectionManager = app['driver'].inst_coll_manager
    jpim: JobPrivateInstanceManager = app['driver'].job_private_inst_manager

    ready_cores = await db.select_and_fetchone(
        """
SELECT CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu
FROM user_inst_coll_resources;
"""
    )
    ready_cores_mcpu = ready_cores['ready_cores_mcpu']

    page_context = {
        'pools': inst_coll_manager.pools.values(),
        'jpim': jpim,
        'instance_id': app['instance_id'],
        'global_total_n_instances': inst_coll_manager.global_total_n_instances,
        'global_total_cores_mcpu': inst_coll_manager.global_total_cores_mcpu,
        'global_live_n_instances': inst_coll_manager.global_live_n_instances,
        'global_live_cores_mcpu': inst_coll_manager.global_live_cores_mcpu,
        'global_n_instances_by_state': inst_coll_manager.global_n_instances_by_state,
        'global_cores_mcpu_by_state': inst_coll_manager.global_cores_mcpu_by_state,
        'global_schedulable_n_instances': inst_coll_manager.global_schedulable_n_instances,
        'global_schedulable_cores_mcpu': inst_coll_manager.global_schedulable_cores_mcpu,
        'global_schedulable_free_cores_mcpu': inst_coll_manager.global_schedulable_free_cores_mcpu,
        'instances': inst_coll_manager.name_instance.values(),
        'ready_cores_mcpu': ready_cores_mcpu,
        'frozen': app['frozen'],
        'feature_flags': app['feature_flags'],
    }
    return await render_template('batch-driver', request, userdata, 'index.html', page_context)


@routes.get('/quotas')
@auth.authenticated_developers_only()
async def get_quotas(request, userdata):
    if CLOUD != 'gcp':
        return await render_template('batch-driver', request, userdata, 'quotas.html', {"plot_json": None})

    data = request.app['driver'].get_quotas()

    regions = list(data.keys())
    new_data = []
    for region in regions:
        region_data = {'region': region}
        quotas_region_data = data[region]['quotas']
        for quota in quotas_region_data:
            if quota['metric'] in ['PREEMPTIBLE_CPUS', 'CPUS', 'SSD_TOTAL_GB', 'LOCAL_SSD_TOTAL_GB', 'DISKS_TOTAL_GB']:
                region_data.update({quota['metric']: {'limit': quota['limit'], 'usage': quota['usage']}})
        new_data.append(region_data)

    df = pd.DataFrame(new_data).set_index("region")

    fig = make_subplots(
        rows=len(df),
        cols=len(df.columns),
        specs=[[{"type": "indicator"} for _ in df.columns] for _ in df.index],
    )
    for r, (region, row) in enumerate(df.iterrows()):
        for c, measure in enumerate(row):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=measure['usage'],
                    title={"text": f"{region}--{df.columns[c]}"},
                    title_font_size=15,
                    gauge={
                        'axis': {
                            'range': [None, measure['limit']],
                            'tickwidth': 1,
                            'tickcolor': "darkblue",
                        },
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': measure['limit'],
                        },
                    },
                ),
                row=r + 1,
                col=c + 1,
            )

    fig.update_layout(
        paper_bgcolor="lavender",
        font={'color': "darkblue", 'family': "Arial"},
        margin={"l": 0, "r": 0, "t": 50, "b": 20},
        height=1150,
        width=2200,
        autosize=True,
        title_font_size=40,
        title_x=0.5,
    )

    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return await render_template('batch-driver', request, userdata, 'quotas.html', {"plot_json": plot_json})


class ConfigError(Exception):
    pass


def validate(session, name, value, predicate, description):
    if not predicate(value):
        set_message(session, f'{name} invalid: {value}.  Must be {description}.', 'error')
        raise ConfigError()
    return value


def validate_int(session, name, value, predicate, description):
    try:
        i = int(value)
    except ValueError as e:
        set_message(session, f'{name} invalid: {value}.  Must be an integer.', 'error')
        raise ConfigError() from e
    return validate(session, name, i, predicate, description)


@routes.post('/configure-feature-flags')
@auth.authenticated_developers_only()
async def configure_feature_flags(request: web.Request, _) -> NoReturn:
    app = request.app
    db: Database = app['db']
    post = await request.post()

    compact_billing_tables = 'compact_billing_tables' in post
    oms_agent = 'oms_agent' in post

    await db.execute_update(
        """
UPDATE feature_flags SET compact_billing_tables = %s, oms_agent = %s;
""",
        (compact_billing_tables, oms_agent),
    )

    row = await db.select_and_fetchone('SELECT * FROM feature_flags')
    app['feature_flags'] = row

    raise web.HTTPFound(deploy_config.external_url('batch-driver', '/'))


@routes.post('/config-update/pool/{pool}')
@auth.authenticated_developers_only()
async def pool_config_update(request: web.Request, _) -> NoReturn:
    app = request.app
    db: Database = app['db']
    inst_coll_manager: InstanceCollectionManager = app['driver'].inst_coll_manager

    session = await aiohttp_session.get_session(request)

    pool_name = request.match_info['pool']
    pool = inst_coll_manager.get_inst_coll(pool_name)
    pool_url_path = f'/inst_coll/pool/{pool_name}'

    try:
        if not isinstance(pool, Pool):
            set_message(session, f'Unknown pool {pool_name}.', 'error')
            raise ConfigError()

        post = await request.post()

        worker_type = pool.worker_type

        boot_disk_size_gb = validate_int(
            session,
            'Worker boot disk size',
            post['boot_disk_size_gb'],
            lambda v: v >= 10,
            'a positive integer greater than or equal to 10',
        )

        if pool.cloud == 'azure' and boot_disk_size_gb != 30:
            set_message(session, 'The boot disk size (GB) must be 30 in azure.', 'error')
            raise ConfigError()

        worker_local_ssd_data_disk = 'worker_local_ssd_data_disk' in post

        worker_external_ssd_data_disk_size_gb = validate_int(
            session,
            'Worker external SSD data disk size (in GB)',
            post['worker_external_ssd_data_disk_size_gb'],
            lambda v: v >= 0,
            'a nonnegative integer',
        )

        if not worker_local_ssd_data_disk and worker_external_ssd_data_disk_size_gb == 0:
            set_message(
                session,
                'Either the worker must use a local SSD or the external SSD data disk must be non-zero.',
                'error',
            )
            raise ConfigError()

        if worker_local_ssd_data_disk and worker_external_ssd_data_disk_size_gb > 0:
            set_message(
                session, 'Worker cannot both use local SSD and have a non-zero external SSD data disk.', 'error'
            )
            raise ConfigError()

        max_instances = validate_int(
            session, 'Max instances', post['max_instances'], lambda v: v >= 0, 'a non-negative integer'
        )

        max_live_instances = validate_int(
            session,
            'Max live instances',
            post['max_live_instances'],
            lambda v: 0 <= v <= max_instances,
            'a non-negative integer',
        )

        min_instances = validate_int(
            session,
            'Min instances',
            post['min_instances'],
            lambda v: 0 <= v <= max_live_instances,
            f'a non-negative integer less than or equal to max_live_instances {max_live_instances}',
        )

        possible_worker_cores = []
        for cores in possible_cores_from_worker_type(pool.cloud, worker_type):
            if not worker_local_ssd_data_disk:
                possible_worker_cores.append(cores)
                continue

            # disk storage for local ssd is proportional to the number of cores in azure
            data_disk_size_gb = local_ssd_size(pool.cloud, worker_type, cores)
            unreserved_disk_storage_gb = unreserved_worker_data_disk_size_gib(data_disk_size_gb, cores)
            if unreserved_disk_storage_gb >= 0:
                possible_worker_cores.append(cores)

        worker_cores = validate_int(
            session,
            f'{worker_type} worker cores',
            post['worker_cores'],
            lambda c: c in possible_worker_cores,
            f'one of {", ".join(str(c) for c in possible_worker_cores)}',
        )

        standing_worker_cores = validate_int(
            session,
            f'{worker_type} standing worker cores',
            post['standing_worker_cores'],
            lambda c: c in possible_worker_cores,
            f'one of {", ".join(str(c) for c in possible_worker_cores)}',
        )

        if not worker_local_ssd_data_disk:
            unreserved_disk_storage_gb = unreserved_worker_data_disk_size_gib(
                worker_external_ssd_data_disk_size_gb, worker_cores
            )
            if unreserved_disk_storage_gb < 0:
                min_disk_storage = worker_external_ssd_data_disk_size_gb - unreserved_disk_storage_gb
                set_message(session, f'External SSD must be at least {min_disk_storage} GB', 'error')
                raise ConfigError()

        max_new_instances_per_autoscaler_loop = validate_int(
            session,
            'Max instances per autoscaler loop',
            post['max_new_instances_per_autoscaler_loop'],
            lambda v: v > 0,
            'a positive integer',
        )

        autoscaler_loop_period_secs = validate_int(
            session,
            'Autoscaler loop period in seconds',
            post['autoscaler_loop_period_secs'],
            lambda v: v > 0,
            'a positive integer',
        )

        worker_max_idle_time_secs = validate_int(
            session,
            'Worker max idle time in seconds',
            post['worker_max_idle_time_secs'],
            lambda v: v > 0,
            'a positive integer',
        )

        standing_worker_max_idle_time_secs = validate_int(
            session,
            'Standing worker max idle time in seconds',
            post['standing_worker_max_idle_time_secs'],
            lambda v: v > 0,
            'a positive integer',
        )

        job_queue_scheduling_window_secs = validate_int(
            session,
            'Job queue scheduling window in seconds',
            post['job_queue_scheduling_window_secs'],
            lambda v: v > 0,
            'a positive integer',
        )

        proposed_pool_config = PoolConfig(
            name=pool_name,
            cloud=pool.cloud,
            worker_type=worker_type,
            worker_cores=worker_cores,
            worker_local_ssd_data_disk=worker_local_ssd_data_disk,
            worker_external_ssd_data_disk_size_gb=worker_external_ssd_data_disk_size_gb,
            standing_worker_cores=standing_worker_cores,
            boot_disk_size_gb=boot_disk_size_gb,
            min_instances=min_instances,
            max_instances=max_instances,
            max_live_instances=max_live_instances,
            preemptible=pool.preemptible,
            max_new_instances_per_autoscaler_loop=max_new_instances_per_autoscaler_loop,
            autoscaler_loop_period_secs=autoscaler_loop_period_secs,
            worker_max_idle_time_secs=worker_max_idle_time_secs,
            standing_worker_max_idle_time_secs=standing_worker_max_idle_time_secs,
            job_queue_scheduling_window_secs=job_queue_scheduling_window_secs,
        )

        current_client_pool_config = json.loads(str(post['_pool_config_json']))
        current_server_pool_config = pool.config()

        client_items = current_client_pool_config.items()
        server_items = current_server_pool_config.items()

        match = client_items == server_items
        if not match:
            set_message(
                session,
                'The pool config was stale; please re-enter config updates and try again',
                'error',
            )
            raise ConfigError()

        await proposed_pool_config.update_database(db)
        pool.configure(proposed_pool_config)

        set_message(session, f'Updated configuration for {pool}.', 'info')
    except ConfigError:
        pass
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception(f'error while updating pool configuration for {pool}')
        raise

    raise web.HTTPFound(deploy_config.external_url('batch-driver', pool_url_path))


@routes.post('/config-update/jpim')
@auth.authenticated_developers_only()
async def job_private_config_update(request: web.Request, _) -> NoReturn:
    app = request.app
    jpim: JobPrivateInstanceManager = app['driver'].job_private_inst_manager

    session = await aiohttp_session.get_session(request)

    url_path = '/inst_coll/jpim'

    post = await request.post()

    try:
        boot_disk_size_gb = validate_int(
            session,
            'Worker boot disk size',
            post['boot_disk_size_gb'],
            lambda v: v >= 10,
            'a positive integer greater than or equal to 10',
        )

        if jpim.cloud == 'azure' and boot_disk_size_gb != 30:
            set_message(session, 'The boot disk size (GB) must be 30 in azure.', 'error')
            raise ConfigError()

        max_instances = validate_int(
            session, 'Max instances', post['max_instances'], lambda v: v > 0, 'a positive integer'
        )

        max_live_instances = validate_int(
            session, 'Max live instances', post['max_live_instances'], lambda v: v > 0, 'a positive integer'
        )

        max_new_instances_per_autoscaler_loop = validate_int(
            session,
            'Max instances per autoscaler loop',
            post['max_new_instances_per_autoscaler_loop'],
            lambda v: v > 0,
            'a positive integer',
        )

        autoscaler_loop_period_secs = validate_int(
            session,
            'Autoscaler loop period in seconds',
            post['autoscaler_loop_period_secs'],
            lambda v: v > 0,
            'a positive integer',
        )

        worker_max_idle_time_secs = validate_int(
            session,
            'Worker max idle time in seconds',
            post['worker_max_idle_time_secs'],
            lambda v: v > 0,
            'a positive integer',
        )

        await jpim.configure(
            boot_disk_size_gb=boot_disk_size_gb,
            max_instances=max_instances,
            max_live_instances=max_live_instances,
            max_new_instances_per_autoscaler_loop=max_new_instances_per_autoscaler_loop,
            autoscaler_loop_period_secs=autoscaler_loop_period_secs,
            worker_max_idle_time_secs=worker_max_idle_time_secs,
        )

        set_message(session, f'Updated configuration for {jpim}.', 'info')
    except ConfigError:
        pass
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception(f'error while updating pool configuration for {jpim}')
        raise

    raise web.HTTPFound(deploy_config.external_url('batch-driver', url_path))


@routes.get('/inst_coll/pool/{pool}')
@auth.authenticated_developers_only()
async def get_pool(request, userdata):
    app = request.app
    inst_coll_manager: InstanceCollectionManager = app['driver'].inst_coll_manager

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

    ready_cores_mcpu = sum(record['ready_cores_mcpu'] for record in user_resources)

    pool_config_json = json.dumps(pool.config())

    page_context = {
        'pool': pool,
        'pool_config_json': pool_config_json,
        'instances': pool.name_instance.values(),
        'user_resources': user_resources,
        'ready_cores_mcpu': ready_cores_mcpu,
    }

    return await render_template('batch-driver', request, userdata, 'pool.html', page_context)


@routes.get('/inst_coll/jpim')
@auth.authenticated_developers_only()
async def get_job_private_inst_manager(request, userdata):
    app = request.app
    jpim: JobPrivateInstanceManager = app['driver'].job_private_inst_manager

    user_resources = await jpim.compute_fair_share()
    user_resources = sorted(
        user_resources.values(),
        key=lambda record: record['n_ready_jobs'] + record['n_creating_jobs'] + record['n_running_jobs'],
        reverse=True,
    )

    n_ready_jobs = sum(record['n_ready_jobs'] for record in user_resources)
    n_creating_jobs = sum(record['n_creating_jobs'] for record in user_resources)
    n_running_jobs = sum(record['n_running_jobs'] for record in user_resources)

    page_context = {
        'jpim': jpim,
        'instances': jpim.name_instance.values(),
        'user_resources': user_resources,
        'n_ready_jobs': n_ready_jobs,
        'n_creating_jobs': n_creating_jobs,
        'n_running_jobs': n_running_jobs,
    }

    return await render_template('batch-driver', request, userdata, 'job_private.html', page_context)


@routes.post('/freeze')
@auth.authenticated_developers_only()
async def freeze_batch(request: web.Request, _) -> NoReturn:
    app = request.app
    db: Database = app['db']
    session = await aiohttp_session.get_session(request)

    if app['frozen']:
        set_message(session, 'Batch is already frozen.', 'info')
        raise web.HTTPFound(deploy_config.external_url('batch-driver', '/'))

    await db.execute_update(
        """
UPDATE globals SET frozen = 1;
"""
    )

    app['frozen'] = True

    set_message(session, 'Froze all instance collections and batch submissions.', 'info')

    raise web.HTTPFound(deploy_config.external_url('batch-driver', '/'))


@routes.post('/unfreeze')
@auth.authenticated_developers_only()
async def unfreeze_batch(request: web.Request, _) -> NoReturn:
    app = request.app
    db: Database = app['db']
    session = await aiohttp_session.get_session(request)

    if not app['frozen']:
        set_message(session, 'Batch is already unfrozen.', 'info')
        raise web.HTTPFound(deploy_config.external_url('batch-driver', '/'))

    await db.execute_update(
        """
UPDATE globals SET frozen = 0;
"""
    )

    app['frozen'] = False

    set_message(session, 'Unfroze all instance collections and batch submissions.', 'info')

    raise web.HTTPFound(deploy_config.external_url('batch-driver', '/'))


@routes.get('/user_resources')
@auth.authenticated_developers_only()
async def get_user_resources(request, userdata):
    app = request.app
    db: Database = app['db']

    records = db.execute_and_fetchall(
        """
SELECT user,
  CAST(COALESCE(SUM(n_ready_jobs), 0) AS SIGNED) AS n_ready_jobs,
  CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu,
  CAST(COALESCE(SUM(n_running_jobs), 0) AS SIGNED) AS n_running_jobs,
  CAST(COALESCE(SUM(running_cores_mcpu), 0) AS SIGNED) AS running_cores_mcpu
FROM user_inst_coll_resources
GROUP BY user
HAVING n_ready_jobs + n_running_jobs > 0;
"""
    )

    user_resources = sorted(
        [record async for record in records],
        key=lambda record: record['ready_cores_mcpu'] + record['running_cores_mcpu'],
        reverse=True,
    )

    page_context = {'user_resources': user_resources}
    return await render_template('batch-driver', request, userdata, 'user_resources.html', page_context)


async def check_incremental(db):
    @transaction(db, read_only=True)
    async def check(tx):
        user_inst_coll_with_broken_resources = tx.execute_and_fetchall(
            """
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
      (jobs.always_run OR NOT (jobs.cancelled OR job_groups_cancelled.id IS NOT NULL)) AS runnable,
      (NOT jobs.always_run AND (jobs.cancelled OR job_groups_cancelled.id IS NOT NULL)) AS cancelled
    FROM batches
    INNER JOIN jobs ON batches.id = jobs.batch_id
    LEFT JOIN job_groups_cancelled ON batches.id = job_groups_cancelled.id
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
   OR expected_n_ready_jobs != 0
   OR expected_ready_cores_mcpu != 0
   OR expected_n_running_jobs != 0
   OR expected_running_cores_mcpu != 0
   OR expected_n_creating_jobs != 0
   OR expected_n_cancelled_ready_jobs != 0
   OR expected_n_cancelled_running_jobs != 0
   OR expected_n_cancelled_creating_jobs != 0
LOCK IN SHARE MODE;
"""
        )

        failures = [record async for record in user_inst_coll_with_broken_resources]
        if len(failures) > 0:
            raise ValueError(json.dumps(failures))

    await check()


async def check_resource_aggregation(db):
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
        result: Dict[str, Any] = {}
        for k, v in d.items():
            seqop(result, key_f(k), v)
        return result

    @transaction(db, read_only=True)
    async def check(tx):
        attempt_resources = tx.execute_and_fetchall(
            """
SELECT attempt_resources.batch_id, attempt_resources.job_id, attempt_resources.attempt_id,
  JSON_OBJECTAGG(resources.resource, quantity * GREATEST(COALESCE(rollup_time - start_time, 0), 0)) as resources
FROM attempt_resources
INNER JOIN attempts
ON attempts.batch_id = attempt_resources.batch_id AND
  attempts.job_id = attempt_resources.job_id AND
  attempts.attempt_id = attempt_resources.attempt_id
LEFT JOIN resources ON attempt_resources.resource_id = resources.resource_id
WHERE GREATEST(COALESCE(rollup_time - start_time, 0), 0) != 0
GROUP BY batch_id, job_id, attempt_id
LOCK IN SHARE MODE;
"""
        )

        agg_job_resources = tx.execute_and_fetchall(
            """
SELECT batch_id, job_id, JSON_OBJECTAGG(resource, `usage`) as resources
FROM aggregated_job_resources_v3
LEFT JOIN resources ON aggregated_job_resources_v3.resource_id = resources.resource_id
GROUP BY batch_id, job_id
LOCK IN SHARE MODE;
"""
        )

        agg_batch_resources = tx.execute_and_fetchall(
            """
SELECT batch_id, billing_project, JSON_OBJECTAGG(resource, `usage`) as resources
FROM (
  SELECT batch_id, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM aggregated_job_group_resources_v3
  GROUP BY batch_id, resource_id) AS t
LEFT JOIN resources ON t.resource_id = resources.resource_id
JOIN batches ON batches.id = t.batch_id
GROUP BY t.batch_id, billing_project
LOCK IN SHARE MODE;
"""
        )

        agg_billing_project_resources = tx.execute_and_fetchall(
            """
SELECT billing_project, JSON_OBJECTAGG(resource, `usage`) as resources
FROM (
  SELECT billing_project, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM aggregated_billing_project_user_resources_v3
  GROUP BY billing_project, resource_id) AS t
LEFT JOIN resources ON t.resource_id = resources.resource_id
GROUP BY t.billing_project
LOCK IN SHARE MODE;
"""
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

    await check()


async def _cancel_batch(app, batch_id):
    try:
        await cancel_batch_in_db(app['db'], batch_id)
    except BatchUserError as exc:
        log.info(f'cannot cancel batch because {exc.message}')
        return
    set_cancel_state_changed(app)


async def monitor_billing_limits(app):
    db: Database = app['db']

    records = await query_billing_projects_with_cost(db)
    for record in records:
        limit = record['limit']
        accrued_cost = record['accrued_cost']
        if limit is not None and accrued_cost >= limit:
            running_batches = db.execute_and_fetchall(
                """
SELECT id
FROM batches
WHERE billing_project = %s AND state = 'running';
""",
                (record['billing_project'],),
            )
            async for batch in running_batches:
                await _cancel_batch(app, batch['id'])


async def cancel_fast_failing_batches(app):
    db: Database = app['db']

    records = db.select_and_fetchall(
        """
SELECT batches.id, job_groups_n_jobs_in_complete_states.n_failed
FROM batches
LEFT JOIN job_groups_n_jobs_in_complete_states
  ON batches.id = job_groups_n_jobs_in_complete_states.id
WHERE state = 'running' AND cancel_after_n_failures IS NOT NULL AND n_failed >= cancel_after_n_failures
"""
    )
    async for batch in records:
        await _cancel_batch(app, batch['id'])


USER_CORES = pc.Gauge('batch_user_cores', 'Batch user cores (i.e. total in-use cores)', ['state', 'user', 'inst_coll'])
USER_JOBS = pc.Gauge('batch_user_jobs', 'Batch user jobs', ['state', 'user', 'inst_coll'])
ACTIVE_USER_INST_COLL_PAIRS: Set[Tuple[str, str]] = set()

FREE_CORES = pc.Gauge('batch_free_cores', 'Batch total free cores', ['inst_coll'])
FREE_SCHEDULABLE_CORES = pc.Gauge('batch_free_schedulable_cores', 'Batch total free cores', ['inst_coll'])
TOTAL_CORES = pc.Gauge('batch_total_cores', 'Batch total cores', ['inst_coll'])
COST_PER_HOUR = pc.Gauge('batch_cost_per_hour', 'Batch cost ($/hr)', ['measure', 'inst_coll'])
INSTANCES = pc.Gauge('batch_instances', 'Batch instances', ['inst_coll', 'state'])
INSTANCE_CORE_UTILIZATION = pc.Histogram(
    'batch_instance_core_utilization',
    'Batch per-instance percentage of revenue generating cores',
    ['inst_coll'],
    # Buckets were chosen to distinguish instances with:
    # - no jobs (<= 1/8 core in use)
    # - 1 1/4 core job
    # - 1 1/2 core job
    # - 1 core in use
    # - 2 cores in use,
    # - etc.
    #
    # NB: we conflate some utilizations, for example, using 1.25 cores and using 2 cores.
    buckets=[c / 16 for c in [1 / 8, 1 / 4, 1 / 2, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16]],
)


async def monitor_user_resources(app):
    global ACTIVE_USER_INST_COLL_PAIRS
    db: Database = app['db']

    records = db.select_and_fetchall(
        """
SELECT user, inst_coll,
  CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu,
  CAST(COALESCE(SUM(running_cores_mcpu), 0) AS SIGNED) AS running_cores_mcpu,
  CAST(COALESCE(SUM(n_ready_jobs), 0) AS SIGNED) AS n_ready_jobs,
  CAST(COALESCE(SUM(n_running_jobs), 0) AS SIGNED) AS n_running_jobs,
  CAST(COALESCE(SUM(n_creating_jobs), 0) AS SIGNED) AS n_creating_jobs
FROM user_inst_coll_resources
GROUP BY user, inst_coll;
"""
    )

    current_user_inst_coll_pairs: Set[Tuple[str, str]] = set()

    async for record in records:
        user = record['user']
        inst_coll = record['inst_coll']

        current_user_inst_coll_pairs.add((user, inst_coll))
        labels = {'user': user, 'inst_coll': inst_coll}

        USER_CORES.labels(state='ready', **labels).set(record['ready_cores_mcpu'] / 1000)
        USER_CORES.labels(state='running', **labels).set(record['running_cores_mcpu'] / 1000)
        USER_JOBS.labels(state='ready', **labels).set(record['n_ready_jobs'])
        USER_JOBS.labels(state='running', **labels).set(record['n_running_jobs'])
        USER_JOBS.labels(state='creating', **labels).set(record['n_creating_jobs'])

    for user, inst_coll in ACTIVE_USER_INST_COLL_PAIRS - current_user_inst_coll_pairs:
        USER_CORES.remove('ready', user, inst_coll)
        USER_CORES.remove('running', user, inst_coll)
        USER_JOBS.remove('ready', user, inst_coll)
        USER_JOBS.remove('running', user, inst_coll)
        USER_JOBS.remove('creating', user, inst_coll)

    ACTIVE_USER_INST_COLL_PAIRS = current_user_inst_coll_pairs


def monitor_instances(app) -> None:
    driver: CloudDriver = app['driver']
    inst_coll_manager = driver.inst_coll_manager
    resource_rates = driver.billing_manager.resource_rates

    for inst_coll in inst_coll_manager.name_inst_coll.values():
        total_free_schedulable_cores = 0.0
        total_free_cores = 0.0
        total_cores = 0.0
        total_cost_per_hour = 0.0
        total_revenue_per_hour = 0.0
        instances_by_state: Dict[str, int] = defaultdict(int)

        for instance in inst_coll.name_instance.values():
            if instance.state == 'active':
                total_free_schedulable_cores += instance.free_cores_mcpu_nonnegative / 1000
            if instance.state != 'deleted':
                total_free_cores += instance.free_cores_mcpu_nonnegative / 1000
                total_cores += instance.cores_mcpu / 1000
                total_cost_per_hour += instance.cost_per_hour(resource_rates)
                total_revenue_per_hour += instance.revenue_per_hour(resource_rates)
                INSTANCE_CORE_UTILIZATION.labels(inst_coll=inst_coll.name).observe(instance.percent_cores_used)

            instances_by_state[instance.state] += 1

        FREE_CORES.labels(inst_coll=inst_coll.name).set(total_free_cores)
        FREE_SCHEDULABLE_CORES.labels(inst_coll=inst_coll.name).set(total_free_schedulable_cores)
        TOTAL_CORES.labels(inst_coll=inst_coll.name).set(total_cores)
        COST_PER_HOUR.labels(inst_coll=inst_coll.name, measure='actual').set(total_cost_per_hour)
        COST_PER_HOUR.labels(inst_coll=inst_coll.name, measure='billed').set(total_revenue_per_hour)
        for state, count in instances_by_state.items():
            INSTANCES.labels(inst_coll=inst_coll.name, state=state).set(count)


async def monitor_system(app):
    await monitor_user_resources(app)
    monitor_instances(app)


async def compact_agg_billing_project_users_table(app, db: Database):
    if not app['feature_flags']['compact_billing_tables']:
        return

    @transaction(db)
    async def compact(tx: Transaction, target: dict):
        original_usage = await tx.execute_and_fetchone(
            """
SELECT CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
FROM aggregated_billing_project_user_resources_v3
WHERE billing_project = %s AND `user` = %s AND resource_id = %s
FOR UPDATE;
""",
            (target['billing_project'], target['user'], target['resource_id']),
        )

        await tx.just_execute(
            """
DELETE FROM aggregated_billing_project_user_resources_v3
WHERE billing_project = %s AND `user` = %s AND resource_id = %s;
""",
            (target['billing_project'], target['user'], target['resource_id']),
        )

        await tx.execute_update(
            """
INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, `user`, resource_id, token, `usage`)
VALUES (%s, %s, %s, %s, %s);
""",
            (
                target['billing_project'],
                target['user'],
                target['resource_id'],
                0,
                original_usage['usage'],
            ),
        )

        new_usage = await tx.execute_and_fetchone(
            """
SELECT CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
FROM aggregated_billing_project_user_resources_v3
WHERE billing_project = %s AND `user` = %s AND resource_id = %s
GROUP BY billing_project, `user`, resource_id;
""",
            (target['billing_project'], target['user'], target['resource_id']),
        )

        if new_usage['usage'] != original_usage['usage']:
            raise ValueError(
                f'problem in audit for {target}. original usage = {original_usage} but new usage is {new_usage}. aborting'
            )

    targets = db.execute_and_fetchall(
        """
SELECT billing_project, `user`, resource_id, COUNT(*) AS n_tokens
FROM aggregated_billing_project_user_resources_v3
WHERE token != 0
GROUP BY billing_project, `user`, resource_id
ORDER BY n_tokens DESC
LIMIT 10000;
""",
        query_name='find_agg_billing_project_user_resource_to_compact',
    )

    targets = [target async for target in targets]

    for target in targets:
        await compact(target)


async def compact_agg_billing_project_users_by_date_table(app, db: Database):
    if not app['feature_flags']['compact_billing_tables']:
        return

    @transaction(db)
    async def compact(tx: Transaction, target: dict):
        original_usage = await tx.execute_and_fetchone(
            """
SELECT CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
FROM aggregated_billing_project_user_resources_by_date_v3
WHERE billing_date = %s AND billing_project = %s AND `user` = %s AND resource_id = %s
FOR UPDATE;
""",
            (target['billing_date'], target['billing_project'], target['user'], target['resource_id']),
        )

        await tx.just_execute(
            """
DELETE FROM aggregated_billing_project_user_resources_by_date_v3
WHERE billing_date = %s AND billing_project = %s AND `user` = %s AND resource_id = %s;
""",
            (target['billing_date'], target['billing_project'], target['user'], target['resource_id']),
        )

        await tx.execute_update(
            """
INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, `user`, resource_id, token, `usage`)
VALUES (%s, %s, %s, %s, %s, %s);
""",
            (
                target['billing_date'],
                target['billing_project'],
                target['user'],
                target['resource_id'],
                0,
                original_usage['usage'],
            ),
        )

        new_usage = await tx.execute_and_fetchone(
            """
SELECT CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
FROM aggregated_billing_project_user_resources_by_date_v3
WHERE billing_date = %s AND billing_project = %s AND `user` = %s AND resource_id = %s
GROUP BY billing_date, billing_project, `user`, resource_id;
""",
            (target['billing_date'], target['billing_project'], target['user'], target['resource_id']),
        )

        if new_usage['usage'] != original_usage['usage']:
            raise ValueError(
                f'problem in audit for {target}. original usage = {original_usage} but new usage is {new_usage}. aborting'
            )

    targets = db.execute_and_fetchall(
        """
SELECT billing_date, billing_project, `user`, resource_id, COUNT(*) AS n_tokens
FROM aggregated_billing_project_user_resources_by_date_v3
WHERE token != 0
GROUP BY billing_date, billing_project, `user`, resource_id
ORDER BY n_tokens DESC
LIMIT 10000;
""",
        query_name='find_agg_billing_project_user_resource_by_date_to_compact',
    )

    targets = [target async for target in targets]

    for target in targets:
        await compact(target)


async def scheduling_cancelling_bump(app):
    log.info('scheduling cancelling bump loop')
    app['scheduler_state_changed'].notify()
    app['cancel_ready_state_changed'].set()
    app['cancel_creating_state_changed'].set()
    app['cancel_running_state_changed'].set()


Resource = namedtuple('Resource', ['resource_id', 'deduped_resource_id'])


async def refresh_globals_from_db(app, db):
    resource_ids = {
        record['resource']: Resource(record['resource_id'], record['deduped_resource_id'])
        async for record in db.select_and_fetchall(
            """
SELECT resource, resource_id, deduped_resource_id FROM resources;
"""
        )
    }

    app['resource_name_to_id'] = resource_ids


class BatchDriverAccessLogger(AccessLogger):
    def __init__(self, logger: logging.Logger, log_format: str):
        super().__init__(logger, log_format)
        if DEFAULT_NAMESPACE == 'default':
            self.exclude = [
                (endpoint[0], re.compile(deploy_config.base_path('batch-driver') + endpoint[1]))
                for endpoint in [
                    ('POST', '/api/v1alpha/instances/billing_update'),
                    ('POST', '/api/v1alpha/instances/job_complete'),
                    ('POST', '/api/v1alpha/instances/job_started'),
                    ('PATCH', '/api/v1alpha/batches/.*/.*/close'),
                    ('POST', '/api/v1alpha/batches/cancel'),
                    ('PATCH', '/api/v1alpha/batches/.*/.*/update'),
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
    exit_stack = AsyncExitStack()
    app['exit_stack'] = exit_stack

    kubernetes_asyncio.config.load_incluster_config()
    app['k8s_client'] = kubernetes_asyncio.client.CoreV1Api()
    app['k8s_cache'] = K8sCache(app['k8s_client'])

    async def close_and_wait():
        # - Following warning mitigation described here: https://github.com/aio-libs/aiohttp/pull/2045
        # - Fixed in aiohttp 4.0.0: https://github.com/aio-libs/aiohttp/issues/1925
        await app['k8s_client'].api_client.close()
        await asyncio.sleep(0.250)

    exit_stack.push_async_callback(close_and_wait)

    db = Database()
    await db.async_init(maxsize=50)
    app['db'] = db
    exit_stack.push_async_callback(app['db'].async_close)

    row = await db.select_and_fetchone(
        """
SELECT instance_id, frozen FROM globals;
"""
    )
    instance_id = row['instance_id']
    log.info(f'instance_id {instance_id}')
    app['instance_id'] = instance_id
    app['frozen'] = row['frozen']

    row = await db.select_and_fetchone('SELECT * FROM feature_flags')
    app['feature_flags'] = row

    await refresh_globals_from_db(app, db)

    app['scheduler_state_changed'] = Notice()
    app['cancel_ready_state_changed'] = asyncio.Event()
    app['cancel_creating_state_changed'] = asyncio.Event()
    app['cancel_running_state_changed'] = asyncio.Event()

    app['async_worker_pool'] = AsyncWorkerPool(100, queue_size=100)
    exit_stack.push_async_callback(app['async_worker_pool'].shutdown_and_wait)

    fs = get_cloud_async_fs()
    app['file_store'] = FileStore(fs, BATCH_STORAGE_URI, instance_id)
    exit_stack.push_async_callback(app['file_store'].close)

    inst_coll_configs = await InstanceCollectionConfigs.create(db)

    app['client_session'] = httpx.client_session()
    exit_stack.push_async_callback(app['client_session'].close)

    app['driver'] = await get_cloud_driver(app, db, MACHINE_NAME_PREFIX, DEFAULT_NAMESPACE, inst_coll_configs)
    exit_stack.push_async_callback(app['driver'].shutdown)

    app['canceller'] = await Canceller.create(app)
    exit_stack.push_async_callback(app['canceller'].shutdown_and_wait)

    task_manager = aiotools.BackgroundTaskManager()
    app['task_manager'] = task_manager
    exit_stack.push_async_callback(app['task_manager'].shutdown_and_wait)

    task_manager.ensure_future(periodically_call(10, monitor_billing_limits, app))
    task_manager.ensure_future(periodically_call(10, cancel_fast_failing_batches, app))
    task_manager.ensure_future(periodically_call(60, scheduling_cancelling_bump, app))
    task_manager.ensure_future(periodically_call(15, monitor_system, app))
    task_manager.ensure_future(periodically_call(5, refresh_globals_from_db, app, db))
    task_manager.ensure_future(periodically_call(60, compact_agg_billing_project_users_table, app, db))
    task_manager.ensure_future(periodically_call(60, compact_agg_billing_project_users_by_date_table, app, db))


async def on_cleanup(app):
    try:
        await app['exit_stack'].aclose()
    finally:
        await asyncio.gather(*(t for t in asyncio.all_tasks() if t is not asyncio.current_task()))


def run():
    install_profiler_if_requested('batch-driver')

    app = web.Application(
        client_max_size=HTTP_CLIENT_MAX_SIZE, middlewares=[check_csrf_token, monitor_endpoints_middleware]
    )
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
        port=int(os.environ['PORT']),
        access_log_class=BatchDriverAccessLogger,
    )
