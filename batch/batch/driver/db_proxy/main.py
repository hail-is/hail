import asyncio
from aiohttp import web
import logging
from typing import Dict

import collections

import json

from hailtop import aiotools, httpx
from hailtop.config import get_deploy_config
from hailtop.hail_logging import AccessLogger
from hailtop.utils import time_msecs, periodically_call
from gear import Database, setup_aiohttp_session, monitor_endpoints_middleware

import googlecloudprofiler
from prometheus_async.aio.web import server_stats

from ..job import add_attempt_resources, notify_batch_job_complete
from ...globals import complete_states, HTTP_CLIENT_MAX_SIZE

log = logging.getLogger('db-proxy')

deploy_config = get_deploy_config()

routes = web.RouteTableDef()

OPEN_CORES: Dict[str, int] = collections.defaultdict(int)


def instance_name_from_request(request) -> str:
    instance_name = request.headers.get('X-Hail-Instance-Name')
    if instance_name is None:
        raise ValueError(f'request is missing required header X-Hail-Instance-Name: {request}')
    return instance_name


async def notify_driver_open_cores(app):
    client_session: httpx.ClientSession = app['client_session']
    if OPEN_CORES:
        cores_to_post = OPEN_CORES.copy()
        OPEN_CORES.clear()
        async with client_session.post(
            deploy_config.url('batch-driver', f'/api/v1alpha/instances/adjust_cores'),
            json={'open_cores': cores_to_post},
        ):
            pass


async def mark_job_complete(
    app, batch_id, job_id, attempt_id, instance_name, new_state, status, start_time, end_time, reason, resources
):
    db: Database = app['db']
    client_session: httpx.ClientSession = app['client_session']

    id = (batch_id, job_id)

    log.info(f'marking job {id} complete new_state {new_state}')

    now = time_msecs()

    try:
        rv = await db.execute_and_fetchone(
            'CALL mark_job_complete(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);',
            (
                batch_id,
                job_id,
                attempt_id,
                instance_name,
                new_state,
                json.dumps(status) if status is not None else None,
                start_time,
                end_time,
                reason,
                now,
            ),
            'mark_job_complete',
        )
    except Exception:
        log.exception(f'error while marking job {id} complete on instance {instance_name}')
        raise

    if instance_name:
        delta_cores_mcpu = rv['delta_cores_mcpu']
        if delta_cores_mcpu != 0:
            OPEN_CORES[instance_name] += delta_cores_mcpu

    await add_attempt_resources(db, batch_id, job_id, attempt_id, resources)

    if rv['rc'] != 0:
        log.info(f'mark_job_complete returned {rv} for job {id}')
        return

    old_state = rv['old_state']
    if old_state in complete_states:
        log.info(f'old_state {old_state} complete for job {id}, doing nothing')
        # already complete, do nothing
        return

    asyncio.ensure_future(notify_batch_job_complete(db, client_session, batch_id))


async def job_complete_1(request):
    body = await request.json()

    instance_name = instance_name_from_request(request)
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
        instance_name,
        new_state,
        status,
        start_time,
        end_time,
        'completed',
        resources,
    )

    return web.Response()


@routes.post('/api/v1alpha/instances/job_complete')
async def job_complete(request):
    return await asyncio.shield(job_complete_1(request))


async def mark_job_started(app, batch_id, job_id, attempt_id, instance_name, start_time, resources):
    db: Database = app['db']

    id = (batch_id, job_id)

    log.info(f'mark job {id} started')

    try:
        rv = await db.execute_and_fetchone(
            '''
CALL mark_job_started(%s, %s, %s, %s, %s);
''',
            (batch_id, job_id, attempt_id, instance_name, start_time),
            'mark_job_started',
        )
    except Exception:
        log.info(f'error while marking job {id} started on {instance_name}')
        raise

    if rv['delta_cores_mcpu'] != 0:
        OPEN_CORES[instance_name] += rv['delta_cores_mcpu']

    await add_attempt_resources(db, batch_id, job_id, attempt_id, resources)


async def job_started_1(request):
    body = await request.json()
    job_status = body['status']

    batch_id = job_status['batch_id']
    job_id = job_status['job_id']
    attempt_id = job_status['attempt_id']
    start_time = job_status['start_time']
    resources = job_status.get('resources')
    instance_name = instance_name_from_request(request)

    await mark_job_started(request.app, batch_id, job_id, attempt_id, instance_name, start_time, resources)
    return web.Response()


@routes.post('/api/v1alpha/instances/job_started')
async def job_started(request):
    return await asyncio.shield(job_started_1(request))


async def on_startup(app: web.Application):
    db = Database()
    await db.async_init(maxsize=50)
    app['db'] = db
    app['client_session'] = httpx.client_session()

    app['task_manager'] = aiotools.BackgroundTaskManager()
    app['task_manager'].ensure_future(periodically_call(0.1, notify_driver_open_cores, app))


def run():
    profiler_tag = 'dgoldste'
    googlecloudprofiler.start(
        service='batch-db-proxy',
        service_version=profiler_tag,
        # https://cloud.google.com/profiler/docs/profiling-python#agent_logging
        verbose=3,
    )
    app = web.Application(client_max_size=HTTP_CLIENT_MAX_SIZE, middlewares=[monitor_endpoints_middleware])
    setup_aiohttp_session(app)

    app.add_routes(routes)
    app.on_startup.append(on_startup)
    app.router.add_get("/metrics", server_stats)

    web.run_app(
        deploy_config.prefix_application(app, 'batch-db-proxy'),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
    )
