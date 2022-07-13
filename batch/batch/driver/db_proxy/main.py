import asyncio
from aiohttp import web
import logging

import json

from hailtop import httpx
from hailtop.config import get_deploy_config
from hailtop.hail_logging import AccessLogger
from hailtop.tls import internal_server_ssl_context
from hailtop.utils import Notice, time_msecs
from gear import Database, setup_aiohttp_session

from ..job import add_attempt_resources, notify_batch_job_complete
from ...globals import complete_states

log = logging.getLogger('db-proxy')

deploy_config = get_deploy_config()

routes = web.RouteTableDef()


def instance_name_from_request(request) -> str:
    instance_name = request.headers.get('X-Hail-Instance-Name')
    if instance_name is None:
        raise ValueError(f'request is missing required header X-Hail-Instance-Name: {request}')
    return instance_name


async def mark_job_complete(
    app, batch_id, job_id, attempt_id, instance_name, new_state, status, start_time, end_time, reason, resources
):
    scheduler_state_changed: Notice = app['scheduler_state_changed']
    cancel_ready_state_changed: asyncio.Event = app['cancel_ready_state_changed']
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

    scheduler_state_changed.notify()
    cancel_ready_state_changed.set()

    if instance_name:
        delta_cores_mcpu = rv['delta_cores_mcpu']
        if delta_cores_mcpu != 0:
            async with client_session.post(
                f'https://batch-driver/api/v1alpha/instances/adjust_cores/{instance_name}',
                json={'delta_cores_mcpu': delta_cores_mcpu},
            ):
                pass

    await add_attempt_resources(db, batch_id, job_id, attempt_id, resources)

    if rv['rc'] != 0:
        log.info(f'mark_job_complete returned {rv} for job {id}')
        return

    old_state = rv['old_state']
    if old_state in complete_states:
        log.info(f'old_state {old_state} complete for job {id}, doing nothing')
        # already complete, do nothing
        return

    await notify_batch_job_complete(db, client_session, batch_id)


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


async def on_startup(app: web.Application):
    db = Database()
    await db.async_init(maxsize=50)
    app['db'] = db
    app['scheduler_state_changed'] = Notice()
    app['cancel_ready_state_changed'] = asyncio.Event()
    app['client_session'] = httpx.client_session()


def run():
    app = web.Application()
    setup_aiohttp_session(app)

    app.add_routes(routes)
    app.on_startup.append(on_startup)

    web.run_app(
        deploy_config.prefix_application(app, 'batch-db-proxy'),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=internal_server_ssl_context(),
    )
