import asyncio
from aiohttp import web
import logging

from hailtop.config import get_deploy_config
from hailtop.hail_logging import AccessLogger
from hailtop.tls import internal_server_ssl_context
from gear import Database, setup_aiohttp_session

from ..job import mark_job_complete

log = logging.getLogger('db-proxy')

deploy_config = get_deploy_config()

routes = web.RouteTableDef()


def instance_name_from_request(request) -> str:
    instance_name = request.headers.get('X-Hail-Instance-Name')
    if instance_name is None:
        raise ValueError(f'request is missing required header X-Hail-Instance-Name: {request}')
    return instance_name


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
