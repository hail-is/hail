import os
import logging
import json
import mimetypes
import jinja2
import aiohttp
from aiohttp import web
import aiohttp_jinja2
from prometheus_async.aio.web import server_stats  # type: ignore

from hailtop.utils import time_msecs, secret_alnum_string
from hailtop.hail_logging import AccessLogger
from hailtop.tls import internal_server_ssl_context
from hailtop.config import get_deploy_config
from hailtop.aiocloud import aiogoogle
from hailtop import httpx
from gear import (
    Database,
    setup_aiohttp_session,
    web_authenticated_developers_only,
    check_csrf_token,
    new_csrf_token,
    monitor_endpoints_middleware,
)


# styling of embedded editor

BUCKET = os.environ['HAIL_ATGU_BUCKET']

log = logging.getLogger(__name__)

deploy_config = get_deploy_config()

routes = web.RouteTableDef()


def render_template(file):
    def wrap(f):
        async def wrapped(request, *args, **kwargs):
            if '_csrf' in request.cookies:
                csrf_token = request.cookies['_csrf']
            else:
                csrf_token = new_csrf_token()

            context = await f(request, *args, **kwargs)
            context['csrf_token'] = csrf_token
            context['base_path'] = deploy_config.base_path('atgu')

            response = aiohttp_jinja2.render_template(file, request, context)
            response.set_cookie('_csrf', csrf_token, domain=os.environ['HAIL_DOMAIN'], secure=True, httponly=True)
            return response

        return wrapped

    return wrap


def resource_record_to_dict(record):
    return {
        'id': record['id'],
        'time_created': record['time_created'],
        'title': record['title'],
        'description': record['description'],
        'contents': json.loads(record['contents']),
        'tags': record['tags'],
        'attachments': json.loads(record['attachments']),
        'time_updated': record['time_updated'],
    }


@routes.get('')
@routes.get('/')
@routes.get('/resources')
@web_authenticated_developers_only()
@render_template('resources.html')
async def get_resources(request, userdata):  # pylint: disable=unused-argument
    db = request.app['db']
    resources = [
        resource_record_to_dict(record)
        async for record in db.select_and_fetchall(
            '''
SELECT * FROM atgu_resources
ORDER BY time_created DESC;
'''
        )
    ]
    return {'resources': resources}


@routes.get('/resources/create')
@web_authenticated_developers_only()
@render_template('create_resource.html')
async def get_create_resource(request, userdata):  # pylint: disable=unused-argument
    return {}


@routes.post('/resources/create')
# this method has special content handling, can't call `request.post()`
# @check_csrf_token
@web_authenticated_developers_only(redirect=False)
async def post_create_resource(request, userdata):  # pylint: disable=unused-argument
    db = request.app['db']
    storage_client = request.app['storage_client']

    checked_csrf = False
    attachments = {}
    post = {}
    reader = aiohttp.MultipartReader(request.headers, request.content)
    while True:
        part = await reader.next()  # pylint: disable=not-callable
        if not part:
            break
        if part.name == '_csrf':
            # check csrf token
            # form fields are delivered in ordrer, the _csrf hidden field should appear first
            # https://stackoverflow.com/questions/7449861/multipart-upload-form-is-order-guaranteed
            token1 = request.cookies.get('_csrf')
            token2 = await part.text()
            if token1 is None or token2 is None or token1 != token2:
                log.info('request made with invalid csrf tokens')
                raise web.HTTPUnauthorized()
            checked_csrf = True
        elif part.name == 'file':
            if not checked_csrf:
                raise web.HTTPUnauthorized()
            filename = part.filename
            if not filename:
                continue
            attachment_id = secret_alnum_string()
            async with await storage_client.insert_object(BUCKET, f'atgu/attachments/{attachment_id}') as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    await f.write(chunk)
            attachments[attachment_id] = filename
        else:
            post[part.name] = await part.text()

    if not checked_csrf:
        raise web.HTTPUnauthorized()

    now = time_msecs()
    id = await db.execute_insertone(
        '''
INSERT INTO `atgu_resources` (`time_created`, `title`, `description`, `contents`, `tags`, `attachments`, `time_updated`)
VALUES (%s, %s, %s, %s, %s, %s, %s);
''',
        (now, post['title'], post['description'], post['contents'], post['tags'], json.dumps(attachments), now),
    )

    return web.HTTPFound(deploy_config.external_url('atgu', f'/resources/{id}'))


@routes.get('/resources/{id}')
@web_authenticated_developers_only()
@render_template('resource.html')
async def get_resource(request, userdata):  # pylint: disable=unused-argument
    db = request.app['db']
    id = int(request.match_info['id'])
    record = await db.select_and_fetchone(
        '''
SELECT * FROM atgu_resources
WHERE id = %s;
''',
        (id),
    )
    if not record:
        raise web.HTTPNotFound()
    return {'resource': resource_record_to_dict(record)}


@routes.get('/resources/{id}/edit')
@web_authenticated_developers_only()
@render_template('edit_resource.html')
async def get_edit_resource(request, userdata):  # pylint: disable=unused-argument
    db = request.app['db']
    id = int(request.match_info['id'])
    record = await db.select_and_fetchone(
        '''
SELECT * FROM atgu_resources
WHERE id = %s;
''',
        (id),
    )
    if not record:
        raise web.HTTPNotFound()
    return {'resource': resource_record_to_dict(record)}


@routes.post('/resources/{id}/edit')
# this method has special content handling, can't call `request.post()`
# @check_csrf_token
@web_authenticated_developers_only(redirect=False)
async def post_edit_resource(request, userdata):  # pylint: disable=unused-argument
    db = request.app['db']
    storage_client = request.app['storage_client']
    id = int(request.match_info['id'])
    old_record = await db.select_and_fetchone(
        '''
SELECT attachments FROM atgu_resources
WHERE id = %s;
''',
        (id),
    )
    if not old_record:
        raise web.HTTPNotFound()

    old_attachments = json.loads(old_record['attachments'])

    checked_csrf = False
    attachments = {}
    post = {}
    reader = aiohttp.MultipartReader(request.headers, request.content)
    while True:
        part = await reader.next()  # pylint: disable=not-callable
        if not part:
            break
        if part.name == '_csrf':
            # check csrf token
            token1 = request.cookies.get('_csrf')
            token2 = await part.text()
            if token1 is None or token2 is None or token1 != token2:
                log.info('request made with invalid csrf tokens')
                raise web.HTTPUnauthorized()
            checked_csrf = True
        elif part.name == 'attachment':
            if not checked_csrf:
                raise web.HTTPUnauthorized()
            attachment_id = await part.text()
            assert attachment_id in old_attachments
            attachments[attachment_id] = old_attachments[attachment_id]
        elif part.name == 'file':
            filename = part.filename
            if not filename:
                continue
            attachment_id = secret_alnum_string()
            async with await storage_client.insert_object(BUCKET, f'atgu/attachments/{attachment_id}') as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    await f.write(chunk)
            attachments[attachment_id] = filename
        else:
            post[part.name] = await part.text()

    if not checked_csrf:
        raise web.HTTPUnauthorized()

    now = time_msecs()
    await db.execute_update(
        '''
UPDATE atgu_resources SET
title = %s,
description = %s,
contents = %s,
tags = %s,
attachments = %s,
time_updated = %s
WHERE id = %s
''',
        (post['title'], post['description'], post['contents'], post['tags'], json.dumps(attachments), now, id),
    )

    return web.HTTPFound(deploy_config.external_url('atgu', f'/resources/{id}'))


@routes.post('/resources/{id}/delete')
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
async def post_delete_resource(request, userdata):  # pylint: disable=unused-argument
    db = request.app['db']
    storage_client = request.app['storage_client']
    id = int(request.match_info['id'])

    record = await db.select_and_fetchone(
        '''
SELECT * FROM atgu_resources
WHERE id = %s;
''',
        (id),
    )
    if not record:
        raise web.HTTPNotFound()
    resource = resource_record_to_dict(record)

    await db.just_execute(
        '''
DELETE FROM `atgu_resources`
WHERE id = %s;
''',
        (id,),
    )

    for attachment_id in resource['attachments']:
        try:
            await storage_client.delete_object(BUCKET, f'atgu/attachments/{attachment_id}')
        except aiohttp.ClientResponseError as exc:
            if exc.status == 404:
                pass
            raise

    return web.HTTPFound(deploy_config.external_url('atgu', '/resources'))


@routes.get('/resources/{resource_id}/attachments/{attachment_id}')
@web_authenticated_developers_only()
async def get_attachment(request, userdata):  # pylint: disable=unused-argument
    db = request.app['db']
    storage_client = request.app['storage_client']
    resource_id = int(request.match_info['resource_id'])
    record = await db.select_and_fetchone(
        '''
SELECT attachments FROM atgu_resources
WHERE id = %s;
''',
        (resource_id),
    )
    if not record:
        raise web.HTTPNotFound()

    attachments = json.loads(record['attachments'])
    attachment_id = request.match_info['attachment_id']
    if attachment_id not in attachments:
        raise web.HTTPNotFound()

    filename = attachments[attachment_id]

    ct, encoding = mimetypes.guess_type(filename)
    if not ct:
        ct = 'application/octet-stream'

    headers = {'Content-Disposition': f'attachment; filename="{filename}"', 'Content-Type': ct}
    if encoding:
        headers['Content-Encoding'] = encoding

    resp = web.StreamResponse(status=200, reason='OK', headers=headers)
    await resp.prepare(request)

    async with await storage_client.get_object(BUCKET, f'atgu/attachments/{attachment_id}') as f:
        while True:
            b = await f.read(8 * 1024)
            if not b:
                break
            await resp.write(b)
    await resp.write_eof()

    return resp


async def on_startup(app):
    db = Database()
    await db.async_init()
    app['db'] = db
    app['storage_client'] = aiogoogle.GoogleStorageClient()
    app['client_session'] = httpx.client_session()


async def on_cleanup(app):
    try:
        await app['storage_client'].close()
    finally:
        try:
            await app['client_session'].close()
        finally:
            await app['db'].async_close()


def run():
    app = web.Application(middlewares=[monitor_endpoints_middleware])

    setup_aiohttp_session(app)

    aiohttp_jinja2.setup(app, loader=jinja2.ChoiceLoader([jinja2.PackageLoader('atgu')]))

    app.add_routes(routes)
    app.router.add_get("/metrics", server_stats)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    web.run_app(
        deploy_config.prefix_application(app, 'atgu'),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=internal_server_ssl_context(),
    )
