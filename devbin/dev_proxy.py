import os

from hailtop import httpx
from hailtop.config import get_deploy_config
from hailtop.auth import hail_credentials
from hailtop.aiocloud.common import Session

from web_common import render_template, setup_aiohttp_jinja2, setup_common_static_routes, web_security_headers

from aiohttp import web
import aiohttp_session
from aiohttp_session.cookie_storage import EncryptedCookieStorage

SERVICE = os.environ['SERVICE']

deploy_config = get_deploy_config()

routes = web.RouteTableDef()
setup_common_static_routes(routes)

STATIC_DIRS: dict[str, list[tuple[str, str]]] = {
    'ci': [('/ci/static/compiled-js', 'ci/ci/static/compiled-js')],
}
for _path, _directory in STATIC_DIRS.get(SERVICE, []):
    routes.static(_path, _directory)
IS_DEVELOPER = bool(os.getenv('IS_DEVELOPER', True))
MODULES = {'batch': 'batch.front_end', 'batch-driver': 'batch.driver'}
BC = web.AppKey('backend_client', Session)

@routes.view('/api/{route:.*}')
@web_security_headers
async def default_proxied_api_route(request: web.Request):
    return web.json_response(await proxy(request))


if SERVICE == 'ci':

    @routes.get('/flaky_tests')
    @web_security_headers
    async def flaky_tests_local(request: web.Request):
        fake_userdata = {'username': 'dev', 'is_developer': IS_DEVELOPER}
        return await render_template('ci', request, fake_userdata, 'flaky_tests.html', {})

    if os.getenv('MOCK_API_DATA'):

        @routes.get('/api/v1alpha/retried_tests')
        async def retried_tests_mock(request: web.Request):
            rows = [
                {'id': 120, 'batch_id': 9001, 'job_id': 3, 'job_name': 'test_hail_python_1', 'state': 'Success', 'exit_code': 0, 'pr_number': 15310, 'target_branch': 'main', 'source_branch': 'fix-something', 'source_sha': 'aabbcc1', 'retried_by': 'ci', 'retried_at': '2026-03-10T11:03:00'},
                {'id': 119, 'batch_id': 9001, 'job_id': 3, 'job_name': 'test_hail_python_1', 'state': 'Failed',  'exit_code': 1, 'pr_number': 15308, 'target_branch': 'main', 'source_branch': 'other-fix',    'source_sha': 'aabbcc2', 'retried_by': 'ci', 'retried_at': '2026-03-10T09:45:00'},
                {'id': 118, 'batch_id': 8990, 'job_id': 5, 'job_name': 'test_hail_python_5', 'state': 'Success', 'exit_code': 0, 'pr_number': 15310, 'target_branch': 'main', 'source_branch': 'fix-something', 'source_sha': 'aabbcc1', 'retried_by': 'ci', 'retried_at': '2026-03-10T08:11:00'},
                {'id': 117, 'batch_id': 8980, 'job_id': 2, 'job_name': 'test_hail_python_unchecked_allocator_3', 'state': 'Failed', 'exit_code': 1, 'pr_number': 15305, 'target_branch': 'main', 'source_branch': 'perf-work', 'source_sha': 'ddeeff3', 'retried_by': 'ci', 'retried_at': '2026-03-09T22:45:00'},
                {'id': 116, 'batch_id': 8970, 'job_id': 3, 'job_name': 'test_hail_python_1', 'state': 'Success', 'exit_code': 0, 'pr_number': 15301, 'target_branch': 'main', 'source_branch': 'my-branch',    'source_sha': 'ddeeff4', 'retried_by': 'ci', 'retried_at': '2026-03-09T14:32:00'},
                {'id': 115, 'batch_id': 8960, 'job_id': 4, 'job_name': 'test_hail_python_2', 'state': 'Success', 'exit_code': 0, 'pr_number': 15308, 'target_branch': 'main', 'source_branch': 'other-fix',    'source_sha': 'aabbcc2', 'retried_by': 'ci', 'retried_at': '2026-03-09T10:03:00'},
                {'id': 114, 'batch_id': 8950, 'job_id': 3, 'job_name': 'test_hail_python_1', 'state': 'Failed',  'exit_code': 1, 'pr_number': 15299, 'target_branch': 'main', 'source_branch': 'old-branch',    'source_sha': 'aabbcc5', 'retried_by': 'ci', 'retried_at': '2026-03-08T18:20:00'},
                {'id': 113, 'batch_id': 8940, 'job_id': 6, 'job_name': 'test_hail_python_unchecked_allocator_3', 'state': 'Success', 'exit_code': 0, 'pr_number': 15301, 'target_branch': 'main', 'source_branch': 'my-branch', 'source_sha': 'ddeeff4', 'retried_by': 'ci', 'retried_at': '2026-03-08T12:10:00'},
                {'id': 112, 'batch_id': 8930, 'job_id': 2, 'job_name': 'test_hail_python_5', 'state': 'Failed',  'exit_code': 1, 'pr_number': 15295, 'target_branch': 'main', 'source_branch': 'wip',           'source_sha': 'aabbcc6', 'retried_by': 'ci', 'retried_at': '2026-03-07T16:20:00'},
                {'id': 111, 'batch_id': 8920, 'job_id': 1, 'job_name': 'test_batch_2',        'state': 'Success', 'exit_code': 0, 'pr_number': 15290, 'target_branch': 'main', 'source_branch': 'cleanup',       'source_sha': 'aabbcc7', 'retried_by': 'ci', 'retried_at': '2026-03-07T09:55:00'},
            ]
            return web.json_response({'rows': rows, 'cursor': None, 'has_more': False})


@routes.view('/{route:.*}')
@web_security_headers
async def default_proxied_web_route(request: web.Request):
    return await render_html(request, await proxy(request))


async def proxy(request: web.Request):
    backend_client = request.app[BC]
    backend_route = deploy_config.external_url(SERVICE, request.raw_path)
    headers = {'x-hail-return-jinja-context': '1'}
    try:
        async with await backend_client.request(request.method, backend_route, headers=headers) as resp:
            return await resp.json()
    except httpx.ClientResponseError as e:
        if e.status == 404:
            raise web.HTTPNotFound()
        raise


async def render_html(request: web.Request, context: dict):
    # Make links point back to the local dev server and not use
    # the dev namespace path rewrite shenanigans.
    context['page_context']['base_path'] = ''
    context['page_context']['is_developer'] = IS_DEVELOPER
    context['userdata']['is_developer'] = IS_DEVELOPER
    return await render_template(SERVICE, request, **context)


async def on_startup(app: web.Application):
    app[BC] = Session(credentials=hail_credentials())


async def on_cleanup(app: web.Application):
    await app[BC].close()


app = web.Application()
setup_aiohttp_jinja2(app, MODULES.get(SERVICE, SERVICE))
app.add_routes(routes)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)
aiohttp_session.setup(app, EncryptedCookieStorage(b'Thirty  two  length  bytes  key.'))

if __name__ == '__main__':
    web.run_app(app)
