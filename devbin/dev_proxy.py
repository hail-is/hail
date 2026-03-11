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
                {'id': 120, 'batch_id': 9001, 'job_id': 3, 'job_name': 'test_hail_python_1',                    'state': 'Failed', 'exit_code': 1, 'pr_number': 15310, 'target_branch': 'main', 'source_branch': 'fix-something', 'source_sha': 'aabbcc1', 'retried_by': 'ci', 'retried_at': '2026-03-10T11:03:00'},
                {'id': 119, 'batch_id': 9001, 'job_id': 3, 'job_name': 'test_hail_python_1',                    'state': 'Failed',  'exit_code': 1, 'pr_number': 15308, 'target_branch': 'main', 'source_branch': 'other-fix',    'source_sha': 'aabbcc2', 'retried_by': 'ci', 'retried_at': '2026-03-10T09:45:00'},
                {'id': 118, 'batch_id': 8990, 'job_id': 5, 'job_name': 'test_hail_python_5',                    'state': 'Error', 'exit_code': None, 'pr_number': 15310, 'target_branch': 'main', 'source_branch': 'fix-something', 'source_sha': 'aabbcc1', 'retried_by': 'ci', 'retried_at': '2026-03-10T08:11:00'},
                {'id': 117, 'batch_id': 8980, 'job_id': 2, 'job_name': 'test_hail_python_unchecked_allocator_3','state': 'Failed',  'exit_code': 1, 'pr_number': 15305, 'target_branch': 'main', 'source_branch': 'perf-work',    'source_sha': 'ddeeff3', 'retried_by': 'turbo_tester', 'retried_at': '2026-03-09T22:45:00'},
                {'id': 116, 'batch_id': 8970, 'job_id': 3, 'job_name': 'test_hail_python_1',                    'state': 'Failed', 'exit_code': 1, 'pr_number': 15301, 'target_branch': 'main', 'source_branch': 'my-branch',    'source_sha': 'ddeeff4', 'retried_by': 'ci', 'retried_at': '2026-03-09T14:32:00'},
                {'id': 115, 'batch_id': 8960, 'job_id': 4, 'job_name': 'test_hail_python_2',                    'state': 'Error', 'exit_code': None, 'pr_number': 15308, 'target_branch': 'main', 'source_branch': 'other-fix',    'source_sha': 'aabbcc2', 'retried_by': 'ci', 'retried_at': '2026-03-09T10:03:00'},
                {'id': 114, 'batch_id': 8950, 'job_id': 3, 'job_name': 'test_hail_python_1',                    'state': 'Failed',  'exit_code': 1, 'pr_number': 15299, 'target_branch': 'main', 'source_branch': 'old-branch',   'source_sha': 'aabbcc5', 'retried_by': 'flaky_mcflakeface', 'retried_at': '2026-03-08T18:20:00'},
                {'id': 113, 'batch_id': 8940, 'job_id': 6, 'job_name': 'test_hail_python_unchecked_allocator_3','state': 'Error', 'exit_code': None, 'pr_number': 15301, 'target_branch': 'main', 'source_branch': 'my-branch',    'source_sha': 'ddeeff4', 'retried_by': 'ci', 'retried_at': '2026-03-08T12:10:00'},
                {'id': 112, 'batch_id': 8930, 'job_id': 2, 'job_name': 'test_hail_python_5',                    'state': 'Failed',  'exit_code': 1, 'pr_number': 15295, 'target_branch': 'main', 'source_branch': 'wip',          'source_sha': 'aabbcc6', 'retried_by': 'ci', 'retried_at': '2026-03-07T16:20:00'},
                {'id': 111, 'batch_id': 8920, 'job_id': 1, 'job_name': 'test_batch_2',                          'state': 'Failed', 'exit_code': 1, 'pr_number': 15290, 'target_branch': 'main', 'source_branch': 'cleanup',      'source_sha': 'aabbcc7', 'retried_by': 'ci', 'retried_at': '2026-03-07T09:55:00'},
                # states: ~18 Failed, ~8 Error across 26 rows
                {'id': 110, 'batch_id': 8910, 'job_id': 3, 'job_name': 'test_hail_python_1',                    'state': 'Failed',  'exit_code': 1, 'pr_number': 15287, 'target_branch': 'main', 'source_branch': 'feat-a',      'source_sha': 'aa0001', 'retried_by': 'turbo_tester', 'retried_at': '2026-03-06T21:10:00'},
                {'id': 109, 'batch_id': 8900, 'job_id': 3, 'job_name': 'test_hail_python_unchecked_allocator_3','state': 'Failed',  'exit_code': 1, 'pr_number': 15285, 'target_branch': 'main', 'source_branch': 'feat-b',      'source_sha': 'aa0002', 'retried_by': 'ci', 'retried_at': '2026-03-06T18:30:00'},
                {'id': 108, 'batch_id': 8890, 'job_id': 5, 'job_name': 'test_hail_python_5',                    'state': 'Failed', 'exit_code': 1, 'pr_number': 15283, 'target_branch': 'main', 'source_branch': 'feat-c',      'source_sha': 'aa0003', 'retried_by': 'ci', 'retried_at': '2026-03-06T14:05:00'},
                {'id': 107, 'batch_id': 8880, 'job_id': 3, 'job_name': 'test_hail_python_1',                    'state': 'Failed', 'exit_code': 1, 'pr_number': 15281, 'target_branch': 'main', 'source_branch': 'bugfix-1',    'source_sha': 'aa0004', 'retried_by': 'ci', 'retried_at': '2026-03-06T09:55:00'},
                {'id': 106, 'batch_id': 8870, 'job_id': 2, 'job_name': 'test_hail_python_unchecked_allocator_3','state': 'Error', 'exit_code': None, 'pr_number': 15279, 'target_branch': 'main', 'source_branch': 'bugfix-2',    'source_sha': 'aa0005', 'retried_by': 'retry_queen', 'retried_at': '2026-03-05T22:40:00'},
                {'id': 105, 'batch_id': 8860, 'job_id': 3, 'job_name': 'test_hail_python_1',                    'state': 'Failed',  'exit_code': 1, 'pr_number': 15277, 'target_branch': 'main', 'source_branch': 'refactor-x',  'source_sha': 'aa0006', 'retried_by': 'ci', 'retried_at': '2026-03-05T16:15:00'},
                {'id': 104, 'batch_id': 8850, 'job_id': 2, 'job_name': 'test_hail_python_unchecked_allocator_3','state': 'Failed',  'exit_code': 1, 'pr_number': 15275, 'target_branch': 'main', 'source_branch': 'refactor-y',  'source_sha': 'aa0007', 'retried_by': 'flaky_mcflakeface', 'retried_at': '2026-03-05T11:30:00'},
                {'id': 103, 'batch_id': 8840, 'job_id': 3, 'job_name': 'test_hail_python_1',                    'state': 'Error', 'exit_code': None, 'pr_number': 15272, 'target_branch': 'main', 'source_branch': 'opt-pass',    'source_sha': 'aa0008', 'retried_by': 'ci', 'retried_at': '2026-03-04T20:00:00'},
                {'id': 102, 'batch_id': 8830, 'job_id': 4, 'job_name': 'test_hail_python_5',                    'state': 'Failed',  'exit_code': 1, 'pr_number': 15270, 'target_branch': 'main', 'source_branch': 'opt-pass',    'source_sha': 'aa0008', 'retried_by': 'ci', 'retried_at': '2026-03-04T15:45:00'},
                {'id': 101, 'batch_id': 8820, 'job_id': 3, 'job_name': 'test_hail_python_1',                    'state': 'Failed',  'exit_code': 1, 'pr_number': 15268, 'target_branch': 'main', 'source_branch': 'type-fixes',  'source_sha': 'aa0009', 'retried_by': 'turbo_tester', 'retried_at': '2026-03-04T10:20:00'},
                {'id': 100, 'batch_id': 8810, 'job_id': 2, 'job_name': 'test_hail_python_unchecked_allocator_3','state': 'Error', 'exit_code': None, 'pr_number': 15265, 'target_branch': 'main', 'source_branch': 'type-fixes',  'source_sha': 'aa0009', 'retried_by': 'ci', 'retried_at': '2026-03-03T19:10:00'},
                {'id':  99, 'batch_id': 8800, 'job_id': 3, 'job_name': 'test_hail_python_1',                    'state': 'Failed', 'exit_code': 1, 'pr_number': 15263, 'target_branch': 'main', 'source_branch': 'schema-v2',   'source_sha': 'aa0010', 'retried_by': 'ci', 'retried_at': '2026-03-03T14:05:00'},
                {'id':  98, 'batch_id': 8790, 'job_id': 2, 'job_name': 'test_hail_python_unchecked_allocator_3','state': 'Failed',  'exit_code': 1, 'pr_number': 15261, 'target_branch': 'main', 'source_branch': 'schema-v2',   'source_sha': 'aa0010', 'retried_by': 'debugosaurus', 'retried_at': '2026-03-02T22:30:00'},
                {'id':  97, 'batch_id': 8780, 'job_id': 3, 'job_name': 'test_hail_python_1',                    'state': 'Failed',  'exit_code': 1, 'pr_number': 15258, 'target_branch': 'main', 'source_branch': 'cleanup-2',   'source_sha': 'aa0011', 'retried_by': 'retry_queen', 'retried_at': '2026-03-02T16:50:00'},
                {'id':  96, 'batch_id': 8770, 'job_id': 1, 'job_name': 'test_batch_1',                          'state': 'Failed',  'exit_code': 1, 'pr_number': 15255, 'target_branch': 'main', 'source_branch': 'batch-fix',   'source_sha': 'aa0012', 'retried_by': 'ci', 'retried_at': '2026-03-01T20:15:00'},
                {'id':  95, 'batch_id': 8760, 'job_id': 7, 'job_name': 'test_hailtop_fs_1',                     'state': 'Failed',  'exit_code': 1, 'pr_number': 15252, 'target_branch': 'main', 'source_branch': 'fs-patch',    'source_sha': 'aa0013', 'retried_by': 'ci', 'retried_at': '2026-03-01T11:00:00'},
                {'id':  94, 'batch_id': 8750, 'job_id': 2, 'job_name': 'test_ci_1',                             'state': 'Failed',  'exit_code': 1, 'pr_number': 15249, 'target_branch': 'main', 'source_branch': 'ci-tweak',    'source_sha': 'aa0014', 'retried_by': 'ci', 'retried_at': '2026-02-28T17:40:00'},
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
