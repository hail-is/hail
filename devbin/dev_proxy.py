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
    'batch': [('/batch/static/js', 'batch/batch/front_end/static/js')],
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
        return await render_template('ci', request, fake_userdata, 'flaky_tests.html', {'use_tailwind': True, 'base_path': ''})

    if os.getenv('MOCK_API_DATA'):

        @routes.get('/api/v1alpha/retried_tests')
        async def retried_tests_mock(request: web.Request):
            import random
            from datetime import datetime, timedelta, timezone
            from dateutil.parser import isoparse
            now = datetime.now(timezone.utc)
            rng = random.Random(42)

            job_names = [
                'test_hail_python', 'test_hail_python_unchecked_allocator',
                'test_hail_java', 'test_batch', 'test_hailtop_python_fs', 'test_ci',
            ]
            branches = [
                'big-refactor', 'perf-improvements', 'fix-aggregations', 'new-backend',
                'query-optimiser', 'type-system', 'batch-scaling', 'fs-cleanup',
                'ci-overhaul', 'docs-update', 'spark-compat', 'memory-fixes',
                'interval-tree', 'mt-operations', 'vcf-export', 'wgs-pipeline',
                'locus-expr', 'shuffle-fix', 'jvm-warmup', 'codec-v2',
                'auth-refresh', 'billing-v2', 'driver-oom', 'worker-pool',
                'hailtop-async',
            ]
            # one heavy retrier, a few occasional ones
            retried_by_pool = ['ci'] * 14 + ['turbo_tester'] * 3 + ['retry_queen'] * 2 + ['flaky_mcflakeface']

            # assign each branch a window within the last 90 days
            pr_base = 15200
            prs = []
            for i, branch in enumerate(branches):
                start_days_ago = rng.uniform(2, 90)
                lifespan = rng.uniform(1, 6)
                n_retries = rng.choices(
                    [0, 1, 2, 3, 5, 8, 12, 20, 35, 50],
                    weights=[5, 10, 15, 15, 12, 10, 8, 5, 3, 2],
                )[0]
                prs.append((pr_base + i, branch, start_days_ago, lifespan, n_retries))

            # generate retry events; group nearby retries on the same PR into a batch
            retries = []
            batch_counter = 8000
            for pr_number, branch, start_days_ago, lifespan, n_retries in prs:
                source_sha = rng.randbytes(3).hex()
                pr_retries = []
                for _ in range(n_retries):
                    days_ago = rng.uniform(max(0, start_days_ago - lifespan), start_days_ago)
                    pr_retries.append(now - timedelta(days=days_ago, seconds=rng.randint(0, 3600)))
                pr_retries.sort()

                # group into batches: new batch if gap > 2 hours
                batch_id = None
                batch_ts = None
                job_counter: dict = {}
                for ts in pr_retries:
                    if batch_ts is None or (ts - batch_ts).total_seconds() > 7200:
                        batch_counter += 1
                        batch_id = batch_counter
                        batch_ts = ts
                        job_counter = {}
                    job_name = f'{rng.choice(job_names)}_{rng.randint(1, 6)}'
                    job_counter[job_name] = job_counter.get(job_name, 0) + 1
                    state = rng.choices(['Failed', 'Error'], weights=[3, 1])[0]
                    retries.append({
                        'batch_id': batch_id,
                        'job_id': job_counter[job_name],
                        'job_name': job_name,
                        'state': state,
                        'exit_code': 1 if state == 'Failed' else None,
                        'pr_number': pr_number,
                        'target_branch': 'main',
                        'source_branch': branch,
                        'source_sha': source_sha,
                        'retried_by': rng.choice(retried_by_pool),
                        'retried_at': ts.isoformat(),
                    })

            retries.sort(key=lambda r: r['retried_at'], reverse=True)
            rows = [{'id': i + 1, **r} for i, r in enumerate(retries)]
            after_str = request.rel_url.query.get('after')
            if after_str is not None:
                after_dt = isoparse(after_str)
            else:
                after_dt = datetime.now(timezone.utc) - timedelta(days=14)
            rows = [r for r in rows if isoparse(r['retried_at']) >= after_dt]
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


@web.middleware
async def dev_csp_middleware(request: web.Request, handler):
    response = await handler(request)
    csp = response.headers.get('Content-Security-Policy', '')
    if csp:
        csp = csp.replace('script-src ', 'script-src http://localhost:8001 ')
        csp += ' connect-src \'self\' ws://localhost:8001;'
        response.headers['Content-Security-Policy'] = csp
    return response


app = web.Application(middlewares=[dev_csp_middleware])
setup_aiohttp_jinja2(app, MODULES.get(SERVICE, SERVICE))
app.add_routes(routes)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)
aiohttp_session.setup(app, EncryptedCookieStorage(b'Thirty  two  length  bytes  key.'))

if __name__ == '__main__':
    web.run_app(app)
