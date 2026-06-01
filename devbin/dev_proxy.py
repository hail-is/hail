import os

import aiohttp_jinja2
import jinja2
from hailtop import httpx
from hailtop.config import get_deploy_config
from hailtop.auth import hail_credentials
from hailtop.aiocloud.common import Session
from gear import new_csrf_token, SystemPermission

from web_common import base_context, setup_common_static_routes, web_security_headers
from web_common.web_common import WEB_COMMON_ROOT, TAILWIND_SERVICES

from aiohttp import web
import aiohttp_session
from aiohttp_session.cookie_storage import EncryptedCookieStorage

deploy_config = get_deploy_config()

# All services the devserver can proxy to
ALL_SERVICES: frozenset[str] = frozenset(['batch', 'batch-driver', 'ci', 'monitoring', 'auth'])

# Service name → Python module for template loading (only services that differ)
MODULES: dict[str, str] = {'batch': 'batch.front_end', 'batch-driver': 'batch.driver'}

# Per-service jinja2 app keys — needed to avoid template-name conflicts between services
_SERVICE_APP_KEYS: dict[str, web.AppKey] = {
    svc: web.AppKey(f'jinja_{svc}', jinja2.Environment)
    for svc in ALL_SERVICES
}

routes = web.RouteTableDef()
setup_common_static_routes(routes)

# With base_path = '/<service>', templates generate '/<service>/<svc-name>/static/...'
# so static routes must carry both the service prefix and the hardcoded service path.
_ALL_STATIC_DIRS: list[tuple[str, str]] = [
    ('/batch/batch/static/compiled-js', 'batch/batch/front_end/static/compiled-js'),
    ('/batch/batch/static/js', 'batch/batch/front_end/static/js'),
    ('/batch-driver/batch/static/compiled-js', 'batch/batch/driver/static/compiled-js'),
    ('/batch-driver/batch_driver/static/js', 'batch/batch/driver/static/js'),
    ('/ci/ci/static/compiled-js', 'ci/ci/static/compiled-js'),
    ('/monitoring/monitoring/static/compiled-js', 'monitoring/monitoring/static/compiled-js'),
    ('/auth/auth/static/compiled-js', 'auth/auth/static/compiled-js'),
]
for _path, _directory in _ALL_STATIC_DIRS:
    routes.static(_path, _directory)

# common_static must also be service-prefixed
_WEB_COMMON_STATIC = f'{WEB_COMMON_ROOT}/static'
for _svc in ALL_SERVICES:
    routes.static(f'/{_svc}/common_static', _WEB_COMMON_STATIC)

_IS_DEVELOPER_ENV = os.getenv('IS_DEVELOPER')
IS_DEVELOPER: bool | None = None if _IS_DEVELOPER_ENV is None else _IS_DEVELOPER_ENV.lower() not in ('0', 'false', 'no', '')

_FAKE_DEV_USERDATA = {'username': 'dev', 'system_permissions': {p.value: True for p in SystemPermission}}

BC = web.AppKey('backend_client', Session)


def _service_from_path(path: str) -> str | None:
    first_segment = path.lstrip('/').split('/')[0]
    return first_segment if first_segment in ALL_SERVICES else None


def _backend_url(service: str, raw_path: str) -> str:
    """Strip the /<service> prefix from the devserver path before forwarding."""
    prefix = f'/{service}'
    stripped = raw_path[len(prefix):] if raw_path.startswith(prefix) else raw_path
    if stripped and not stripped.startswith('/'):
        stripped = '/' + stripped
    return deploy_config.external_url(service, stripped or '/')


# Pages served from local templates (React shell with client-side data fetching).
# Paths include the service prefix that matches the new URL model.
_LOCAL_REACT_ROUTES: list[tuple[str, str, str, str]] = [
    ('monitoring',   'GET', '/monitoring/helloreact', 'hello_react.html'),
    ('auth',         'GET', '/auth/helloreact', 'hello_react.html'),
    ('batch-driver', 'GET', '/batch-driver/helloreact', 'hello_react.html'),
    ('ci',           'GET', '/ci/flaky_tests', 'flaky_tests.html'),
]

for _service, _verb, _path, _template in _LOCAL_REACT_ROUTES:
    async def _local_handler(
        request: web.Request,
        _s: str = _service,
        _t: str = _template,
    ) -> web.Response:
        return await _render_html(request, _s, _FAKE_DEV_USERDATA, _t, {'use_tailwind': True})
    routes.route(_verb, _path)(web_security_headers(_local_handler))


@routes.view('/{service:[^/]+}/api/{route:.*}')
@web_security_headers
async def default_proxied_api_route(request: web.Request) -> web.Response:
    service = request.match_info['service']
    if service not in ALL_SERVICES:
        raise web.HTTPNotFound()
    backend_client = request.app[BC]
    backend_route = _backend_url(service, request.raw_path)
    try:
        async with await backend_client.request(request.method, backend_route) as resp:
            body = await resp.read()
            content_type = resp.content_type
    except httpx.ClientResponseError as e:
        if e.status == 404:
            raise web.HTTPNotFound()
        raise
    return web.Response(body=body, content_type=content_type)


@routes.view('/{route:.*}')
@web_security_headers
async def default_proxied_web_route(request: web.Request) -> web.Response:
    service = _service_from_path(request.path)
    if service is None:
        raise web.HTTPNotFound()
    return await _render_html(request, service, **await _proxy(request, service))


async def _proxy(request: web.Request, service: str) -> dict:
    backend_client = request.app[BC]
    backend_route = _backend_url(service, request.raw_path)
    headers = {'x-hail-return-jinja-context': '1'}
    try:
        async with await backend_client.request(request.method, backend_route, headers=headers) as resp:
            return await resp.json()
    except httpx.ClientResponseError as e:
        if e.status == 404:
            raise web.HTTPNotFound()
        raise


async def _render_html(
    request: web.Request,
    service: str,
    userdata,
    file: str,
    page_context: dict,
    status_code: int = 200,
) -> web.Response:
    page_context['base_path'] = f'/{service}'
    if IS_DEVELOPER is not None:
        all_permissions = {p.value: IS_DEVELOPER for p in SystemPermission}
        page_context['system_permissions'] = all_permissions
        if userdata:
            userdata = dict(userdata)
            userdata['system_permissions'] = all_permissions

    if '_csrf' in request.cookies:
        csrf_token = request.cookies['_csrf']
    else:
        csrf_token = new_csrf_token()

    session = await aiohttp_session.get_session(request)
    context = base_context(session, userdata, service)
    context.update({
        'base_url': f'/{service}',
        'auth_base_url': '/auth',
        'batch_base_url': '/batch',
        'batch_driver_base_url': '/batch-driver',
        'ci_base_url': '/ci',
        'monitoring_base_url': '/monitoring',
    })
    context.update(page_context)
    context['use_tailwind'] = page_context.get('use_tailwind', service in TAILWIND_SERVICES)
    context['csrf_token'] = csrf_token

    response = aiohttp_jinja2.render_template(
        file, request, context, app_key=_SERVICE_APP_KEYS[service], status=status_code
    )
    response.set_cookie('_csrf', csrf_token, secure=True, httponly=True, samesite='strict')
    return response


async def on_startup(app: web.Application) -> None:
    app[BC] = Session(credentials=hail_credentials())


async def on_cleanup(app: web.Application) -> None:
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

# Set up a separate jinja2 env per service to avoid template-name conflicts
for _svc in ALL_SERVICES:
    _module = MODULES.get(_svc, _svc)
    _env = aiohttp_jinja2.setup(
        app,
        loader=jinja2.ChoiceLoader([jinja2.PackageLoader('web_common'), jinja2.PackageLoader(_module)]),
        app_key=_SERVICE_APP_KEYS[_svc],
    )
    _env.add_extension('jinja2.ext.do')

app.add_routes(routes)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)
aiohttp_session.setup(app, EncryptedCookieStorage(b'Thirty  two  length  bytes  key.'))

if __name__ == '__main__':
    web.run_app(app)
