from typing import Set
import os
from aiohttp import web
import jinja2
import logging
import aiohttp_session
from prometheus_async.aio.web import server_stats  # type: ignore

from hailtop.config import get_deploy_config
from hailtop.tls import internal_server_ssl_context
from hailtop.hail_logging import AccessLogger
from gear import setup_aiohttp_session, web_maybe_authenticated_user, monitor_endpoint
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, render_template, sass_compile


MODULE_PATH = os.path.dirname(__file__)
log = logging.getLogger('website')
deploy_config = get_deploy_config()
routes = web.RouteTableDef()


def redirect(from_url, to_url):
    async def serve(request):  # pylint: disable=unused-argument
        raise web.HTTPFound(to_url)

    routes.get(from_url)(serve)


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=unused-argument
    return web.Response()


@routes.get('/robots.txt')
async def get_robots(request):  # pylint: disable=unused-argument
    return web.Response(text='user-agent: *\nAllow: /')


routes.static('/static', f'{MODULE_PATH}/static')
redirect('/favicon.ico', 'common_static/hail_logo_sq.ico')
redirect('/docs/batch', 'batch/index.html')
redirect('/docs/batch/', 'index.html')
redirect('/docs/0.2', '0.2/index.html')
redirect('/docs/0.2/', 'index.html')


DOCS_PATH = f'{MODULE_PATH}/docs/'
STATIC_DOCS_PATHS = ['0.2/_static', '0.2/_sources', 'batch']
FQ_STATIC_DOCS_PATHS: Set[str] = set()


for path in STATIC_DOCS_PATHS:
    routes.static('/docs/' + path, DOCS_PATH + path)
    FQ_STATIC_DOCS_PATHS.add(DOCS_PATH + path)


docs_pages = set(
    dirname[len(DOCS_PATH) :] + '/' + file
    for dirname, _, filenames in os.walk(DOCS_PATH)
    if dirname not in FQ_STATIC_DOCS_PATHS
    for file in filenames
)


@routes.get('/docs/{tail:.*}')
@monitor_endpoint
@web_maybe_authenticated_user
async def serve_docs(request, userdata):
    tail = request.match_info['tail']
    if tail in docs_pages:
        if tail.endswith('.html'):
            return await render_template('www', request, userdata, tail, dict())
        # Chrome fails to download the tutorials.tar.gz file without the Content-Type header.
        return web.FileResponse(f'{DOCS_PATH}/{tail}', headers={'Content-Type': 'application/octet-stream'})
    raise web.HTTPNotFound()


def make_template_handler(template_fname):
    @web_maybe_authenticated_user
    async def serve(request, userdata):
        return await render_template('www', request, userdata, template_fname, dict())

    return serve


for fname in os.listdir(f'{MODULE_PATH}/pages'):
    routes.get(f'/{fname}')(make_template_handler(fname))
    if fname == 'index.html':
        routes.get('')(make_template_handler(fname))  # so internal.hail.is/<namespace>/www can work without a /
        routes.get('/')(make_template_handler(fname))


def run(local_mode):
    app = web.Application()

    if local_mode:
        log.error('running in local mode with bogus cookie storage key')
        aiohttp_session.setup(
            app,
            aiohttp_session.cookie_storage.EncryptedCookieStorage(
                b'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                cookie_name=deploy_config.auth_session_cookie_name(),
                secure=True,
                httponly=True,
                domain=os.environ['HAIL_DOMAIN'],
                # 2592000s = 30d
                max_age=2592000,
            ),
        )
    else:
        setup_aiohttp_session(app)

    setup_aiohttp_jinja2(
        app, 'website', jinja2.PackageLoader('website', 'pages'), jinja2.PackageLoader('website', 'docs')
    )
    setup_common_static_routes(routes)
    app.add_routes(routes)
    app.router.add_get("/metrics", server_stats)
    sass_compile('website')
    web.run_app(
        deploy_config.prefix_application(app, 'www'),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=None if local_mode else internal_server_ssl_context(),
    )
