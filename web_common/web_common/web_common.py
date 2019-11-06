import os
import importlib
import sass
import jinja2
import aiohttp_jinja2
import aiohttp_session
from hailtop.config import get_deploy_config
from gear import new_csrf_token

deploy_config = get_deploy_config()

WEB_COMMON_ROOT = os.path.dirname(os.path.abspath(__file__))


def sass_compile(module_name):
    module = importlib.import_module(module_name)
    module_root = os.path.dirname(os.path.abspath(module.__file__))

    scss_path = f'{module_root}/styles'
    css_path = f'{module_root}/static/css'
    os.makedirs(css_path, exist_ok=True)

    sass.compile(
        dirname=(scss_path, css_path), output_style='compressed',
        include_paths=[f'{WEB_COMMON_ROOT}/styles'])


def setup_aiohttp_jinja2(app, module):
    aiohttp_jinja2.setup(
        app, loader=jinja2.ChoiceLoader([
            jinja2.PackageLoader('web_common'),
            jinja2.PackageLoader(module)
        ]))


_compiled = False


def setup_common_static_routes(routes):
    global _compiled

    if not _compiled:
        sass_compile('web_common')
        _compiled = True
    routes.static('/common_static', f'{WEB_COMMON_ROOT}/static')


def set_message(session, text, type):
    assert type in ('info', 'error')
    session['message'] = {
        'text': text,
        'type': type
    }


def base_context(session, userdata, service):
    context = {
        'base_path': deploy_config.base_path(service),
        'base_url': deploy_config.external_url(service, ''),
        'notebook_base_url': deploy_config.external_url('notebook', ''),
        'workshop_base_url': deploy_config.external_url('workshop', ''),
        'auth_base_url': deploy_config.external_url('auth', ''),
        'batch_base_url': deploy_config.external_url('batch', ''),
        'batch2_base_url': deploy_config.external_url('batch2', ''),
        'batch2_driver_base_url': deploy_config.external_url('batch2-driver', ''),
        'ci_base_url': deploy_config.external_url('ci', ''),
        'scorecard_base_url': deploy_config.external_url('scorecard', ''),
        'userdata': userdata
    }
    if 'message' in session:
        context['message'] = session.pop('message')
    return context


async def render_template(service, request, userdata, file, page_context):
    if '_csrf' in request.cookies:
        csrf_token = request.cookies['_csrf']
    else:
        csrf_token = new_csrf_token()

    session = await aiohttp_session.get_session(request)
    context = base_context(session, userdata, service)
    context.update(page_context)
    context['csrf_token'] = csrf_token

    response = aiohttp_jinja2.render_template(file, request, context)
    response.set_cookie('_csrf', csrf_token, domain=os.environ['HAIL_DOMAIN'], secure=True, httponly=True)
    return response
