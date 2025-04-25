import importlib
import os
from functools import wraps
from typing import Any, Dict, Optional

import aiohttp_jinja2
import aiohttp_session
import jinja2
import sass
from aiohttp import web

from gear import UserData, new_csrf_token
from hailtop.config import get_deploy_config

deploy_config = get_deploy_config()

WEB_COMMON_ROOT = os.path.dirname(os.path.abspath(__file__))


TAILWIND_SERVICES = {'auth', 'batch'}


def sass_compile(module_name):
    module = importlib.import_module(module_name)
    module_filename = module.__file__
    assert module_filename
    module_root = os.path.dirname(os.path.abspath(module_filename))

    scss_path = f'{module_root}/styles'
    css_path = f'{module_root}/static/css'
    os.makedirs(scss_path, exist_ok=True)
    os.makedirs(css_path, exist_ok=True)

    sass.compile(dirname=(scss_path, css_path), output_style='compressed', include_paths=[f'{WEB_COMMON_ROOT}/styles'])


def setup_aiohttp_jinja2(app: web.Application, module: str, *extra_loaders: jinja2.BaseLoader):
    jinja_env = aiohttp_jinja2.setup(
        app,
        loader=jinja2.ChoiceLoader([jinja2.PackageLoader('web_common'), jinja2.PackageLoader(module), *extra_loaders]),
    )
    jinja_env.add_extension('jinja2.ext.do')


_compiled = False


def setup_common_static_routes(routes):
    global _compiled

    if not _compiled:
        sass_compile('web_common')
        _compiled = True
    routes.static('/common_static', f'{WEB_COMMON_ROOT}/static')


def set_message(session, text, type):
    assert type in ('info', 'error')
    session['message'] = {'text': text, 'type': type}


def base_context(session, userdata, service):
    context = {
        'base_path': deploy_config.base_path(service),
        'base_url': deploy_config.external_url(service, ''),
        'www_base_url': deploy_config.external_url('www', ''),
        'auth_base_url': deploy_config.external_url('auth', ''),
        'batch_base_url': deploy_config.external_url('batch', ''),
        'batch_driver_base_url': deploy_config.external_url('batch-driver', ''),
        'ci_base_url': deploy_config.external_url('ci', ''),
        'grafana_base_url': deploy_config.external_url('grafana', ''),
        'monitoring_base_url': deploy_config.external_url('monitoring', ''),
        'userdata': userdata,
    }
    if 'message' in session:
        context['message'] = session.pop('message')
    return context


async def render_template(
    service: str,
    request: web.Request,
    userdata: Optional[UserData],
    file: str,
    page_context: Dict[str, Any],
) -> web.Response:
    if request.headers.get('x-hail-return-jinja-context'):
        if userdata and userdata['is_developer']:
            return web.json_response({'file': file, 'page_context': page_context, 'userdata': userdata})
        raise ValueError('Only developers can request the jinja context')

    if '_csrf' in request.cookies:
        csrf_token = request.cookies['_csrf']
    else:
        csrf_token = new_csrf_token()

    session = await aiohttp_session.get_session(request)
    context = base_context(session, userdata, service)
    context.update(page_context)
    context['use_tailwind'] = service in TAILWIND_SERVICES
    context['csrf_token'] = csrf_token

    response = aiohttp_jinja2.render_template(file, request, context)
    response.set_cookie('_csrf', csrf_token, secure=True, httponly=True, samesite='strict')
    return response


def api_security_headers(fun):
    @wraps(fun)
    async def wrapped(request, *args, **kwargs):
        response = await fun(request, *args, **kwargs)
        response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains;'
        return response

    return wrapped


def web_security_headers(fun):
    # Although this looks like a boring passthrough, we're explicitly not changing the optional parameters that
    # would otherwise make the fun-wrapping via annotations behave funky.
    return web_security_header_generator(fun)


def web_security_headers_swagger(fun):
    return web_security_header_generator(
        fun, extra_script='unpkg.com', extra_style='unpkg.com', extra_img='validator.swagger.io'
    )


def web_security_headers_unsafe_eval(fun):
    return web_security_header_generator(fun, extra_script='\'unsafe-eval\'')


def web_security_header_generator(fun, extra_script: str = '', extra_style: str = '', extra_img: str = ''):
    @wraps(fun)
    async def wrapped(request, *args, **kwargs):
        response = await fun(request, *args, **kwargs)
        response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains;'

        default_src = 'default-src \'self\';'
        style_src = f'style-src \'self\' \'unsafe-inline\' {extra_style} fonts.googleapis.com fonts.gstatic.com;'
        font_src = 'font-src \'self\' fonts.gstatic.com;'
        script_src = f'script-src \'self\' \'unsafe-inline\' {extra_script} cdn.jsdelivr.net cdn.plot.ly;'
        img_src = f'img-src \'self\' {extra_img};'
        frame_ancestors = 'frame-ancestors \'self\';'

        response.headers['Content-Security-Policy'] = (
            f'{default_src} {font_src} {style_src} {script_src} {img_src} {frame_ancestors}'
        )
        return response

    return wrapped
