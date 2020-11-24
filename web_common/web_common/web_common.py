from typing import Dict, Any
import os
from typing_extensions import Literal
import importlib
import sass
import jinja2
import aiohttp
import aiohttp_jinja2
import aiohttp_session
from hailtop.config import get_deploy_config
from gear import new_csrf_token

# https://github.com/aio-libs/aiohttp/issues/3714#issuecomment-486973504
aiohttp.web_request.Request.__module__ = 'aiohttp.web'
aiohttp.web.Application.__module__ = 'aiohttp.web'

deploy_config = get_deploy_config()

WEB_COMMON_ROOT = os.path.dirname(os.path.abspath(__file__))


def sass_compile(module_name: str):
    """Compile a module's Sass files.

    Parameters
    ----------
    module_name:
        The module in which to find sass files to compile.
    """
    module = importlib.import_module(module_name)
    module_root = os.path.dirname(os.path.abspath(module.__file__))

    scss_path = f'{module_root}/styles'
    css_path = f'{module_root}/static/css'
    os.makedirs(css_path, exist_ok=True)

    sass.compile(
        dirname=(scss_path, css_path), output_style='compressed',
        include_paths=[f'{WEB_COMMON_ROOT}/styles'])


def setup_aiohttp_jinja2(app: aiohttp.web.Application, module: str):
    """Enable aiohttp applications to use Jinja2 templates.

    Parameters
    ----------
    app:
        The application in which to enable jinja2 rendering.
    module:
        The name of the module in which the application is defined.
    """
    aiohttp_jinja2.setup(
        app, loader=jinja2.ChoiceLoader([
            jinja2.PackageLoader('web_common'),
            jinja2.PackageLoader(module)
        ]))


_compiled = False


def setup_common_static_routes(routes: aiohttp.web.RouteTableDef):
    """Serve web_common's static (e.g. CSS) files at /common_static.

    Parameters
    ----------
    route:
        The route table to which to add the ``/common_static`` route.
    """
    global _compiled

    if not _compiled:
        sass_compile('web_common')
        _compiled = True
    routes.static('/common_static', f'{WEB_COMMON_ROOT}/static')


WebMessageType = Literal['info', 'error']


def set_message(session: Dict, text: str, type: WebMessageType):
    """Set a message for display in the response to the web browser.

    Parameters
    ----------
    session:
        The aiohttp session for the current request.
    text:
        A short message to display to the user.
    type:
        Info is displayed on a green background whereas error is displayed on a
        red background.
    """
    assert type in ('info', 'error')
    session['message'] = {
        'text': text,
        'type': type
    }


def base_context(session: Dict, userdata: Dict, service: str):
    """The base page context for Hail Jinja2 templates.

    Parameters
    ----------
    session:
        The aiohttp session for the current request.
    userdata:
        The userdata FIXME: link to gear.
    service:
        The current service.
    """
    context = {
        'base_path': deploy_config.base_path(service),
        'base_url': deploy_config.external_url(service, ''),
        'www_base_url': deploy_config.external_url('www', ''),
        'notebook_base_url': deploy_config.external_url('notebook', ''),
        'workshop_base_url': deploy_config.external_url('workshop', ''),
        'auth_base_url': deploy_config.external_url('auth', ''),
        'batch_base_url': deploy_config.external_url('batch', ''),
        'batch_driver_base_url': deploy_config.external_url('batch-driver', ''),
        'ci_base_url': deploy_config.external_url('ci', ''),
        'scorecard_base_url': deploy_config.external_url('scorecard', ''),
        'monitoring_base_url': deploy_config.external_url('monitoring', ''),
        'benchmark_base_url': deploy_config.external_url('benchmark', ''),
        'userdata': userdata
    }
    if 'message' in session:
        context['message'] = session.pop('message')
    return context


async def render_template(service: str,
                          request: aiohttp.web.Request,
                          userdata: Dict,
                          file: str,
                          page_context: Dict[str, Any]):
    """Render a Jinja2 template.

    Parameters
    ----------
    service:
        The current service.
    request:
        The current request.
    userdata:
        The userdata.
    file:
        The template to render.
    page_context:
        A mapping from variables used in the template to values.
    """
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
