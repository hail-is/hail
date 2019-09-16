import os
import importlib
import sass
import jinja2
import aiohttp_jinja2

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


def setup_common_static_routes(routes):
    sass_compile('web_common')
    routes.static('/common_static', f'{WEB_COMMON_ROOT}/static')


def base_context(deploy_config, userdata, service):
    return {
        'base_path': deploy_config.base_path(service),
        'base_url': deploy_config.external_url(service, ''),
        'notebook_base_url': deploy_config.external_url('notebook2', ''),
        'auth_base_url': deploy_config.external_url('auth', ''),
        'batch_base_url': deploy_config.external_url('batch', ''),
        'ci_base_url': deploy_config.external_url('ci', ''),
        'scorecard_base_url': deploy_config.external_url('scorecard', ''),
        'userdata': userdata
    }
