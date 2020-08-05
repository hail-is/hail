from typing import Any, Dict
import aiohttp_jinja2
from aiohttp import web
import logging
from gear import setup_aiohttp_session, web_authenticated_developers_only, check_csrf_token
from hailtop.config import get_deploy_config
from hailtop.tls import get_in_cluster_server_ssl_context
from hailtop.hail_logging import AccessLogger, configure_logging
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, render_template
from benchmark.utils import ReadGoogleStorage, get_geometric_mean
import json
import re
import plotly
import plotly.express as px

configure_logging()
router = web.RouteTableDef()
logging.basicConfig(level=logging.DEBUG)
deploy_config = get_deploy_config()
log = logging.getLogger('benchmark')

FILE_PATH_REGEX = re.compile(r'gs://hail-benchmarks/((?P<user>[^/]+)/)((?P<version>[^-]+)-)((?P<sha>[^-]+))(-(?P<tag>[^\.]+))?\.json')
default_filepath = 'gs://hail-benchmarks/tpoterba/0.2.45-ac6815ee857c-master.json'


def get_benchmarks(file_path):

    read_gs = ReadGoogleStorage()
    file_info = read_gs.parse_file_path(file_path)
    sha = file_info['sha']
    try:
        json_data = read_gs.get_data_as_string(file_path)
        pre_data = json.loads(json_data)
    except Exception:
        message = 'could not find file'
        log.info('could not get blob: ' + message, exc_info=True)
        raise web.HTTPBadRequest(text=message)

    data = []
    prod_of_means = 1
    for d in pre_data['benchmarks']:
        stats = dict()
        stats['name'] = d['name']
        stats['failed'] = d['failed']
        if not d['failed']:
            prod_of_means *= d['mean']
            stats['f-stat'] = round(d['f-stat'], 6)
            stats['mean'] = round(d['mean'], 6)
            stats['median'] = round(d['median'], 6)
            stats['p-value'] = round(d['p-value'], 6)
            stats['stdev'] = round(d['stdev'], 6)
            stats['times'] = d['times']
        data.append(stats)
    geometric_mean = get_geometric_mean(prod_of_means, len(pre_data['benchmarks']))
        # prod_of_means ** (1.0 / len(pre_data['benchmarks']))

    benchmarks = dict()
    benchmarks['sha'] = sha
    benchmarks['geometric_mean'] = geometric_mean
    benchmarks['data'] = sorted(data, key=lambda i: i['name'])
    return benchmarks


@router.get('/healthcheck')
async def healthcheck(request: web.Request) -> web.Response:  # pylint: disable=unused-argument
    return web.Response()


@router.get('/{username}')
async def greet_user(request: web.Request) -> web.Response:

    context = {
        'username': request.match_info.get('username', ''),
        'current_date': 'July 10, 2020'
    }
    response = aiohttp_jinja2.render_template('user.html', request,
                                              context=context)
    return response


@router.get('/name/{name}')
@web_authenticated_developers_only(redirect=False)
async def show_name(request: web.Request, userdata) -> web.Response:  # pylint: disable=unused-argument
    benchmarks = get_benchmarks(default_filepath)
    name_data = next((item if item['name'] == str(request.match_info['name']) else NameError("name could not be found")
                      for item in benchmarks['data']), None)
    # next((item for item in benchmarks['data'] if item['name'] == str(request.match_info['name'])), None)
    fig = px.scatter(x=list(range(0, len(name_data['times']))), y=name_data['times'])

    plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    context = {
        'name': request.match_info.get('name', ''),
        'plot': plot
    }

    return await render_template('benchmark', request, userdata, 'user.html', context)


@router.get('/')
@router.get('')
@web_authenticated_developers_only(redirect=False)
async def index(request: web.Request, userdata) -> Dict[str, Any]:  # pylint: disable=unused-argument
    benchmarks_context = get_benchmarks(default_filepath)
    return await render_template('benchmark', request, userdata, 'index.html', benchmarks_context)


@router.post('/lookup')
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
async def lookup(request, userdata):  # pylint: disable=unused-argument
    data = await request.post()
    file = data['file']
    global default_filepath
    default_filepath = file
    benchmarks_context = get_benchmarks(file)
    return await render_template('benchmark', request, userdata, 'index.html', benchmarks_context)


def init_app() -> web.Application:
    app = web.Application()
    setup_aiohttp_jinja2(app, 'benchmark')
    setup_aiohttp_session(app)

    setup_common_static_routes(router)
    app.add_routes(router)
    return app


web.run_app(deploy_config.prefix_application(init_app(), 'benchmark'),
            host='0.0.0.0',
            port=5000,
            access_log_class=AccessLogger,
            ssl_context=get_in_cluster_server_ssl_context())
