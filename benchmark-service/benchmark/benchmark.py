from typing import Any, Dict
import aiohttp_jinja2
from aiohttp import web
import logging
from gear import setup_aiohttp_session, web_authenticated_developers_only
from hailtop.config import get_deploy_config
from hailtop.tls import get_in_cluster_server_ssl_context
from hailtop.hail_logging import AccessLogger, configure_logging
from web_common import setup_aiohttp_jinja2, setup_common_static_routes
import json
import re
# from google.cloud import storage
import plotly
import plotly.express as px

configure_logging()
router = web.RouteTableDef()
logging.basicConfig(level=logging.DEBUG)
deploy_config = get_deploy_config()
log = logging.getLogger('benchmark')


# storage_client = storage.Client()
# bucket = storage_client.get_bucket('hail-benchmarks')
# blob = bucket.get_blob('0.2.20-3b2b439cabf9.json')
#
# blob_str = blob.download_as_string(client=None)
# pre_data = json.loads(blob_str)

file_path = '/0.2.45-ac6815ee857c-master.json'
with open(file_path) as f:
    pre_data = json.load(f)

x = re.findall('.*/+(.*)-(.*)-(.*)?\.json', file_path)  #was file_path
sha = x[0][1]

data = list()
prod_of_means = 1
for d in pre_data['benchmarks']:
    stats = dict()
    stats['name'] = d['name']
    stats['failed'] = d['failed']
    if not (d['failed']):
        prod_of_means *= d['mean']
        stats['f-stat'] = round(d['f-stat'], 6)
        stats['mean'] = round(d['mean'], 6)
        stats['median'] = round(d['median'], 6)
        stats['p-value'] = round(d['p-value'], 6)
        stats['stdev'] = round(d['stdev'], 6)
        stats['times'] = d['times']
    data.append(stats)
geometric_mean = prod_of_means ** (1.0 / len(pre_data['benchmarks']))

benchmarks = dict()
benchmarks['sha'] = sha
benchmarks['geometric_mean'] = geometric_mean
benchmarks['data'] = sorted(data, key=lambda i: i['name'])


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
async def show_name(request: web.Request) -> web.Response:

    name_data = next((item for item in data if item['name'] == str(request.match_info['name'])), None)
    fig = px.scatter(x=[item for item in range(0, len(name_data['times']))], y=name_data['times'])

    # context = {
    #     'name': request.match_info.get('name', ''),
    #     'name_data': name_data,
    #     'fig': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # }
    plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    context = {
        'name': request.match_info.get('name', ''),
        'plot': plot
    }

    response = aiohttp_jinja2.render_template('user.html', request,
                                              context=context)
    return response


@router.get('/')
@router.get('')
@web_authenticated_developers_only(redirect=False)
async def index(request: web.Request, userdata) -> Dict[str, Any]:  # pylint: disable=unused-argument
    context = {
        'current_date': 'July 10, 2020'
    }
    response = aiohttp_jinja2.render_template('index.html', request,
                                              context=benchmarks)
    return response

@router.post('/lookup')
async def lookup(request, userdata):  # pylint: disable=unused-argument
    data = await request.post()
    file = data['file']
   


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
