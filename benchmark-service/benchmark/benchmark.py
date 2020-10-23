import asyncio
import os
import aiohttp
from aiohttp import web
import logging
from gear import setup_aiohttp_session, web_authenticated_developers_only
from hailtop.config import get_deploy_config
from hailtop.tls import get_in_cluster_server_ssl_context
from hailtop.hail_logging import AccessLogger, configure_logging
from hailtop.utils import retry_long_running
import hailtop.batch_client.aioclient as bc
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, render_template
from benchmark.utils import ReadGoogleStorage, get_geometric_mean, parse_file_path, enumerate_list_of_trials,\
    list_benchmark_files, round_if_defined, submit_batch
import json
import re
import plotly
import plotly.express as px
from scipy.stats.mstats import gmean, hmean
import numpy as np
import pandas as pd
import gidgethub
import gidgethub.aiohttp
from .config import BENCHMARK_TEST_BUCKET_NAME, START_POINT

configure_logging()
router = web.RouteTableDef()
logging.basicConfig(level=logging.DEBUG)
deploy_config = get_deploy_config()
log = logging.getLogger('benchmark')

BENCHMARK_FILE_REGEX = re.compile(r'gs://((?P<bucket>[^/]+)/)((?P<user>[^/]+)/)((?P<version>[^-]+)-)((?P<sha>[^-]+))(-(?P<tag>[^\.]+))?\.json')

BENCHMARK_ROOT = os.path.dirname(os.path.abspath(__file__))

benchmark_data = None


def get_benchmarks(app, file_path):
    gs_reader = app['gs_reader']
    try:
        json_data = gs_reader.get_data_as_string(file_path)
        pre_data = json.loads(json_data)
    except Exception as e:
        message = f'could not find file, {file_path}'
        log.info('could not get blob: ' + message, exc_info=True)
        raise web.HTTPBadRequest(text=message) from e

    data = {}
    prod_of_means = 1
    for d in pre_data['benchmarks']:
        stats = dict()
        stats['name'] = d.get('name')
        stats['failed'] = d.get('failed')
        if not d['failed']:
            prod_of_means *= d.get('mean', 1)
            stats['f-stat'] = round_if_defined(d.get('f-stat'))
            stats['mean'] = round_if_defined(d.get('mean'))
            stats['median'] = round_if_defined(d.get('median'))
            stats['p-value'] = round_if_defined(d.get('p-value'))
            stats['stdev'] = round_if_defined(d.get('stdev'))
            stats['times'] = d.get('times')
            stats['trials'] = d.get('trials')
        data[stats['name']] = stats
    geometric_mean = get_geometric_mean(prod_of_means, len(pre_data['benchmarks']))

    file_info = parse_file_path(BENCHMARK_FILE_REGEX, file_path)
    sha = file_info['sha']
    benchmarks = dict()
    benchmarks['sha'] = sha
    benchmarks['geometric_mean'] = geometric_mean
    benchmarks['data'] = data
    return benchmarks


def get_comparisons(benchmarks1, benchmarks2, metric):
    def get_metric(data):
        if metric == 'median':
            return data.get('median')
        assert metric == 'best'
        times = data.get('times')
        if times:
            return min(times)
        return None

    d1_keys = set(benchmarks1['data'].keys())
    d2_keys = set(benchmarks2['data'].keys())
    set_of_names = d1_keys.union(d2_keys)

    comparisons = []
    for name in set_of_names:
        data1 = benchmarks1['data'].get(name)
        data2 = benchmarks2['data'].get(name)
        if data2 is None:
            comparisons.append((name, get_metric(data1), None))
        elif data1 is None:
            comparisons.append((name, None, get_metric(data2)))
        else:
            comparisons.append((name, get_metric(data1), get_metric(data2)))

    return comparisons


def fmt_time(t):
    if t is not None:
        return round(t, 3)
    return None


def fmt_diff(ratio):
    if ratio is not None:
        return round(ratio * 100, 3)
    return None


def final_comparisons(comparisons):
    comps = []
    ratios = []
    final_comps = {}
    for name, r1, r2 in comparisons:
        if r1 is None:
            comps.append((name, None, None, fmt_time(r2)))
        elif r2 is None:
            comps.append((name, None, fmt_time(r1), None))
        else:
            r = r1 / r2
            ratios.append(r)
            comps.append((name, fmt_diff(r), fmt_time(r1), fmt_time(r2)))
    final_comps['comps'] = comps
    if len(ratios) == 0:
        final_comps['harmonic_mean'] = None
        final_comps['geometric_mean'] = None
        final_comps['arithmetic_mean'] = None
        final_comps['median'] = None
    else:
        final_comps['harmonic_mean'] = fmt_diff(hmean(ratios))
        final_comps['geometric_mean'] = fmt_diff(gmean(ratios))
        final_comps['arithmetic_mean'] = fmt_diff(np.mean(ratios))
        final_comps['median'] = fmt_diff(np.median(ratios))
    return final_comps


@router.get('/healthcheck')
async def healthcheck(request: web.Request) -> web.Response:  # pylint: disable=unused-argument
    return web.Response()


@router.get('/name/{name}')
@web_authenticated_developers_only(redirect=False)
async def show_name(request: web.Request, userdata) -> web.Response:  # pylint: disable=unused-argument
    file_path = request.query.get('file')
    benchmarks = get_benchmarks(request.app, file_path)
    name_data = benchmarks['data'][str(request.match_info['name'])]

    try:
        data = enumerate_list_of_trials(name_data['trials'])
        d = {
            'trial': data['trial_indices'],
            'wall_time': data['wall_times'],
            'index': data['within_group_index']
        }
        df = pd.DataFrame(d)
        fig = px.scatter(df, x=df.trial, y=df.wall_time, hover_data=['index'])
        plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        message = 'could not find name'
        log.info('name is of type NoneType: ' + message, exc_info=True)
        raise web.HTTPBadRequest(text=message) from e

    context = {
        'name': request.match_info.get('name', ''),
        'plot': plot
    }

    return await render_template('benchmark', request, userdata, 'name.html', context)


@router.get('/')
@router.get('')
@web_authenticated_developers_only(redirect=False)
async def index(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    file = request.query.get('file')
    if file is None:
        benchmarks_context = None
    else:
        benchmarks_context = get_benchmarks(request.app, file)
    context = {'file': file,
               'benchmarks': benchmarks_context,
               'benchmark_file_list': list_benchmark_files(app['gs_reader'])}
    return await render_template('benchmark', request, userdata, 'index.html', context)


@router.get('/compare')
@web_authenticated_developers_only(redirect=False)
async def compare(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    file1 = request.query.get('file1')
    file2 = request.query.get('file2')
    metric = request.query.get('metrics')
    if file1 is None or file2 is None:
        benchmarks_context1 = None
        benchmarks_context2 = None
        comparisons = None
    else:
        benchmarks_context1 = get_benchmarks(app, file1)
        benchmarks_context2 = get_benchmarks(app, file2)
        comparisons = final_comparisons(get_comparisons(benchmarks_context1, benchmarks_context2, metric))
    context = {'file1': file1,
               'file2': file2,
               'metric': metric,
               'benchmarks1': benchmarks_context1,
               'benchmarks2': benchmarks_context2,
               'comparisons': comparisons,
               'benchmark_file_list': list_benchmark_files(app['gs_reader'])}
    return await render_template('benchmark', request, userdata, 'compare.html', context)


async def update_commits(app):
    global benchmark_data
    github_client = app['github_client']
    batch_client = app['batch_client']
    gs_reader = app['gs_reader']

    request_string = f'/repos/hail-is/hail/commits?since={START_POINT}'
    gh_data = await github_client.getitem(request_string)
    new_commits = []
    formatted_new_commits = []
    for gh_commit in gh_data:

        sha = gh_commit.get('sha')

        batches = [b async for b in batch_client.list_batches(q=f'sha={sha} running')]
        try:
            batch = batches[-1]
            batch_status = await batch.status()
        except Exception:  # pylint: disable=broad-except
            batch_status = None

        file_path = f'gs://{BENCHMARK_TEST_BUCKET_NAME}/benchmark-test/{sha}'
        has_results_file = gs_reader.file_exists(file_path)

        if not batches and not has_results_file:
            new_commits.append(gh_commit)

    log.info('got new commits')
    for gh_commit in new_commits:
        batch_id = await submit_batch(gh_commit, batch_client)
        batch = batch_client.get_batch(batch_id)
        batch_status = await batch.last_known_status()
        sha = gh_commit.get('sha')
        log.info(f'submitted a batch {batch_id} for commit {sha}')
        commit = {
            'sha': sha,
            'title': gh_commit['commit']['message'],
            'author': gh_commit['commit']['author']['name'],
            'date': gh_commit['commit']['author']['date'],
            'status': batch_status
        }
        formatted_new_commits.append(commit)

    benchmark_data = {
        'commits': formatted_new_commits
    }


async def github_polling_loop(app):
    while True:
        await update_commits(app)
        log.info('successfully queried github')
        await asyncio.sleep(180)


async def on_startup(app):
    with open(os.environ.get('HAIL_CI_OAUTH_TOKEN', 'oauth-token/oauth-token'), 'r') as f:
        oauth_token = f.read().strip()
    app['gs_reader'] = ReadGoogleStorage(service_account_key_file='/benchmark-gsa-key/key.json')
    app['github_client'] = gidgethub.aiohttp.GitHubAPI(aiohttp.ClientSession(),
                                                       'hail-is/hail',
                                                       oauth_token=oauth_token)
    app['batch_client'] = bc.BatchClient(billing_project='test')
    asyncio.ensure_future(retry_long_running('github_polling_loop', github_polling_loop, app))


def run():
    app = web.Application()
    setup_aiohttp_jinja2(app, 'benchmark')
    setup_aiohttp_session(app)

    setup_common_static_routes(router)
    router.static('/static', f'{BENCHMARK_ROOT}/static')
    app.add_routes(router)
    app.on_startup.append(on_startup)
    web.run_app(deploy_config.prefix_application(app, 'benchmark'),
                host='0.0.0.0',
                port=5000,
                access_log_class=AccessLogger,
                ssl_context=get_in_cluster_server_ssl_context())
