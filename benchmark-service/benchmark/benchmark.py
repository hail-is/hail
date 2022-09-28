import asyncio
import json
import logging
import os
import re

import gidgethub
import gidgethub.aiohttp
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from aiohttp import web
from benchmark.utils import (
    enumerate_list_of_trials,
    get_geometric_mean,
    list_benchmark_files,
    parse_file_path,
    round_if_defined,
    submit_test_batch,
)
from scipy.stats.mstats import gmean, hmean

import hailtop.batch_client.aioclient as bc
from gear import AuthClient, setup_aiohttp_session
from hailtop import aiotools, httpx
from hailtop.aiocloud import aiogoogle
from hailtop.config import get_deploy_config
from hailtop.hail_logging import AccessLogger, configure_logging
from hailtop.tls import internal_server_ssl_context
from hailtop.utils import collect_agen, humanize_timedelta_msecs, retry_long_running
from web_common import render_template, setup_aiohttp_jinja2, setup_common_static_routes

from .config import BENCHMARK_RESULTS_PATH, START_POINT

configure_logging()
router = web.RouteTableDef()
logging.basicConfig(level=logging.DEBUG)
deploy_config = get_deploy_config()
log = logging.getLogger('benchmark')

auth = AuthClient()

BENCHMARK_FILE_REGEX = re.compile(
    r'gs://((?P<bucket>[^/]+)/)((?P<user>[^/]+)/)((?P<instanceId>[^/]*)/)((?P<version>[^-]+)-)((?P<sha>[^-]+))(-(?P<tag>[^\.]+))?\.json'
)

GH_COMMIT_MESSAGE_REGEX = re.compile(r'(?P<title>.*)\s\(#(?P<pr_id>\d+)\)(?P<rest>.*)')

BENCHMARK_ROOT = os.path.dirname(os.path.abspath(__file__))

benchmark_data = {'commits': {}, 'dates': [], 'geo_means': [], 'pr_ids': [], 'shas': []}


with open(os.environ.get('HAIL_CI_OAUTH_TOKEN', 'oauth-token/oauth-token'), 'r', encoding='utf-8') as f:
    oauth_token = f.read().strip()


async def get_benchmarks(app, file_path):
    log.info(f'get_benchmarks file_path={file_path}')
    fs: aiotools.AsyncFS = app['fs']
    try:
        json_data = (await fs.read(file_path)).decode('utf-8')
        pre_data = json.loads(json_data)
    except FileNotFoundError:
        message = f'could not find file, {file_path}'
        log.info('could not get blob: ' + message, exc_info=True)
        return None

    data = {}
    prod_of_means = 1
    for d in pre_data['benchmarks']:
        stats = {}
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
    benchmarks = {}
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
@auth.web_authenticated_developers_only(redirect=False)
async def show_name(request: web.Request, userdata) -> web.Response:  # pylint: disable=unused-argument
    file_path = request.query.get('file')
    benchmarks = await get_benchmarks(request.app, file_path)
    name_data = benchmarks['data'][str(request.match_info['name'])]

    try:
        data = enumerate_list_of_trials(name_data['trials'])
        d = {'trial': data['trial_indices'], 'wall_time': data['wall_times'], 'index': data['within_group_index']}
        df = pd.DataFrame(d)
        fig = px.scatter(df, x=df.trial, y=df.wall_time, hover_data=['index'])
        plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        message = 'could not find name'
        log.info('name is of type NoneType: ' + message, exc_info=True)
        raise web.HTTPBadRequest(text=message) from e

    context = {'name': request.match_info.get('name', ''), 'plot': plot}

    return await render_template('benchmark', request, userdata, 'name.html', context)


@router.get('/')
@router.get('')
async def index(request):
    userdata = {}
    d = {
        'dates': benchmark_data['dates'],
        'geo_means': benchmark_data['geo_means'],
        'pr_ids': benchmark_data['pr_ids'],
        'commits': benchmark_data['shas'],
    }
    assert len(d['dates']) == len(d['geo_means']), d
    df = pd.DataFrame(d)
    if not df.dates.empty:
        fig = px.line(df, x=df.dates, y=df.geo_means, hover_data=['pr_ids', 'commits'])
        fig.update_xaxes(rangeslider_visible=True)
        plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        plot = None
    context = {'commits': benchmark_data['commits'], 'plot': plot, 'benchmark_results_path': BENCHMARK_RESULTS_PATH}
    return await render_template('benchmark', request, userdata, 'index.html', context)


@router.get('/lookup')
@auth.web_authenticated_developers_only(redirect=False)
async def lookup(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    file = request.query.get('file')
    if file is None:
        benchmarks_context = None
    else:
        benchmarks_context = await get_benchmarks(request.app, file)
    context = {
        'file': file,
        'benchmarks': benchmarks_context,
        'benchmark_file_list': await list_benchmark_files(app['fs']),
    }
    return await render_template('benchmark', request, userdata, 'lookup.html', context)


@router.get('/compare')
@auth.web_authenticated_developers_only(redirect=False)
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
        benchmarks_context1 = await get_benchmarks(app, file1)
        benchmarks_context2 = await get_benchmarks(app, file2)
        comparisons = final_comparisons(get_comparisons(benchmarks_context1, benchmarks_context2, metric))
    context = {
        'file1': file1,
        'file2': file2,
        'metric': metric,
        'benchmarks1': benchmarks_context1,
        'benchmarks2': benchmarks_context2,
        'comparisons': comparisons,
        'benchmark_file_list': await list_benchmark_files(app['fs']),
    }
    return await render_template('benchmark', request, userdata, 'compare.html', context)


@router.get('/batches/{batch_id}')
@auth.web_authenticated_developers_only()
async def get_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    batch_client = request.app['batch_client']
    b = await batch_client.get_batch(batch_id)
    status = await b.last_known_status()
    jobs = await collect_agen(b.jobs())
    for j in jobs:
        j['duration'] = humanize_timedelta_msecs(j['duration'])
    page_context = {'batch': status, 'jobs': jobs}
    return await render_template('benchmark', request, userdata, 'batch.html', page_context)


@router.get('/batches/{batch_id}/jobs/{job_id}')
@auth.web_authenticated_developers_only()
async def get_job(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    batch_client = request.app['batch_client']
    job = await batch_client.get_job(batch_id, job_id)
    page_context = {
        'batch_id': batch_id,
        'job_id': job_id,
        'job_log': await job.log(),
        'job_status': json.dumps(await job.status(), indent=2),
        'attempts': await job.attempts(),
    }
    return await render_template('benchmark', request, userdata, 'job.html', page_context)


async def update_commits(app):
    github_client = app['github_client']

    request_string = f'/repos/hail-is/hail/commits?since={START_POINT}'
    log.info(f'start point is {START_POINT}')
    gh_data = await github_client.getitem(request_string)
    log.info(f'gh_data length is {len(gh_data)}')

    for gh_commit in gh_data:
        sha = gh_commit.get('sha')
        log.info(f'for commit {sha}')
        await update_commit(app, sha)

    log.info('got new commits')


async def get_commit(app, sha):  # pylint: disable=unused-argument
    log.info(f'get_commit sha={sha}')
    github_client = app['github_client']
    batch_client = app['batch_client']
    fs: aiotools.AsyncFS = app['fs']

    file_path = f'{BENCHMARK_RESULTS_PATH}/0-{sha}.json'
    request_string = f'/repos/hail-is/hail/commits/{sha}'
    gh_commit = await github_client.getitem(request_string)

    message = gh_commit['commit']['message']
    match = GH_COMMIT_MESSAGE_REGEX.search(message)
    message_dict = match.groupdict()
    pr_id = message_dict['pr_id']
    title = message_dict['title']

    has_results_file = await fs.exists(file_path)
    batch_statuses = [b._last_known_status async for b in batch_client.list_batches(q=f'sha={sha} user:benchmark')]
    complete_batch_statuses = [bs for bs in batch_statuses if bs['complete']]
    running_batch_statuses = [bs for bs in batch_statuses if not bs['complete']]

    if has_results_file:
        assert complete_batch_statuses, batch_statuses
        log.info(f'commit {sha} has a results file')
        status = complete_batch_statuses[0]
        batch_id = status['id']
        log.info(f'status of {sha}: {status}')
    elif running_batch_statuses:
        status = running_batch_statuses[0]
        batch_id = status['id']
        log.info(f'batch already exists for commit {sha}')
    else:
        status = None
        batch_id = None
        log.info(f'no batches or results file exists for {sha}')

    commit = {
        'sha': sha,
        'title': title,
        'author': gh_commit['commit']['author']['name'],
        'date': gh_commit['commit']['author']['date'],
        'status': status,
        'batch_id': batch_id,
        'pr_id': pr_id,
    }

    return commit


async def update_commit(app, sha):  # pylint: disable=unused-argument
    log.info('in update_commit')
    fs: aiotools.AsyncFS = app['fs']
    commit = await get_commit(app, sha)
    file_path = f'{BENCHMARK_RESULTS_PATH}/0-{sha}.json'

    if commit['status'] is None:
        batch_client = app['batch_client']
        batch_id = await submit_test_batch(batch_client, sha)
        batch = await batch_client.get_batch(batch_id)
        commit['status'] = batch._last_known_status
        commit['batch_id'] = batch_id
        log.info(f'submitted a batch {batch_id} for commit {sha}')
        benchmark_data['commits'][sha] = commit
        return commit

    has_results_file = await fs.exists(file_path)
    if has_results_file and sha in benchmark_data['commits']:
        benchmarks = await get_benchmarks(app, file_path)
        commit['geo_mean'] = benchmarks['geometric_mean']
        geo_mean = commit['geo_mean']
        log.info(f'geo mean is {geo_mean}')

        benchmark_data['dates'].append(commit['date'])
        benchmark_data['geo_means'].append(commit['geo_mean'])
        benchmark_data['pr_ids'].append(commit['pr_id'])
        benchmark_data['shas'].append(sha)
        benchmark_data['commits'][sha] = commit
    return commit


@router.get('/api/v1alpha/benchmark/commit/{sha}')
async def get_status(request):  # pylint: disable=unused-argument
    sha = str(request.match_info['sha'])
    app = request.app
    commit = await get_commit(app, sha)
    return web.json_response(commit)


@router.delete('/api/v1alpha/benchmark/commit/{sha}')
async def delete_commit(request):  # pylint: disable=unused-argument
    app = request.app
    fs: aiotools.AsyncFS = app['fs']
    batch_client = app['batch_client']
    sha = str(request.match_info['sha'])
    file_path = f'{BENCHMARK_RESULTS_PATH}/0-{sha}.json'

    if await fs.exists(file_path):
        await fs.remove(file_path)
        log.info(f'deleted file for sha {sha}')

    async for b in batch_client.list_batches(q=f'sha={sha} user:benchmark'):
        await b.delete()
        log.info(f'deleted batch for sha {sha}')

    if benchmark_data['commits'].get(sha):
        del benchmark_data['commits'][sha]
        log.info(f'deleted commit {sha} from commit list')

    return web.Response()


@router.post('/api/v1alpha/benchmark/commit/{sha}')
async def call_update_commit(request):  # pylint: disable=unused-argument
    body = await request.json()
    sha = body['sha']
    log.info('call_update_commit')
    commit = await update_commit(request.app, sha)
    return web.json_response(commit)


async def github_polling_loop(app):
    while True:
        await update_commits(app)
        log.info('successfully queried github')
        await asyncio.sleep(600)


async def on_startup(app):
    credentials = aiogoogle.GoogleCredentials.from_file('/benchmark-gsa-key/key.json')
    app['fs'] = aiogoogle.GoogleStorageAsyncFS(credentials=credentials)
    app['client_session'] = httpx.client_session()
    app['github_client'] = gidgethub.aiohttp.GitHubAPI(app['client_session'], 'hail-is/hail', oauth_token=oauth_token)
    app['batch_client'] = await bc.BatchClient.create(billing_project='benchmark')
    app['task_manager'] = aiotools.BackgroundTaskManager()
    app['task_manager'].ensure_future(retry_long_running('github_polling_loop', github_polling_loop, app))


async def on_cleanup(app):
    try:
        await app['client_session'].close()
    finally:
        try:
            await app['fs'].close()
        finally:
            app['task_manager'].shutdown()


def run():
    app = web.Application()
    setup_aiohttp_jinja2(app, 'benchmark')
    setup_aiohttp_session(app)

    setup_common_static_routes(router)
    router.static('/static', f'{BENCHMARK_ROOT}/static')
    app.add_routes(router)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(
        deploy_config.prefix_application(app, 'benchmark'),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=internal_server_ssl_context(),
    )
