import traceback
import json
import os
import asyncio
import concurrent.futures
import datetime
import aiohttp
from aiohttp import web
import uvloop
import jinja2
import humanize
import aiohttp_jinja2
from gidgethub import aiohttp as gh_aiohttp, routing as gh_routing, sansio as gh_sansio
import batch

from .log import log
from .constants import BUCKET
from .github import Repo, FQBranch, WatchedBranch

with open(os.environ.get('CI2_OAUTH_TOKEN', 'oauth-token/oauth-token'), 'r') as f:
    oauth_token = f.read().strip()

uvloop.install()

watched_branches = [
    WatchedBranch(FQBranch.from_short_str(bss))
    for bss in json.loads(os.environ.get('WATCHED_BRANCHES'))
]

app = web.Application()

app['client_session'] = aiohttp.ClientSession(
    raise_for_status=True,
    timeout=aiohttp.ClientTimeout(total=60))
app['github_client'] = gh_aiohttp.GitHubAPI(app['client_session'], 'ci2', oauth_token=oauth_token)
app['batch_client'] = batch.aioclient.BatchClient(app['client_session'], url=os.environ.get('BATCH_SERVER_URL'))

routes = web.RouteTableDef()


@routes.get('/')
@aiohttp_jinja2.template('index.html')
async def index(request):  # pylint: disable=unused-argument
    return {
        'watched_branches': [
            {
                'index': i,
                'branch': wb.branch.short_str(),
                'sha': wb.sha,
                'repo': wb.branch.repo.short_str(),
                'prs': [
                    {
                        'number': pr.number,
                        'title': pr.title,
                        'batch_id': pr.batch.id if pr.batch else None,
                        'build_state': pr.build_state,
                        'review_state': pr.review_state
                    }
                    for pr in wb.prs.values()
                ] if wb.prs else None}
            for i, wb in enumerate(watched_branches)
        ]}


@routes.get('/watched_branches/{watched_branch_index}/pr/{pr_number}')
@aiohttp_jinja2.template('pr.html')
async def get_pr(request):
    watched_branch_index = int(request.match_info['watched_branch_index'])
    pr_number = int(request.match_info['pr_number'])
    try:
        wb = watched_branches[watched_branch_index]
        pr = wb.prs[pr_number]
    except IndexError:
        raise web.HTTPNotFound()

    config = {}
    config['number'] = pr.number
    if pr.batch:
        status = await pr.batch.status()
        for j in status['jobs']:
            if 'duration' in j:
                j['duration'] = humanize.naturaldelta(datetime.timedelta(seconds=j['duration']))
            attrs = j['attributes']
            if 'link' in attrs:
                attrs['link'] = attrs['link'].split(',')
        config['batch'] = status
        config['artifacts'] = f'{BUCKET}/build/{pr.batch.attributes["token"]}'

    return config


@routes.get('/jobs/{job_id}/log')
@aiohttp_jinja2.template('job_log.html')
async def get_job_log(request):
    job_id = int(request.match_info['job_id'])
    batch_client = request.app['batch_client']
    job = await batch_client.get_job(job_id)
    return {
        'job_id': job_id,
        'job_log': await job.log()
    }


@routes.get('/healthcheck')
async def healthcheck(request):  # pylint: disable=unused-argument
    return web.Response(status=200)

gh_router = gh_routing.Router()


@gh_router.register('pull_request')
async def pull_request_callback(event):
    gh_pr = event.data['pull_request']
    number = gh_pr['number']
    target_branch = FQBranch.from_gh_json(gh_pr['base'])
    for wb in watched_branches:
        if (number in wb.prs) or (wb.branch == target_branch):
            await wb.update(event.app)


@gh_router.register('push')
async def push_callback(event):
    data = event.data
    ref = data['ref']
    if ref.startswith('refs/heads/'):
        branch_name = ref[len('refs/heads/'):]
        branch = FQBranch(Repo.from_gh_json(data['repository']), branch_name)
        for wb in watched_branches:
            if wb.branch == branch or any(pr.branch == branch for pr in wb.prs.values()):
                await wb.update(event.app)


@gh_router.register('pull_request_review')
async def pull_request_review_callback(event):
    gh_pr = event.data['pull_request']
    number = gh_pr['number']
    for wb in watched_branches:
        if number in wb.prs:
            await wb.update(event.app)


@routes.post('/callback')
async def callback(request):
    event = gh_sansio.Event.from_http(request.headers, await request.read())
    event.app = request.app
    await gh_router.dispatch(event)
    return web.Response(status=200)


async def refresh_loop(app):
    while True:
        try:
            for wb in watched_branches:
                log.info(f'refreshing {wb.branch}')
                await wb.update(app)
        except concurrent.futures.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            log.error(f'{wb.branch} refresh due to exception: {traceback.format_exc()}{e}')
        await asyncio.sleep(300)

routes.static('/static', 'ci/static')
app.add_routes(routes)

aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('ci/templates'))


async def on_startup(app):
    asyncio.ensure_future(refresh_loop(app))

app.on_startup.append(on_startup)


async def on_cleanup(app):
    session = app['client_session']
    await session.close()

app.on_cleanup.append(on_cleanup)

def run():
    web.run_app(app, host='0.0.0.0', port=5000)
