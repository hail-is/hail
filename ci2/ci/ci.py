import traceback
import json
import os
import uvloop
import asyncio
import aiodns
import aiohttp
from aiohttp import web
import jinja2
import aiohttp_jinja2
from gidgethub import aiohttp as gh_aiohttp
import concurrent.futures
import batch

from log import log
from github import *

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
async def index(request):
    return {
        'watched_branches': [
            {
                'branch': wb.branch.short_str(),
                'sha': wb.sha,
                'repo': wb.branch.repo.short_str(),
                'prs': [
                    {
                        'number': pr.number,
                        'title': pr.title,
                        'batch_id': pr.batch.id if pr.batch else None,
                        'passing': pr.passing,
                        'state': pr.state
                    }
                    for pr in wb.prs.values()
                ] if wb.prs else None}
            for wb in watched_branches
        ]}

@routes.get('/batches/{batch_id}/log')
@aiohttp_jinja2.template('batch_log.html')
async def get_batch_log(request):
    batch_id = int(request.match_info['batch_id'])

    batch_client = request.app['batch_client']
    batch = await batch_client.get_batch(batch_id)
    status = await batch.status()
    print('batch_status', status)
    return {
        'batch_id': batch_id,
        'batch_status': status
    }

@routes.get('/healthcheck')
async def healthcheck(request):
    return web.Response(status=200)

async def refresh_loop(app):
    gh = app['github_client']
    batch = app['batch_client']
    while True:
        try:
            for wb in watched_branches:
                log.info(f'refreshing {wb.branch}')
                await wb.refresh(gh)
                await wb.heal(batch)
            await asyncio.sleep(60)
        except concurrent.futures.CancelledError:
            raise
        except Exception as e:
            log.error(f'{wb.branch} refresh due to exception: {traceback.format_exc()}{e}')

app.add_routes(routes)

aiohttp_jinja2.setup(app,
    loader=jinja2.FileSystemLoader('ci/templates'))

async def on_startup(app):
    asyncio.ensure_future(refresh_loop(app))

app.on_startup.append(on_startup)

async def on_cleanup(app):
    session = app['client_session']
    await session.close()

app.on_cleanup.append(on_cleanup)

web.run_app(app, host='0.0.0.0', port=5000)
