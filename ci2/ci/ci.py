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
                        'state': pr.state
                    }
                    for pr in wb.prs.values()
                ]}
            for wb in watched_branches
        ]}

@routes.get('/healthcheck')
async def healthcheck(request):
    return web.Response(status=200)

async def refresh_loop():
    async with aiohttp.ClientSession() as session:
        # FIXME
        gh = gh_aiohttp.GitHubAPI(session, 'ci2', oauth_token=oauth_token)
        while True:
            try:
                for wb in watched_branches:
                    log.info(f'refreshing {wb.branch}')
                    await wb.refresh(gh)
                await asyncio.sleep(60)
            except concurrent.futures.CancelledError:
                raise
            except Exception as e:
                log.error(f'{wb.branch} refresh due to exception: {traceback.format_exc()}{e}')

app.add_routes(routes)

aiohttp_jinja2.setup(app,
    loader=jinja2.FileSystemLoader('ci/templates'))

async def on_startup(app):
    asyncio.ensure_future(refresh_loop())

app.on_startup.append(on_startup)

web.run_app(app, host='0.0.0.0', port=5000)
