import traceback
import json
import os
import logging
import asyncio
import concurrent.futures
import datetime
import aiohttp
from aiohttp import web
import aiomysql
import uvloop
import jinja2
import humanize
import aiohttp_jinja2
from gidgethub import aiohttp as gh_aiohttp, routing as gh_routing, sansio as gh_sansio

from hailtop.batch_client.aioclient import BatchClient, Job
from hailtop.gear.auth import web_authenticated_developers_only, rest_authenticated_developers_only, new_csrf_token, check_csrf_token
from hailtop import gear
from .constants import BUCKET
from .github import Repo, FQBranch, WatchedBranch, UnwatchedBranch

with open(os.environ.get('HAIL_CI_OAUTH_TOKEN', 'oauth-token/oauth-token'), 'r') as f:
    oauth_token = f.read().strip()

gear.configure_logging()
log = logging.getLogger('ci')

uvloop.install()

watched_branches = [
    WatchedBranch(index, FQBranch.from_short_str(bss), deployable)
    for (index, [bss, deployable]) in enumerate(json.loads(os.environ.get('HAIL_WATCHED_BRANCHES', '[]')))
]

app = web.Application()

routes = web.RouteTableDef()

start_time = datetime.datetime.now()


@routes.get('/')
@web_authenticated_developers_only
async def index(request):  # pylint: disable=unused-argument
    app = request.app
    dbpool = app['dbpool']
    wb_configs = []
    for i, wb in enumerate(watched_branches):
        if wb.prs:
            pr_configs = []
            for pr in wb.prs.values():
                batch_id = pr.batch.id if pr.batch and hasattr(pr.batch, 'id') else None
                build_state = pr.build_state if await pr.authorized(dbpool) else 'unauthorized'
                if build_state is None and batch_id is not None:
                    build_state = 'building'

                pr_config = {
                    'number': pr.number,
                    'title': pr.title,
                    # FIXME generate links to the merge log
                    'batch_id': pr.batch.id if pr.batch and hasattr(pr.batch, 'id') else None,
                    'build_state': build_state,
                    'review_state': pr.review_state,
                    'author': pr.author,
                    'out_of_date': pr.build_state in ['failure', 'success', None] and not pr.is_up_to_date(),
                    'status_age': pr.pretty_status_age(),
                }
                pr_configs.append(pr_config)
        else:
            pr_configs = None
        # FIXME recent deploy history
        wb_config = {
            'index': i,
            'branch': wb.branch.short_str(),
            'sha': wb.sha,
            # FIXME generate links to the merge log
            'deploy_batch_id': wb.deploy_batch.id if wb.deploy_batch and hasattr(wb.deploy_batch, 'id') else None,
            'deploy_state': wb.deploy_state,
            'repo': wb.branch.repo.short_str(),
            'prs': pr_configs,
            'status_age': wb.pretty_status_age(),
        }
        wb_configs.append(wb_config)

    token = new_csrf_token()

    context = {
        'watched_branches': wb_configs,
        'age': humanize.naturaldelta(datetime.datetime.now() - start_time),
        'token': token
    }

    response = aiohttp_jinja2.render_template('index.html',
                                              request,
                                              context)
    response.set_cookie('_csrf', token, secure=True, httponly=True)
    return response


@routes.get('/watched_branches/{watched_branch_index}/pr/{pr_number}')
@web_authenticated_developers_only
@aiohttp_jinja2.template('pr.html')
async def get_pr(request):
    watched_branch_index = int(request.match_info['watched_branch_index'])
    pr_number = int(request.match_info['pr_number'])

    if watched_branch_index < 0 or watched_branch_index >= len(watched_branches):
        raise web.HTTPNotFound()
    wb = watched_branches[watched_branch_index]

    if not wb.prs or pr_number not in wb.prs:
        raise web.HTTPNotFound()
    pr = wb.prs[pr_number]

    config = {}
    config['number'] = pr.number
    # FIXME
    if pr.batch:
        if hasattr(pr.batch, 'id'):
            status = await pr.batch.status()
            for j in status['jobs']:
                j['duration'] = humanize.naturaldelta(Job.total_duration(j))
                j['exit_code'] = Job.exit_code(j)
                attrs = j['attributes']
                if 'link' in attrs:
                    attrs['link'] = attrs['link'].split(',')
            config['batch'] = status
            config['artifacts'] = f'{BUCKET}/build/{pr.batch.attributes["token"]}'
        else:
            config['exception'] = '\n'.join(
                traceback.format_exception(None, pr.batch.exception, pr.batch.exception.__traceback__))

    batch_client = request.app['batch_client']
    batches = await batch_client.list_batches(
        attributes={
            'test': '1',
            'pr': pr_number
        })
    batches = sorted(batches, key=lambda b: b.id, reverse=True)
    config['history'] = [await b.status() for b in batches]

    return config


@routes.get('/batches')
@web_authenticated_developers_only
@aiohttp_jinja2.template('batches.html')
async def get_batches(request):
    batch_client = request.app['batch_client']
    batches = await batch_client.list_batches()
    statuses = [await b.status() for b in batches]
    return {
        'batches': statuses
    }


@routes.get('/batches/{batch_id}')
@web_authenticated_developers_only
@aiohttp_jinja2.template('batch.html')
async def get_batch(request):
    batch_id = int(request.match_info['batch_id'])
    batch_client = request.app['batch_client']
    b = await batch_client.get_batch(batch_id)
    status = await b.status()
    for j in status['jobs']:
        j['duration'] = humanize.naturaldelta(Job.total_duration(j))
        j['exit_code'] = Job.exit_code(j)
    return {
        'batch': status
    }


@routes.get('/batches/{batch_id}/jobs/{job_id}/log')
@web_authenticated_developers_only
@aiohttp_jinja2.template('job_log.html')
async def get_job_log(request):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    batch_client = request.app['batch_client']
    job = await batch_client.get_job(batch_id, job_id)
    return {
        'batch_id': batch_id,
        'job_id': job_id,
        'job_log': await job.log()
    }


@routes.post('/authorize_source_sha')
@check_csrf_token
@web_authenticated_developers_only
async def post_authorized_source_sha(request):
    app = request.app
    dbpool = app['dbpool']
    post = await request.post()
    sha = post['sha'].strip()
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('INSERT INTO authorized_shas (sha) VALUES (%s);', sha)
    log.info(f'authorized sha: {sha}')
    raise web.HTTPFound('/')


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
        if (wb.prs and number in wb.prs) or (wb.branch == target_branch):
            await wb.notify_github_changed(event.app)


@gh_router.register('push')
async def push_callback(event):
    data = event.data
    ref = data['ref']
    if ref.startswith('refs/heads/'):
        branch_name = ref[len('refs/heads/'):]
        branch = FQBranch(Repo.from_gh_json(data['repository']), branch_name)
        for wb in watched_branches:
            if wb.branch == branch or any(pr.branch == branch for pr in wb.prs.values()):
                await wb.notify_github_changed(event.app)


@gh_router.register('pull_request_review')
async def pull_request_review_callback(event):
    gh_pr = event.data['pull_request']
    number = gh_pr['number']
    for wb in watched_branches:
        if number in wb.prs:
            await wb.notify_github_changed(event.app)


async def github_callback_handler(request):
    event = gh_sansio.Event.from_http(request.headers, await request.read())
    event.app = request.app
    await gh_router.dispatch(event)


@routes.post('/github_callback')
async def github_callback(request):
    await asyncio.shield(github_callback_handler(request))
    return web.Response(status=200)


async def batch_callback_handler(request):
    params = await request.json()
    log.info(f'batch callback {params}')
    attrs = params.get('attributes')
    if attrs:
        target_branch = attrs.get('target_branch')
        if target_branch:
            for wb in watched_branches:
                if wb.branch.short_str() == target_branch:
                    log.info(f'watched_branch {wb.branch.short_str()} notify batch changed')
                    await wb.notify_batch_changed()

@routes.post('/api/v1alpha/dev_test_branch/')
@rest_authenticated_developers_only
async def dev_test_branch(request):
    params = await request.json()
    userdata = params['userdata']
    repo = Repo.from_short_str(params['repo'])
    fq_branch = FQBranch(repo, params.get('branch'))
    namespace = params['namespace']
    profile = params['profile']

    gh = app['github_client']
    request_string = f'/repos/{repo.owner}/{repo.name}/git/refs/heads/{fq_branch.name}'
    branch_gh_json = await gh.getitem(request_string)
    sha = branch_gh_json['object']['sha']

    unwatched_branch = UnwatchedBranch(fq_branch, sha, userdata, namespace)

    batch_client = app['batch_client']

    batch_id = await unwatched_branch.deploy(batch_client, profile)

    if batch_id is not None:
        return web.Response(status=200, text=str(batch_id))
    else:
        return web.Response(status=503)

@routes.post('/api/v1alpha/batch_callback')
async def batch_callback(request):
    await asyncio.shield(batch_callback_handler(request))
    return web.Response(status=200)


async def update_loop(app):
    while True:
        try:
            for wb in watched_branches:
                log.info(f'updating {wb.branch.short_str()}')
                await wb.update(app)
        except concurrent.futures.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            log.error(f'{wb.branch.short_str()} update failed due to exception: {traceback.format_exc()}{e}')
        await asyncio.sleep(300)


aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('ci/templates'))


async def on_startup(app):
    app['client_session'] = aiohttp.ClientSession(
        raise_for_status=True,
        timeout=aiohttp.ClientTimeout(total=60))
    app['github_client'] = gh_aiohttp.GitHubAPI(app['client_session'], 'ci', oauth_token=oauth_token)
    app['batch_client'] = BatchClient(app['client_session'], url=os.environ.get('BATCH_SERVER_URL'))

    with open('/ci-user-secret/sql-config.json', 'r') as f:
        config = json.loads(f.read().strip())
        app['dbpool'] = await aiomysql.create_pool(host=config['host'],
                                                   port=config['port'],
                                                   db=config['db'],
                                                   user=config['user'],
                                                   password=config['password'],
                                                   charset='utf8',
                                                   cursorclass=aiomysql.cursors.DictCursor,
                                                   autocommit=True)

    asyncio.ensure_future(update_loop(app))



async def on_cleanup(app):
    session = app['client_session']
    await session.close()

    dbpool = app['dbpool']
    dbpool.close()
    await dbpool.wait_closed()



def run():
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.add_routes(routes)
    routes.static('/static', 'ci/static')
    web.run_app(app, host='0.0.0.0', port=5000)
