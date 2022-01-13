import traceback
import json
import os
import logging
import asyncio
import concurrent.futures
from aiohttp import web
import aiohttp_session  # type: ignore
import uvloop  # type: ignore
from prometheus_async.aio.web import server_stats  # type: ignore
from gidgethub import aiohttp as gh_aiohttp, routing as gh_routing, sansio as gh_sansio
from hailtop.utils import collect_agen, humanize_timedelta_msecs
from hailtop.batch_client.aioclient import BatchClient, Batch
from hailtop.config import get_deploy_config
from hailtop.tls import internal_server_ssl_context
from hailtop.hail_logging import AccessLogger
from hailtop import aiotools, httpx
from gear import (
    setup_aiohttp_session,
    rest_authenticated_developers_only,
    web_authenticated_developers_only,
    check_csrf_token,
    create_database_pool,
    monitor_endpoints_middleware,
)
from typing import Dict, Any, Optional, List
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, render_template, set_message

from .environment import STORAGE_URI
from .github import Repo, FQBranch, WatchedBranch, UnwatchedBranch, MergeFailureBatch, PR, select_random_teammate
from .constants import AUTHORIZED_USERS, TEAMS

with open(os.environ.get('HAIL_CI_OAUTH_TOKEN', 'oauth-token/oauth-token'), 'r') as f:
    oauth_token = f.read().strip()

log = logging.getLogger('ci')

uvloop.install()

deploy_config = get_deploy_config()

watched_branches: List[WatchedBranch] = [
    WatchedBranch(index, FQBranch.from_short_str(bss), deployable, mergeable)
    for (index, [bss, deployable, mergeable]) in enumerate(json.loads(os.environ.get('HAIL_WATCHED_BRANCHES', '[]')))
]

routes = web.RouteTableDef()


async def pr_config(app, pr: PR) -> Dict[str, Any]:
    batch_id = pr.batch.id if pr.batch and hasattr(pr.batch, 'id') else None
    build_state = pr.build_state if await pr.authorized(app['dbpool']) else 'unauthorized'
    if build_state is None and batch_id is not None:
        build_state = 'building'
    return {
        'number': pr.number,
        'title': pr.title,
        # FIXME generate links to the merge log
        'batch_id': pr.batch.id if pr.batch and hasattr(pr.batch, 'id') else None,
        'build_state': build_state,
        'source_branch_name': pr.source_branch.name,
        'review_state': pr.review_state,
        'author': pr.author,
        'assignees': pr.assignees,
        'reviewers': pr.reviewers,
        'labels': pr.labels,
        'out_of_date': pr.build_state in ['failure', 'success', None] and not pr.is_up_to_date(),
    }


async def watched_branch_config(app, wb: WatchedBranch, index: int) -> Dict[str, Optional[Any]]:
    if wb.prs:
        pr_configs = [await pr_config(app, pr) for pr in wb.prs.values()]
    else:
        pr_configs = []
    # FIXME recent deploy history
    return {
        'index': index,
        'branch': wb.branch.short_str(),
        'sha': wb.sha,
        # FIXME generate links to the merge log
        'deploy_batch_id': wb.deploy_batch.id if wb.deploy_batch and hasattr(wb.deploy_batch, 'id') else None,
        'deploy_state': wb.deploy_state,
        'repo': wb.branch.repo.short_str(),
        'prs': pr_configs,
    }


@routes.get('')
@routes.get('/')
@web_authenticated_developers_only()
async def index(request, userdata):  # pylint: disable=unused-argument
    wb_configs = [await watched_branch_config(request.app, wb, i) for i, wb in enumerate(watched_branches)]
    page_context = {'watched_branches': wb_configs}
    return await render_template('ci', request, userdata, 'index.html', page_context)


def wb_and_pr_from_request(request):
    watched_branch_index = int(request.match_info['watched_branch_index'])
    pr_number = int(request.match_info['pr_number'])

    if watched_branch_index < 0 or watched_branch_index >= len(watched_branches):
        raise web.HTTPNotFound()
    wb = watched_branches[watched_branch_index]

    if not wb.prs or pr_number not in wb.prs:
        raise web.HTTPNotFound()
    return wb, wb.prs[pr_number]


@routes.get('/watched_branches/{watched_branch_index}/pr/{pr_number}')
@web_authenticated_developers_only()
async def get_pr(request, userdata):  # pylint: disable=unused-argument
    wb, pr = wb_and_pr_from_request(request)

    page_context = {}
    page_context['repo'] = wb.branch.repo.short_str()
    page_context['wb'] = wb
    page_context['pr'] = pr
    # FIXME
    batch = pr.batch
    if batch:
        if hasattr(batch, 'id'):
            status = await batch.last_known_status()
            jobs = await collect_agen(batch.jobs())
            for j in jobs:
                j['duration'] = humanize_timedelta_msecs(j['duration'])
            page_context['batch'] = status
            page_context['jobs'] = jobs
            artifacts_uri = f'{STORAGE_URI}/build/{batch.attributes["token"]}'
            page_context['artifacts_uri'] = artifacts_uri
            page_context['artifacts_url'] = storage_uri_to_url(artifacts_uri)
        else:
            page_context['exception'] = '\n'.join(
                traceback.format_exception(None, batch.exception, batch.exception.__traceback__)
            )

    batch_client = request.app['batch_client']
    target_branch = wb.branch.short_str()
    batches = batch_client.list_batches(f'test=1 ' f'pr={pr.number} ' f'target_branch={target_branch} ' f'user:ci')
    batches = sorted([b async for b in batches], key=lambda b: b.id, reverse=True)
    page_context['history'] = [await b.last_known_status() for b in batches]

    return await render_template('ci', request, userdata, 'pr.html', page_context)


def storage_uri_to_url(uri: str) -> str:
    if uri.startswith('gs://'):
        protocol = 'gs://'
        path = uri[len(protocol) :]
        return f'https://console.cloud.google.com/storage/browser/{path}'
    return uri


async def retry_pr(wb, pr, request):
    app = request.app
    session = await aiohttp_session.get_session(request)

    if pr.batch is None:
        log.info('retry cannot be requested for PR #{pr.number} because it has no batch')
        set_message(session, f'Retry cannot be requested for PR #{pr.number} because it has no batch.', 'error')
        return

    batch_id = pr.batch.id
    dbpool = app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('INSERT INTO invalidated_batches (batch_id) VALUES (%s);', batch_id)
    await wb.notify_batch_changed(app)

    log.info(f'retry requested for PR: {pr.number}')
    set_message(session, f'Retry requested for PR #{pr.number}.', 'info')


@routes.post('/watched_branches/{watched_branch_index}/pr/{pr_number}/retry')
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
async def post_retry_pr(request, userdata):  # pylint: disable=unused-argument
    wb, pr = wb_and_pr_from_request(request)

    await asyncio.shield(retry_pr(wb, pr, request))
    return web.HTTPFound(deploy_config.external_url('ci', f'/watched_branches/{wb.index}/pr/{pr.number}'))


@routes.get('/batches')
@web_authenticated_developers_only()
async def get_batches(request, userdata):
    batch_client = request.app['batch_client']
    batches = [b async for b in batch_client.list_batches()]
    statuses = [await b.last_known_status() for b in batches]
    page_context = {'batches': statuses}
    return await render_template('ci', request, userdata, 'batches.html', page_context)


@routes.get('/batches/{batch_id}')
@web_authenticated_developers_only()
async def get_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    batch_client = request.app['batch_client']
    b = await batch_client.get_batch(batch_id)
    status = await b.last_known_status()
    jobs = await collect_agen(b.jobs())
    for j in jobs:
        j['duration'] = humanize_timedelta_msecs(j['duration'])
    wb = get_maybe_wb_for_batch(b)
    page_context = {'batch': status, 'jobs': jobs, 'wb': wb}
    return await render_template('ci', request, userdata, 'batch.html', page_context)


def get_maybe_wb_for_batch(b: Batch):
    if 'target_branch' in b.attributes and 'pr' in b.attributes:
        branch = b.attributes['target_branch']
        wbs = [wb for wb in watched_branches if wb.branch.short_str() == branch]
        if len(wbs) == 0:
            pr = b.attributes['pr']
            log.exception(f"Attempted to load PR {pr} for unwatched branch {branch}")
        else:
            assert len(wbs) == 1
            return wbs[0].index
    return None


@routes.get('/batches/{batch_id}/jobs/{job_id}')
@web_authenticated_developers_only()
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
    return await render_template('ci', request, userdata, 'job.html', page_context)


def filter_wbs(wbs, pred):
    return [{**wb, 'prs': [pr for pr in wb['prs'] if pred(pr)]} for wb in wbs]


def is_pr_author(gh_username, pr_config):
    return gh_username == pr_config['author']


def is_pr_reviewer(gh_username, pr_config):
    return gh_username in pr_config['assignees'] or gh_username in pr_config['reviewers']


def pr_requires_action(gh_username, pr_config):
    build_state = pr_config['build_state']
    review_state = pr_config['review_state']
    return (
        is_pr_author(gh_username, pr_config) and (build_state == 'failure' or review_state == 'changes_requested')
    ) or (is_pr_reviewer(gh_username, pr_config) and review_state == 'pending')


@routes.get('/me')
@web_authenticated_developers_only()
async def get_user(request, userdata):
    username = userdata['username']
    gh_username = None
    pr_wbs = []
    review_wbs = []
    actionable_wbs = []
    for user in AUTHORIZED_USERS:
        if user.hail_username == username:
            gh_username = user.gh_username
            wbs = [await watched_branch_config(request.app, wb, i) for i, wb in enumerate(watched_branches)]
            pr_wbs = filter_wbs(wbs, lambda pr: is_pr_author(gh_username, pr))
            review_wbs = filter_wbs(wbs, lambda pr: is_pr_reviewer(gh_username, pr))
            actionable_wbs = filter_wbs(wbs, lambda pr: pr_requires_action(gh_username, pr))
            break

    batch_client = request.app['batch_client']
    dev_deploys = batch_client.list_batches(f'user={username} dev_deploy=1', limit=10)
    dev_deploys = sorted([b async for b in dev_deploys], key=lambda b: b.id, reverse=True)

    team_random_member = {team: select_random_teammate(team).gh_username for team in TEAMS}

    page_context = {
        'username': username,
        'gh_username': gh_username,
        'pr_wbs': pr_wbs,
        'review_wbs': review_wbs,
        'actionable_wbs': actionable_wbs,
        'team_member': team_random_member,
        'dev_deploys': [await b.last_known_status() for b in dev_deploys],
    }
    return await render_template('ci', request, userdata, 'user.html', page_context)


@routes.post('/authorize_source_sha')
@check_csrf_token
@web_authenticated_developers_only(redirect=False)
async def post_authorized_source_sha(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    dbpool = app['dbpool']
    post = await request.post()
    sha = post['sha'].strip()
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('INSERT INTO authorized_shas (sha) VALUES (%s);', sha)
    log.info(f'authorized sha: {sha}')
    session = await aiohttp_session.get_session(request)
    set_message(session, f'SHA {sha} authorized.', 'info')
    return web.HTTPFound(deploy_config.external_url('ci', '/'))


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
        branch_name = ref[len('refs/heads/') :]
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
    app = request.app
    params = await request.json()
    log.info(f'batch callback {params}')
    attrs = params.get('attributes')
    if attrs:
        target_branch = attrs.get('target_branch')
        if target_branch:
            for wb in watched_branches:
                if wb.branch.short_str() == target_branch:
                    log.info(f'watched_branch {wb.branch.short_str()} notify batch changed')
                    await wb.notify_batch_changed(app)


@routes.get('/api/v1alpha/deploy_status')
@rest_authenticated_developers_only
async def deploy_status(request, userdata):  # pylint: disable=unused-argument
    batch_client = request.app['batch_client']

    async def get_failure_information(batch):
        if isinstance(batch, MergeFailureBatch):
            exc = batch.exception
            return traceback.format_exception(etype=type(exc), value=exc, tb=exc.__traceback__)
        jobs = await collect_agen(batch.jobs())

        async def fetch_job_and_log(j):
            full_job = await batch_client.get_job(j['batch_id'], j['job_id'])
            log = await full_job.log()
            return {**full_job._status, 'log': log}

        return await asyncio.gather(*[fetch_job_and_log(j) for j in jobs if j['state'] in ('Error', 'Failed')])

    wb_configs = [
        {
            'branch': wb.branch.short_str(),
            'sha': wb.sha,
            'deploy_batch_id': wb.deploy_batch.id if wb.deploy_batch and hasattr(wb.deploy_batch, 'id') else None,
            'deploy_state': wb.deploy_state,
            'repo': wb.branch.repo.short_str(),
            'failure_information': None
            if wb.deploy_state == 'success'
            else await get_failure_information(wb.deploy_batch),
        }
        for wb in watched_branches
    ]
    return web.json_response(wb_configs)


@routes.post('/api/v1alpha/update')
@rest_authenticated_developers_only
async def post_update(request, userdata):  # pylint: disable=unused-argument
    log.info('developer triggered update')

    async def update_all():
        for wb in watched_branches:
            await wb.update(request.app)

    request.app['task_manager'].ensure_future(update_all())
    return web.Response(status=200)


@routes.post('/api/v1alpha/dev_deploy_branch')
@rest_authenticated_developers_only
async def dev_deploy_branch(request, userdata):
    app = request.app
    try:
        params = await request.json()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        message = 'could not read body as JSON'
        log.info('dev deploy failed: ' + message, exc_info=True)
        raise web.HTTPBadRequest(text=message) from e

    try:
        branch = FQBranch.from_short_str(params['branch'])
        steps = params['steps']
        excluded_steps = params['excluded_steps']
        extra_config = params.get('extra_config', {})
    except asyncio.CancelledError:
        raise
    except Exception as e:
        message = f'parameters are wrong; check the branch and steps syntax.\n\n{params}'
        log.info('dev deploy failed: ' + message, exc_info=True)
        raise web.HTTPBadRequest(text=message) from e

    gh = app['github_client']
    request_string = f'/repos/{branch.repo.owner}/{branch.repo.name}/git/refs/heads/{branch.name}'

    try:
        branch_gh_json = await gh.getitem(request_string)
        sha = branch_gh_json['object']['sha']
    except asyncio.CancelledError:
        raise
    except Exception as e:
        message = f'error finding {branch} at GitHub'
        log.info('dev deploy failed: ' + message, exc_info=True)
        raise web.HTTPBadRequest(text=message) from e

    unwatched_branch = UnwatchedBranch(branch, sha, userdata, extra_config)

    batch_client = app['batch_client']

    try:
        batch_id = await unwatched_branch.deploy(batch_client, steps, excluded_steps=excluded_steps)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # pylint: disable=broad-except
        message = traceback.format_exc()
        raise web.HTTPBadRequest(text=f'starting the deploy failed due to\n{message}') from e
    return web.json_response({'sha': sha, 'batch_id': batch_id})


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
        except Exception:  # pylint: disable=broad-except
            log.exception(f'{wb.branch.short_str()} update failed due to exception')
        await asyncio.sleep(300)


async def on_startup(app):
    app['client_session'] = httpx.client_session()
    app['github_client'] = gh_aiohttp.GitHubAPI(app['client_session'], 'ci', oauth_token=oauth_token)
    app['batch_client'] = await BatchClient.create('ci')
    app['dbpool'] = await create_database_pool()

    app['task_manager'] = aiotools.BackgroundTaskManager()
    app['task_manager'].ensure_future(update_loop(app))


async def on_cleanup(app):
    try:
        dbpool = app['dbpool']
        dbpool.close()
        await dbpool.wait_closed()
        await app['client_session'].close()
        await app['batch_client'].close()
    finally:
        app['task_manager'].shutdown()


def run():
    app = web.Application(middlewares=[monitor_endpoints_middleware])
    setup_aiohttp_jinja2(app, 'ci')
    setup_aiohttp_session(app)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    setup_common_static_routes(routes)
    app.add_routes(routes)
    app.router.add_get("/metrics", server_stats)

    web.run_app(
        deploy_config.prefix_application(app, 'ci'),
        host='0.0.0.0',
        port=5000,
        access_log_class=AccessLogger,
        ssl_context=internal_server_ssl_context(),
    )
