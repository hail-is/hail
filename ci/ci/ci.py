import asyncio
import concurrent.futures
import json
import logging
import os
import traceback
from contextlib import AsyncExitStack
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Tuple, TypedDict

import aiohttp_session  # type: ignore
import kubernetes_asyncio
import kubernetes_asyncio.client
import kubernetes_asyncio.config
import uvloop  # type: ignore
import yaml
from aiohttp import web
from gidgethub import aiohttp as gh_aiohttp
from gidgethub import routing as gh_routing
from gidgethub import sansio as gh_sansio
from prometheus_async.aio.web import server_stats  # type: ignore

from gear import (
    AuthServiceAuthenticator,
    CommonAiohttpAppKeys,
    Database,
    UserData,
    check_csrf_token,
    json_request,
    json_response,
    monitor_endpoints_middleware,
    setup_aiohttp_session,
)
from gear.profiling import install_profiler_if_requested
from hailtop import aiotools, httpx
from hailtop.auth import hail_credentials
from hailtop.batch_client.aioclient import Batch, BatchClient
from hailtop.config import get_deploy_config
from hailtop.hail_logging import AccessLogger
from hailtop.utils import collect_aiter, humanize_timedelta_msecs, periodically_call, retry_transient_errors
from web_common import render_template, set_message, setup_aiohttp_jinja2, setup_common_static_routes

from .constants import AUTHORIZED_USERS, TEAMS
from .environment import CLOUD, DEFAULT_NAMESPACE, STORAGE_URI
from .envoy import create_cds_response, create_rds_response
from .github import PR, WIP, FQBranch, MergeFailureBatch, Repo, UnwatchedBranch, WatchedBranch, select_random_teammate
from .utils import gcp_logging_queries

with open(os.environ.get('HAIL_CI_OAUTH_TOKEN', 'oauth-token/oauth-token'), 'r', encoding='utf-8') as f:
    oauth_token = f.read().strip()

log = logging.getLogger('ci')

uvloop.install()

deploy_config = get_deploy_config()

watched_branches: List[WatchedBranch] = []

routes = web.RouteTableDef()

auth = AuthServiceAuthenticator()


class PRConfig(TypedDict):
    number: int
    title: str
    batch_id: Optional[int]
    build_state: Optional[str]
    gh_statuses: Dict[str, str]
    source_branch_name: str
    review_state: Optional[str]
    author: str
    assignees: Set[str]
    reviewers: Set[str]
    labels: Set[str]
    out_of_date: bool


async def pr_config(app: web.Application, pr: PR) -> PRConfig:
    batch_id = pr.batch.id if pr.batch and isinstance(pr.batch, Batch) else None
    build_state = pr.build_state if await pr.authorized(app[AppKeys.DB]) else 'unauthorized'
    if build_state is None and batch_id is not None:
        build_state = 'building'
    return {
        'number': pr.number,
        'title': pr.title,
        # FIXME generate links to the merge log
        'batch_id': batch_id,
        'build_state': build_state,
        'gh_statuses': {k: v.value for k, v in pr.last_known_github_status.items()},
        'source_branch_name': pr.source_branch.name,
        'review_state': pr.review_state,
        'author': pr.author,
        'assignees': pr.assignees,
        'reviewers': pr.reviewers,
        'labels': pr.labels,
        'out_of_date': pr.build_state in ['failure', 'success', None] and not pr.is_up_to_date(),
    }


class WatchedBranchConfig(TypedDict):
    index: int
    branch: str
    sha: Optional[str]
    deploy_batch_id: Optional[int]
    deploy_state: Optional[str]
    repo: str
    prs: List[PRConfig]
    gh_status_names: Set[str]
    merge_candidate: Optional[str]


async def watched_branch_config(app: web.Application, wb: WatchedBranch, index: int) -> WatchedBranchConfig:
    if wb.prs:
        pr_configs = [await pr_config(app, pr) for pr in wb.prs.values()]
    else:
        pr_configs = []
    # FIXME recent deploy history
    gh_status_names = {k for pr in pr_configs for k in pr['gh_statuses'].keys()}
    return {
        'index': index,
        'branch': wb.branch.short_str(),
        'sha': wb.sha,
        # FIXME generate links to the merge log
        'deploy_batch_id': wb.deploy_batch.id if wb.deploy_batch and isinstance(wb.deploy_batch, Batch) else None,
        'deploy_state': wb.deploy_state,
        'repo': wb.branch.repo.short_str(),
        'prs': pr_configs,
        'gh_status_names': gh_status_names,
        'merge_candidate': wb.merge_candidate.short_str() if wb.merge_candidate else None,
    }


@routes.get('')
@routes.get('/')
@auth.authenticated_developers_only()
async def index(request: web.Request, userdata: UserData) -> web.Response:
    wb_configs = [await watched_branch_config(request.app, wb, i) for i, wb in enumerate(watched_branches)]
    page_context = {
        'watched_branches': wb_configs,
        'frozen_merge_deploy': request.app[AppKeys.FROZEN_MERGE_DEPLOY],
    }
    return await render_template('ci', request, userdata, 'index.html', page_context)


def wb_and_pr_from_request(request: web.Request) -> Tuple[WatchedBranch, PR]:
    watched_branch_index = int(request.match_info['watched_branch_index'])
    pr_number = int(request.match_info['pr_number'])

    if watched_branch_index < 0 or watched_branch_index >= len(watched_branches):
        raise web.HTTPNotFound()
    wb = watched_branches[watched_branch_index]

    if not wb.prs or pr_number not in wb.prs:
        raise web.HTTPNotFound()
    return wb, wb.prs[pr_number]


def filter_jobs(jobs):
    filtered: Dict[str, list] = {
        state: []
        # the order of this list is the order in which the states will be displayed on the page
        for state in ["failed", "error", "cancelled", "running", "pending", "ready", "creating", "success"]
    }
    for job in jobs:
        filtered[job["state"].lower()].append(job)
    return {"jobs": filtered}


@routes.get('/watched_branches/{watched_branch_index}/pr/{pr_number}')
@auth.authenticated_developers_only()
async def get_pr(request: web.Request, userdata: UserData) -> web.Response:
    wb, pr = wb_and_pr_from_request(request)

    page_context: Dict[str, Any] = {}
    page_context['repo'] = wb.branch.repo.short_str()
    page_context['wb'] = wb
    page_context['pr'] = pr
    # FIXME
    batch = pr.batch
    if batch:
        if isinstance(batch, Batch):
            status = await batch.last_known_status()
            jobs = await collect_aiter(batch.jobs())
            for j in jobs:
                j['duration'] = humanize_timedelta_msecs(j['duration'])
            page_context['batch'] = status
            page_context.update(filter_jobs(jobs))
            artifacts_uri = f'{STORAGE_URI}/build/{batch.attributes["token"]}'
            page_context['artifacts_uri'] = artifacts_uri
            page_context['artifacts_url'] = storage_uri_to_url(artifacts_uri)

            if CLOUD == 'gcp':
                start_time = status['time_created']
                end_time = status['time_completed']
                assert start_time is not None
                page_context['logging_queries'] = gcp_logging_queries(
                    batch.attributes['namespace'], start_time, end_time
                )
            else:
                page_context['logging_queries'] = None
        else:
            page_context['exception'] = '\n'.join(
                traceback.format_exception(None, batch.exception, batch.exception.__traceback__)
            )

    batch_client = request.app[AppKeys.BATCH_CLIENT]
    target_branch = wb.branch.short_str()
    batches = batch_client.list_batches(f'test=1 ' f'pr={pr.number} ' f'target_branch={target_branch} ' f'user:ci')
    batches = sorted([b async for b in batches], key=lambda b: b.id, reverse=True)
    page_context['history'] = [await b.last_known_status() for b in batches]

    return await render_template('ci', request, userdata, 'pr.html', page_context)


def storage_uri_to_url(uri: str) -> str:
    if uri.startswith('gs://'):
        path = uri.removeprefix('gs://')
        return f'https://console.cloud.google.com/storage/browser/{path}'
    return uri


async def retry_pr(wb: WatchedBranch, pr: PR, request: web.Request):
    app = request.app
    session = await aiohttp_session.get_session(request)

    if pr.batch is None:
        log.info(f'retry cannot be requested for PR #{pr.number} because it has no batch')
        set_message(session, f'Retry cannot be requested for PR #{pr.number} because it has no batch.', 'error')
        return

    if isinstance(pr.batch, MergeFailureBatch):
        log.info(f'retry cannot be requested for PR #{pr.number} because it was a merge failure')
        set_message(session, f'Retry cannot be requested for PR #{pr.number} because it was a merge failure.', 'error')
        return

    batch_id = pr.batch.id
    db = app[AppKeys.DB]
    await db.execute_insertone('INSERT INTO invalidated_batches (batch_id) VALUES (%s);', batch_id)
    await wb.notify_batch_changed(
        db, app[AppKeys.BATCH_CLIENT], app[AppKeys.GH_CLIENT], app[AppKeys.FROZEN_MERGE_DEPLOY]
    )

    log.info(f'retry requested for PR: {pr.number}')
    set_message(session, f'Retry requested for PR #{pr.number}.', 'info')


@routes.post('/watched_branches/{watched_branch_index}/pr/{pr_number}/retry')
@auth.authenticated_developers_only(redirect=False)
async def post_retry_pr(request: web.Request, _) -> NoReturn:
    wb, pr = wb_and_pr_from_request(request)

    await asyncio.shield(retry_pr(wb, pr, request))
    raise web.HTTPFound(deploy_config.external_url('ci', f'/watched_branches/{wb.index}/pr/{pr.number}'))


@routes.get('/batches')
@auth.authenticated_developers_only()
async def get_batches(request: web.Request, userdata: UserData):
    batch_client = request.app[AppKeys.BATCH_CLIENT]
    batches = [b async for b in batch_client.list_batches()]
    statuses = [await b.last_known_status() for b in batches]
    page_context = {'batches': statuses}
    return await render_template('ci', request, userdata, 'batches.html', page_context)


@routes.get('/batches/{batch_id}')
@auth.authenticated_developers_only()
async def get_batch(request: web.Request, userdata: UserData):
    batch_id = int(request.match_info['batch_id'])
    batch_client = request.app[AppKeys.BATCH_CLIENT]
    b = await batch_client.get_batch(batch_id)
    status = await b.last_known_status()
    jobs = await collect_aiter(b.jobs())
    for j in jobs:
        j['duration'] = humanize_timedelta_msecs(j['duration'])
    wb = get_maybe_wb_for_batch(b)
    page_context = {'batch': status, 'wb': wb}
    page_context.update(filter_jobs(jobs))
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


def filter_wbs(wbs: List[WatchedBranchConfig], pred: Callable[[PRConfig], bool]):
    return [{**wb, 'prs': [pr for pr in wb['prs'] if pred(pr)]} for wb in wbs]


def is_pr_author(gh_username: str, pr_config: PRConfig) -> bool:
    return gh_username == pr_config['author']


def is_pr_reviewer(gh_username: str, pr_config: PRConfig) -> bool:
    return gh_username in pr_config['assignees'] or gh_username in pr_config['reviewers']


def pr_requires_action(gh_username: str, pr_config: PRConfig) -> bool:
    build_state = pr_config['build_state']
    review_state = pr_config['review_state']
    return (
        is_pr_author(gh_username, pr_config)
        and (build_state == 'failure' or review_state == 'changes_requested' or WIP in pr_config['labels'])
    ) or (is_pr_reviewer(gh_username, pr_config) and review_state == 'pending')


@routes.get('/me')
@auth.authenticated_developers_only()
async def get_user(request: web.Request, userdata: UserData) -> web.Response:
    for authorized_user in AUTHORIZED_USERS:
        if authorized_user.hail_username == userdata['username']:
            user = authorized_user
            break
    else:
        raise web.HTTPForbidden()

    wbs = [await watched_branch_config(request.app, wb, i) for i, wb in enumerate(watched_branches)]
    pr_wbs = filter_wbs(wbs, lambda pr: is_pr_author(user.gh_username, pr))
    review_wbs = filter_wbs(wbs, lambda pr: is_pr_reviewer(user.gh_username, pr))
    actionable_wbs = filter_wbs(wbs, lambda pr: pr_requires_action(user.gh_username, pr))

    batch_client = request.app[AppKeys.BATCH_CLIENT]
    dev_deploys = batch_client.list_batches(f'user={user.hail_username} dev_deploy=1', limit=10)
    dev_deploys = sorted([b async for b in dev_deploys], key=lambda b: b.id, reverse=True)

    team_random_member = {team: select_random_teammate(team).gh_username for team in TEAMS}

    page_context = {
        'username': user.hail_username,
        'gh_username': user.gh_username,
        'pr_wbs': pr_wbs,
        'review_wbs': review_wbs,
        'actionable_wbs': actionable_wbs,
        'team_member': team_random_member,
        'dev_deploys': [await b.last_known_status() for b in dev_deploys],
    }
    return await render_template('ci', request, userdata, 'user.html', page_context)


@routes.post('/authorize_source_sha')
@auth.authenticated_developers_only(redirect=False)
async def post_authorized_source_sha(request: web.Request, _) -> NoReturn:
    app = request.app
    db = app[AppKeys.DB]
    post = await request.post()
    sha = str(post['sha']).strip()
    await db.execute_insertone('INSERT INTO authorized_shas (sha) VALUES (%s);', sha)
    log.info(f'authorized sha: {sha}')
    session = await aiohttp_session.get_session(request)
    set_message(session, f'SHA {sha} authorized.', 'info')
    raise web.HTTPFound(deploy_config.external_url('ci', '/'))


@routes.get('/healthcheck')
async def healthcheck(_) -> web.Response:
    return web.Response(status=200)


gh_router = gh_routing.Router()


@gh_router.register('pull_request')
async def pull_request_callback(event):
    gh_pr = event.data['pull_request']
    number = gh_pr['number']
    target_branch = FQBranch.from_gh_json(gh_pr['base'])
    for wb in watched_branches:
        if (wb.prs and number in wb.prs) or (wb.branch == target_branch):
            app: web.Application = event.app
            await wb.notify_github_changed(
                app[AppKeys.DB], app[AppKeys.BATCH_CLIENT], app[AppKeys.GH_CLIENT], app[AppKeys.FROZEN_MERGE_DEPLOY]
            )


@gh_router.register('push')
async def push_callback(event):
    data = event.data
    ref = data['ref']
    if ref.startswith('refs/heads/'):
        branch_name = ref.removeprefix('refs/heads/')
        branch = FQBranch(Repo.from_gh_json(data['repository']), branch_name)
        for wb in watched_branches:
            if wb.branch == branch:
                app: web.Application = event.app
                await wb.notify_github_changed(
                    app[AppKeys.DB], app[AppKeys.BATCH_CLIENT], app[AppKeys.GH_CLIENT], app[AppKeys.FROZEN_MERGE_DEPLOY]
                )


@gh_router.register('pull_request_review')
async def pull_request_review_callback(event):
    gh_pr = event.data['pull_request']
    number = gh_pr['number']
    for wb in watched_branches:
        if number in wb.prs:
            app: web.Application = event.app
            await wb.notify_github_changed(
                app[AppKeys.DB], app[AppKeys.BATCH_CLIENT], app[AppKeys.GH_CLIENT], app[AppKeys.FROZEN_MERGE_DEPLOY]
            )


async def github_callback_handler(request: web.Request):
    event = gh_sansio.Event.from_http(request.headers, await request.read())
    event.app = request.app  # type: ignore
    await gh_router.dispatch(event)


@routes.post('/github_callback')
async def github_callback(request: web.Request):
    await asyncio.shield(github_callback_handler(request))
    return web.Response(status=200)


async def remove_namespace_from_db(db: Database, namespace: str):
    assert namespace != 'default'
    await db.just_execute(
        'DELETE FROM active_namespaces WHERE namespace = %s',
        (namespace,),
    )


async def batch_callback_handler(request: web.Request):
    app = request.app
    db = app[AppKeys.DB]
    params = await json_request(request)
    log.info(f'batch callback {params}')
    attrs = params.get('attributes')
    if attrs:
        target_branch = attrs.get('target_branch')
        if target_branch:
            for wb in watched_branches:
                if wb.branch.short_str() == target_branch:
                    log.info(f'watched_branch {wb.branch.short_str()} notify batch changed')

                    if 'test' in attrs and params['complete']:
                        assert 'deploy' not in attrs
                        assert 'dev' not in attrs
                        namespace = attrs['namespace']
                        if DEFAULT_NAMESPACE == 'default':
                            await remove_namespace_from_db(db, namespace)

                    await wb.notify_batch_changed(
                        db, app[AppKeys.BATCH_CLIENT], app[AppKeys.GH_CLIENT], app[AppKeys.FROZEN_MERGE_DEPLOY]
                    )


@routes.get('/api/v1alpha/deploy_status')
@auth.authenticated_developers_only()
async def deploy_status(request: web.Request, _) -> web.Response:
    batch_client = request.app[AppKeys.BATCH_CLIENT]

    async def get_failure_information(batch):
        if isinstance(batch, MergeFailureBatch):
            exc = batch.exception
            return traceback.format_exception(type(exc), value=exc, tb=exc.__traceback__)
        jobs = await collect_aiter(batch.jobs())

        async def fetch_job_and_log(j):
            full_job = await batch_client.get_job(j['batch_id'], j['job_id'])
            log = await full_job.log()
            status = await full_job.status()
            return {**status, 'log': log}

        return await asyncio.gather(*[fetch_job_and_log(j) for j in jobs if j['state'] in ('Error', 'Failed')])

    wb_configs = [
        {
            'branch': wb.branch.short_str(),
            'sha': wb.sha,
            'deploy_batch_id': wb.deploy_batch.id if wb.deploy_batch and isinstance(wb.deploy_batch, Batch) else None,
            'deploy_state': wb.deploy_state,
            'repo': wb.branch.repo.short_str(),
            'failure_information': None
            if wb.deploy_state == 'success'
            else await get_failure_information(wb.deploy_batch),
        }
        for wb in watched_branches
    ]
    return json_response(wb_configs)


@routes.post('/api/v1alpha/update')
@auth.authenticated_developers_only()
async def post_update(request: web.Request, _) -> web.Response:
    log.info('developer triggered update')
    db = request.app[AppKeys.DB]
    batch_client = request.app[AppKeys.BATCH_CLIENT]
    gh_client = request.app[AppKeys.GH_CLIENT]
    frozen = request.app[AppKeys.FROZEN_MERGE_DEPLOY]

    async def update_all():
        for wb in watched_branches:
            await wb.update(db, batch_client, gh_client, frozen)

    request.app[AppKeys.TASK_MANAGER].ensure_future(update_all())
    return web.Response(status=200)


@routes.post('/api/v1alpha/dev_deploy_branch')
@auth.authenticated_developers_only()
async def dev_deploy_branch(request: web.Request, userdata: UserData) -> web.Response:
    app = request.app
    try:
        params = await json_request(request)
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

    gh = app[AppKeys.GH_CLIENT]
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

    unwatched_branch = UnwatchedBranch(branch, sha, userdata, app[AppKeys.DEVELOPERS], extra_config=extra_config)

    batch_client = app[AppKeys.BATCH_CLIENT]

    try:
        batch_id = await unwatched_branch.deploy(app[AppKeys.DB], batch_client, steps, excluded_steps=excluded_steps)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # pylint: disable=broad-except
        message = traceback.format_exc()
        raise web.HTTPBadRequest(text=f'starting the deploy failed due to\n{message}') from e
    return json_response({'sha': sha, 'batch_id': batch_id})


@routes.post('/api/v1alpha/batch_callback')
async def batch_callback(request: web.Request):
    await asyncio.shield(batch_callback_handler(request))
    return web.Response(status=200)


@routes.post('/freeze_merge_deploy')
@auth.authenticated_developers_only()
async def freeze_deploys(request: web.Request, _) -> NoReturn:
    app = request.app
    db = app[AppKeys.DB]
    session = await aiohttp_session.get_session(request)

    if app[AppKeys.FROZEN_MERGE_DEPLOY]:
        set_message(session, 'CI is already frozen.', 'info')
        raise web.HTTPFound(deploy_config.external_url('ci', '/'))

    await db.execute_update("""
UPDATE globals SET frozen_merge_deploy = 1;
""")

    app[AppKeys.FROZEN_MERGE_DEPLOY] = True

    set_message(session, 'Froze all merges and deploys.', 'info')

    raise web.HTTPFound(deploy_config.external_url('ci', '/'))


@routes.post('/unfreeze_merge_deploy')
@auth.authenticated_developers_only()
async def unfreeze_deploys(request: web.Request, _) -> NoReturn:
    app = request.app
    db = app[AppKeys.DB]
    session = await aiohttp_session.get_session(request)

    if not app[AppKeys.FROZEN_MERGE_DEPLOY]:
        set_message(session, 'CI is already unfrozen.', 'info')
        raise web.HTTPFound(deploy_config.external_url('ci', '/'))

    await db.execute_update("""
UPDATE globals SET frozen_merge_deploy = 0;
""")

    app[AppKeys.FROZEN_MERGE_DEPLOY] = False

    set_message(session, 'Unfroze all merges and deploys.', 'info')

    raise web.HTTPFound(deploy_config.external_url('ci', '/'))


@routes.get('/namespaces')
@auth.authenticated_developers_only()
async def get_active_namespaces(request: web.Request, userdata: UserData) -> web.Response:
    db = request.app[AppKeys.DB]
    namespaces = [
        r
        async for r in db.execute_and_fetchall("""
SELECT active_namespaces.*, JSON_ARRAYAGG(service) as services
FROM active_namespaces
LEFT JOIN deployed_services
ON active_namespaces.namespace = deployed_services.namespace
GROUP BY active_namespaces.namespace""")
    ]
    for ns in namespaces:
        ns['services'] = [s for s in json.loads(ns['services']) if s is not None]
    context = {
        'namespaces': namespaces,
    }
    return await render_template('ci', request, userdata, 'namespaces.html', context)


@routes.post('/namespaces/{namespace}/services/add')
@auth.authenticated_developers_only()
async def add_namespaced_service(request: web.Request, _) -> NoReturn:
    db = request.app[AppKeys.DB]
    post = await request.post()
    service = post['service']
    namespace = request.match_info['namespace']

    record = await db.select_and_fetchone(
        """
SELECT 1 FROM deployed_services
WHERE namespace = %s AND service = %s
""",
        (namespace, service),
    )

    if record:
        session = await aiohttp_session.get_session(request)
        set_message(session, 'Service already registered', 'info')
    else:
        await db.execute_insertone(
            'INSERT INTO deployed_services VALUES (%s, %s)',
            (namespace, service),
        )

    raise web.HTTPFound(deploy_config.external_url('ci', '/namespaces'))


@routes.post('/namespaces/add')
@auth.authenticated_developers_only()
async def add_namespace(request: web.Request, _) -> NoReturn:
    db = request.app[AppKeys.DB]
    post = await request.post()
    namespace = post['namespace']

    record = await db.execute_and_fetchone(
        'SELECT 1 FROM active_namespaces where namespace = %s',
        (namespace,),
    )

    if record:
        session = await aiohttp_session.get_session(request)
        set_message(session, 'Namespace already registered', 'info')
    else:
        await db.execute_insertone(
            'INSERT INTO active_namespaces (`namespace`) VALUES (%s)',
            (namespace,),
        )

    raise web.HTTPFound(deploy_config.external_url('ci', '/namespaces'))


async def cleanup_expired_namespaces(db: Database):
    assert DEFAULT_NAMESPACE == 'default'
    expired_namespaces = [
        record['namespace']
        async for record in db.execute_and_fetchall(
            'SELECT namespace FROM active_namespaces WHERE expiration_time < UTC_TIMESTAMP()'
        )
    ]
    for namespace in expired_namespaces:
        assert namespace != 'default'
        log.info(f'Cleaning up expired namespace: {namespace}')
        await remove_namespace_from_db(db, namespace)


async def update_envoy_configs(db: Database, k8s_client):
    assert DEFAULT_NAMESPACE == 'default'

    api_response = await k8s_client.list_namespace()
    live_namespaces = tuple(ns.metadata.name for ns in api_response.items)
    namespace_arg_list = "(" + ",".join('%s' for _ in live_namespaces) + ")"

    services_per_namespace = {
        r['namespace']: [s for s in json.loads(r['services']) if s is not None]
        async for r in db.execute_and_fetchall(
            f"""
SELECT active_namespaces.namespace, JSON_ARRAYAGG(service) as services
FROM active_namespaces
LEFT JOIN deployed_services
ON active_namespaces.namespace = deployed_services.namespace
WHERE active_namespaces.namespace IN {namespace_arg_list}
GROUP BY active_namespaces.namespace""",
            live_namespaces,
        )
    }
    assert 'default' in services_per_namespace
    default_services = services_per_namespace.pop('default')
    assert set(['batch', 'auth', 'batch-driver', 'ci']).issubset(set(default_services)), default_services

    for proxy in ('gateway', 'internal-gateway'):
        configmap_name = f'{proxy}-xds-config'
        configmap = await k8s_client.read_namespaced_config_map(
            name=configmap_name,
            namespace=DEFAULT_NAMESPACE,
        )
        cds = create_cds_response(default_services, services_per_namespace, proxy)
        rds = create_rds_response(default_services, services_per_namespace, proxy)
        configmap.data['cds.yaml'] = yaml.dump(cds)
        configmap.data['rds.yaml'] = yaml.dump(rds)
        await k8s_client.patch_namespaced_config_map(
            name=configmap_name,
            namespace=DEFAULT_NAMESPACE,
            body=configmap,
        )


async def update_loop(app: web.Application):
    wb: Optional[WatchedBranch] = None
    while True:
        try:
            for wb in watched_branches:
                log.info(f'updating {wb.branch.short_str()}')
                await wb.update(
                    app[AppKeys.DB], app[AppKeys.BATCH_CLIENT], app[AppKeys.GH_CLIENT], app[AppKeys.FROZEN_MERGE_DEPLOY]
                )
        except concurrent.futures.CancelledError:
            raise
        except Exception:  # pylint: disable=broad-except
            if wb:
                log.exception(f'{wb.branch.short_str()} update failed due to exception')
        await asyncio.sleep(300)


class AppKeys(CommonAiohttpAppKeys):
    DB = web.AppKey('db', Database)
    GH_CLIENT = web.AppKey('github_client', gh_aiohttp.GitHubAPI)
    BATCH_CLIENT = web.AppKey('batch_client', BatchClient)
    FROZEN_MERGE_DEPLOY = web.AppKey('frozen_merge_deploy', bool)
    TASK_MANAGER = web.AppKey('task_manager', aiotools.BackgroundTaskManager)
    DEVELOPERS = web.AppKey('developers', List[UserData])
    EXIT_STACK = web.AppKey('exit_stack', AsyncExitStack)


async def on_startup(app: web.Application):
    exit_stack = AsyncExitStack()
    app[AppKeys.EXIT_STACK] = exit_stack

    client_session = httpx.client_session()
    exit_stack.push_async_callback(client_session.close)

    app[AppKeys.CLIENT_SESSION] = client_session
    app[AppKeys.GH_CLIENT] = gh_aiohttp.GitHubAPI(client_session.client_session, 'ci', oauth_token=oauth_token)
    app[AppKeys.BATCH_CLIENT] = await BatchClient.create('ci')
    exit_stack.push_async_callback(app[AppKeys.BATCH_CLIENT].close)

    app[AppKeys.DB] = Database()
    await app[AppKeys.DB].async_init()
    exit_stack.push_async_callback(app[AppKeys.DB].async_close)

    row = await app[AppKeys.DB].select_and_fetchone("""
SELECT frozen_merge_deploy FROM globals;
""")

    app[AppKeys.FROZEN_MERGE_DEPLOY] = row['frozen_merge_deploy']
    app[AppKeys.TASK_MANAGER] = aiotools.BackgroundTaskManager()
    exit_stack.callback(app[AppKeys.TASK_MANAGER].shutdown)

    if DEFAULT_NAMESPACE == 'default':
        kubernetes_asyncio.config.load_incluster_config()
        k8s_client = kubernetes_asyncio.client.CoreV1Api()
        app[AppKeys.TASK_MANAGER].ensure_future(
            periodically_call(10, update_envoy_configs, app[AppKeys.DB], k8s_client)
        )
        app[AppKeys.TASK_MANAGER].ensure_future(periodically_call(10, cleanup_expired_namespaces, app[AppKeys.DB]))

    async with hail_credentials() as creds:
        headers = await creds.auth_headers()
    users = await retry_transient_errors(
        client_session.get_read_json,
        deploy_config.url('auth', '/api/v1alpha/users'),
        headers=headers,
    )
    app[AppKeys.DEVELOPERS] = [u for u in users if u['is_developer'] == 1 and u['state'] == 'active']

    global watched_branches
    watched_branches = [
        WatchedBranch(index, FQBranch.from_short_str(bss), deployable, mergeable, app[AppKeys.DEVELOPERS])
        for (index, [bss, deployable, mergeable]) in enumerate(
            json.loads(os.environ.get('HAIL_WATCHED_BRANCHES', '[]'))
        )
    ]

    app[AppKeys.TASK_MANAGER].ensure_future(update_loop(app))


async def on_cleanup(app: web.Application):
    await app[AppKeys.EXIT_STACK].aclose()


def run():
    install_profiler_if_requested('ci')

    app = web.Application(middlewares=[check_csrf_token, monitor_endpoints_middleware])
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
        ssl_context=deploy_config.server_ssl_context(),
    )
