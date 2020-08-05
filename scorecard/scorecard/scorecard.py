import collections
import datetime
import os
import asyncio
import aiohttp
from aiohttp import web
import gidgethub.aiohttp
import random
import humanize
import logging
from hailtop.config import get_deploy_config
from hailtop.tls import get_in_cluster_server_ssl_context
from hailtop.hail_logging import AccessLogger
from gear import setup_aiohttp_session, web_maybe_authenticated_user
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, render_template

log = logging.getLogger('scorecard')

deploy_config = get_deploy_config()

team_members = {
    'Services Team': ['jigold', 'danking', 'catoverdrive', 'Dania-Abuhijleh'],
    'Compilers Team': ['tpoterba', 'catoverdrive', 'patrick-schultz', 'chrisvittal', 'johnc1231'],
}

default_repo = 'hail'
repos = {
    'hail': 'hail-is/hail',
}

routes = web.RouteTableDef()

data = None
timestamp = None


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=unused-argument
    return web.Response()


@routes.get('')
@routes.get('/')
@web_maybe_authenticated_user
async def index(request, userdata):
    user_data, unassigned, urgent_issues, updated = get_users()
    team_random_member = {c: random.choice(us) for c, us in team_members.items()}
    page_context = {
        'unassigned': unassigned,
        'user_data': user_data,
        'urgent_issues': urgent_issues,
        'team_member': team_random_member,
        'updated': updated
    }
    return await render_template('scorecard', request, userdata, 'index.html', page_context)


@routes.get('/users/{user}')
@web_maybe_authenticated_user
async def html_get_user(request, userdata):
    user = request.match_info['user']
    user_data, updated = get_user(user)
    page_context = {
        'user': user,
        'user_data': user_data,
        'updated': updated,
    }
    return await render_template('scorecard', request, userdata, 'user.html', page_context)


def get_users():
    cur_data = data
    cur_timestamp = timestamp

    unassigned = []
    user_data = collections.defaultdict(
        lambda: {'CHANGES_REQUESTED': [],
                 'NEEDS_REVIEW': [],
                 'ISSUES': []})

    urgent_issues = []

    def add_pr(pr):
        state = pr['state']

        if state == 'CHANGES_REQUESTED':
            d = user_data[pr['user']]
            d[state].append(pr)
        elif state == 'NEEDS_REVIEW':
            for user in pr['assignees']:
                d = user_data[user]
                d[state].append(pr)
        else:
            assert state == 'APPROVED'

    def add_issue(issue):
        for user in issue['assignees']:
            d = user_data[user]
            if issue['urgent']:
                time = datetime.datetime.now() - issue['created_at']
                urgent_issues.append({
                    'USER': user,
                    'ISSUE': issue,
                    'timedelta': time,
                    'AGE': humanize.naturaltime(time)})
            else:
                d['ISSUES'].append(issue)

    for _, repo_data in cur_data.items():
        for pr in repo_data['prs']:
            if len(pr['assignees']) == 0:
                unassigned.append(pr)
                continue

            add_pr(pr)

        for issue in repo_data['issues']:
            add_issue(issue)

    list.sort(urgent_issues, key=lambda issue: issue['timedelta'], reverse=True)

    updated = humanize.naturaltime(
        datetime.datetime.now() - cur_timestamp)

    return (user_data, unassigned, urgent_issues, updated)


def get_user(user):
    global data, timestamp

    cur_data = data
    cur_timestamp = timestamp

    user_data = {
        'CHANGES_REQUESTED': [],
        'NEEDS_REVIEW': [],
        'FAILING': [],
        'ISSUES': []
    }

    for _, repo_data in cur_data.items():
        for pr in repo_data['prs']:
            state = pr['state']
            if state == 'CHANGES_REQUESTED':
                if user == pr['user']:
                    user_data[state].append(pr)
            elif state == 'NEEDS_REVIEW':
                if user in pr['assignees']:
                    user_data[state].append(pr)
            else:
                assert state == 'APPROVED'

            if pr['status'] == 'failure' and user == pr['user']:
                user_data['FAILING'].append(pr)

        for issue in repo_data['issues']:
            if user in issue['assignees']:
                user_data['ISSUES'].append(issue)

    updated = humanize.naturaltime(
        datetime.datetime.now() - cur_timestamp)
    return (user_data, updated)


def get_id(repo_name, number):
    if repo_name == default_repo:
        return f'{number}'
    return f'{repo_name}/{number}'


async def get_pr_data(gh_client, fq_repo, repo_name, pr):
    assignees = [a['login'] for a in pr['assignees']]

    reviews = []
    async for review in gh_client.getiter(f'/repos/{fq_repo}/pulls/{pr["number"]}/reviews'):
        reviews.append(review)

    state = 'NEEDS_REVIEW'
    for review in reversed(reviews):
        review_state = review['state']
        if review_state == 'CHANGES_REQUESTED':  # pylint: disable=no-else-break
            state = review_state
            break
        elif review_state == 'DISMISSED':
            break
        elif review_state == 'APPROVED':
            state = 'APPROVED'
            break
        else:
            if review_state != 'COMMENTED':
                log.warning(f'unknown review state {review_state} on review {review} in pr {pr}')

    sha = pr['head']['sha']
    status = await gh_client.getitem(f'/repos/{fq_repo}/commits/{sha}')

    return {
        'repo': repo_name,
        'id': get_id(repo_name, pr['number']),
        'title': pr['title'],
        'user': pr['user']['login'],
        'assignees': assignees,
        'html_url': pr['html_url'],
        'state': state,
        'status': status
    }


def get_issue_data(repo_name, issue):
    assignees = [a['login'] for a in issue['assignees']]
    return {
        'repo': repo_name,
        'id': get_id(repo_name, issue['number']),
        'title': issue['title'],
        'assignees': assignees,
        'html_url': issue['html_url'],
        'urgent': any(label['name'] == 'prio:high' for label in issue['labels']),
        'created_at': datetime.datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ')
    }


async def update_data(gh_client):
    global data, timestamp

    rate_limit = await gh_client.getitem("/rate_limit")
    log.info(f'rate_limit {rate_limit}')
    log.info('start updating_data')

    new_data = {}

    for repo_name in repos:
        new_data[repo_name] = {
            'prs': [],
            'issues': []
        }

    try:
        for repo_name, fq_repo in repos.items():
            async for pr in gh_client.getiter(f'/repos/{fq_repo}/pulls?state=open'):
                pr_data = await get_pr_data(gh_client, fq_repo, repo_name, pr)
                new_data[repo_name]['prs'].append(pr_data)

            async for issue in gh_client.getiter(f'/repos/{fq_repo}/issues?state=open'):
                print(issue)
                if 'pull_request' not in issue:
                    issue_data = get_issue_data(repo_name, issue)
                    new_data[repo_name]['issues'].append(issue_data)
    except Exception:  # pylint: disable=broad-except
        log.exception('update failed due to except')
        return

    log.info('updating_data done')

    now = datetime.datetime.now()

    data = new_data
    timestamp = now


async def poll(gh_client):
    while True:
        await asyncio.sleep(180)
        try:
            log.info('run update_data')
            await update_data(gh_client)
            log.info('update_data returned')
        except Exception:  # pylint: disable=broad-except
            log.exception('update_data failed with exception')


async def on_startup(app):
    token_file = os.environ.get('GITHUB_TOKEN_PATH',
                                '/secrets/scorecard-github-access-token.txt')
    with open(token_file, 'r') as f:
        token = f.read().strip()
    session = aiohttp.ClientSession(
        raise_for_status=True,
        timeout=aiohttp.ClientTimeout(total=60))
    gh_client = gidgethub.aiohttp.GitHubAPI(session, 'scorecard', oauth_token=token)
    app['gh_client'] = gh_client

    await update_data(gh_client)
    asyncio.ensure_future(poll(gh_client))


def run():
    app = web.Application()
    app.on_startup.append(on_startup)

    setup_aiohttp_jinja2(app, 'scorecard')
    setup_aiohttp_session(app)

    setup_common_static_routes(routes)

    app.add_routes(routes)

    web.run_app(deploy_config.prefix_application(app, 'scorecard'),
                host='0.0.0.0',
                port=5000,
                access_log_class=AccessLogger,
                ssl_context=get_in_cluster_server_ssl_context())
