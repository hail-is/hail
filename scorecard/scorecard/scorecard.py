import time
import collections
import datetime
import os
import sys
from flask import Flask, render_template, request, abort, url_for
from github import Github
from github.GithubException import RateLimitExceededException
import random
import threading
import humanize
import logging

fmt = logging.Formatter(
    # NB: no space after levename because WARNING is so long
    '%(levelname)s\t| %(asctime)s \t| %(filename)s \t| %(funcName)s:%(lineno)d | '
    '%(message)s')

fh = logging.FileHandler('scorecard.log')
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)

log = logging.getLogger('scorecard')
log.setLevel(logging.INFO)

logging.basicConfig(
    handlers=[fh, ch],
    level=logging.INFO)

GITHUB_TOKEN_PATH = os.environ.get('GITHUB_TOKEN_PATH',
                                   '/secrets/scorecard-github-access-token.txt')
with open(GITHUB_TOKEN_PATH, 'r') as f:
    token = f.read().strip()
github = Github(token)

component_users = {
    'Hail front-end (Py)': ['tpoterba', 'jigold', 'catoverdrive', 'patrick-schultz', 'chrisvittal', 'konradjk', 'johnc1231', 'iitalics'],
    'Hail middle-end (Scala)': ['danking', 'tpoterba', 'jigold', 'catoverdrive', 'patrick-schultz', 'chrisvittal', 'johnc1231', 'iitalics'],
    'C++ backend': ['catoverdrive', 'patrick-schultz', 'chrisvittal', 'akotlar', 'iitalics'],
    'hailctl dataproc': ['tpoterba', 'danking', 'konradjk'],
    'k8s, services': ['danking', 'jigold', 'akotlar', 'johnc1231'],
    'Web app (JS)': ['akotlar', 'danking'],
}

default_repo = 'hail'
repos = {
    'hail': 'hail-is/hail',
}

app = Flask('scorecard')

data = None
timsetamp = None

@app.route('/')
def index():
    user_data, unassigned, urgent_issues, updated = get_users()
    component_random_user = {c: random.choice(us) for c, us in component_users.items()}
    return render_template('index.html', unassigned=unassigned,
                           user_data=user_data, urgent_issues=urgent_issues, component_user=component_random_user, updated=updated)

@app.route('/users/<user>')
def html_get_user(user):
    user_data, updated = get_user(user)
    return render_template('user.html', user=user, user_data=user_data, updated=updated)

def get_users():
    cur_data = data
    cur_timestamp = timestamp

    unassigned = []
    user_data = collections.defaultdict(
        lambda: {'CHANGES_REQUESTED': [],
                 'NEEDS_REVIEW': [],
                 'ISSUES': []})

    urgent_issues = []

    def add_pr(repo_name, pr):
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

    def add_issue(repo_name, issue):
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

    for repo_name, repo_data in cur_data.items():
        for pr in repo_data['prs']:
            if len(pr['assignees']) == 0:
                unassigned.append(pr)
                continue

            add_pr(repo_name, pr)

        for issue in repo_data['issues']:
            add_issue(repo_name, issue)

    list.sort(urgent_issues, key=lambda issue: issue['timedelta'], reverse=True)

    updated = humanize.naturaltime(
        datetime.datetime.now() - datetime.timedelta(seconds = time.time() - cur_timestamp))

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

    for repo_name, repo_data in cur_data.items():
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
        datetime.datetime.now() - datetime.timedelta(seconds=time.time() - cur_timestamp))
    return (user_data, updated)


def get_id(repo_name, number):
    if repo_name == default_repo:
        return f'{number}'
    else:
        return f'{repo_name}/{number}'

def get_pr_data(repo, repo_name, pr):
    assignees = [a.login for a in pr.assignees]

    state = 'NEEDS_REVIEW'
    for review in pr.get_reviews().reversed:
        if review.state == 'CHANGES_REQUESTED':
            state = review.state
            break
        elif review.state == 'DISMISSED':
            break
        elif review.state == 'APPROVED':
            state = 'APPROVED'
            break
        else:
            if review.state != 'COMMENTED':
                log.warning(f'unknown review state {review.state} on review {review} in pr {pr}')

    sha = pr.head.sha
    status = repo.get_commit(sha=sha).get_combined_status().state

    return {
        'repo': repo_name,
        'id': get_id(repo_name, pr.number),
        'title': pr.title,
        'user': pr.user.login,
        'assignees': assignees,
        'html_url': pr.html_url,
        'state': state,
        'status': status
    }

def get_issue_data(repo_name, issue):
    assignees = [a.login for a in issue.assignees]
    return {
        'repo': repo_name,
        'id': get_id(repo_name, issue.number),
        'title': issue.title,
        'assignees': assignees,
        'html_url': issue.html_url,
        'urgent': any(label.name == 'prio:high' for label in issue.labels),
        'created_at': issue.created_at
    }

def update_data():
    global data, timestamp

    log.info(f'rate_limit {github.get_rate_limit()}')
    log.info('start updating_data')

    new_data = {}

    for repo_name in repos:
        new_data[repo_name] = {
            'prs': [],
            'issues': []
        }

    try:
        for repo_name, fq_repo in repos.items():
            repo = github.get_repo(fq_repo)

            for pr in repo.get_pulls(state='open'):
                pr_data = get_pr_data(repo, repo_name, pr)
                new_data[repo_name]['prs'].append(pr_data)

            for issue in repo.get_issues(state='open'):
                if issue.pull_request is None:
                    issue_data = get_issue_data(repo_name, issue)
                    new_data[repo_name]['issues'].append(issue_data)
    except RateLimitExceededException as e:
        log.error('Exceeded rate limit: ' + str(e))


    log.info('updating_data done')

    now = time.time()

    data = new_data
    timestamp = now

def poll():
    while True:
        time.sleep(180)
        update_data()

update_data()

def run_forever(target, *args, **kwargs):
    # target should be a function
    target_name = target.__name__

    expected_retry_interval_ms = 15 * 1000 # 15s
    while True:
        start = time.time()
        try:
            log.info(f'run target {target_name}')
            target(*args, **kwargs)
            log.info(f'target {target_name} returned')
        except:
            log.error(f'target {target_name} threw exception', exc_info=sys.exc_info())
        end = time.time()

        run_time_ms = int((end - start) * 1000 + 0.5)
        t = random.randrange(expected_retry_interval_ms * 2) - run_time_ms
        if t > 0:
            log.debug(f'{target_name}: sleep {t}ms')
            time.sleep(t / 1000.0)

poll_thread = threading.Thread(target=run_forever, args=(poll,), daemon=True)
poll_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0')
