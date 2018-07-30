from flask import Flask, request, jsonify
from batch.client import *
import requests
from google.cloud import storage
import os
import collections
import time
import threading
import logging

log = logging.getLogger('ci')
log.setLevel(logging.INFO)
fmt = logging.Formatter('%(levelname)s:%(asctime)s:%(funcName)s:%(lineno)d: %(message)s')

fh = logging.FileHandler('ci.log')
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
log.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
log.addHandler(ch)

REPO = 'hail-is/hail/' # needs trailing slash
GITHUB_URL = 'https://api.github.com/'
CONTEXT = 'hail-ci'
PR_IMAGE = 'gcr.io/broad-ctsa/hail-pr-builder:latest'
SELF_HOSTNAME = os.environ['SELF_HOSTNAME'] # 'http://35.232.159.176:3000'
BATCH_SERVER_URL = os.environ['BATCH_SERVER_URL'] # 'http://localhost:8888'
REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 5 * 60))
GCP_PROJECT = 'broad-ctsa'
VERSION = '0-1'
GCS_BUCKET = 'hail-ci-' + VERSION

log.info(f'BATCH_SERVER_URL {BATCH_SERVER_URL}')
log.info(f'SELF_HOSTNAME {SELF_HOSTNAME}')
log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')

class NoOAuthToken(Exception):
    pass
class NoPRBuildScript(Exception):
    pass
class BadStatus(Exception):
    def __init__(self, data, status_code):
        Exception.__init__(self)
        self.data = data
        self.status_code = status_code

app = Flask(__name__)

try:
    with open('oauth-token/oauth-token', 'r') as f:
        oauth_token = f.read()
except FileNotFoundError as e:
    raise NoOAuthToken(
        "working directory must contain `oauth-token/oauth-token' "
        "containing a valid GitHub oauth token"
    ) from e

try:
    with open('pr-build-script', 'r') as f:
        PR_BUILD_SCRIPT = f.read()
except FileNotFoundError as e:
    raise NoPRBuildScript(
        "working directory must contain a file called `pr-build-script' "
        "containing a string that is passed to `/bin/sh -c'"
    ) from e

# this is a bit of a hack, but makes my development life easier
if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcloud-token/hail-ci-' + VERSION + '.key'
gcs_client = storage.Client(project=GCP_PROJECT)

###############################################################################
### Global State & Setup

class Status(object):
    def __init__(self,
                 state,
                 review_state,
                 source_sha,
                 target_sha,
                 pr_number,
                 job_id=None):
        assert state == 'failure' or state == 'pending' or state == 'success' or state == 'running'
        assert review_state == 'approved' or review_state == 'changes_requested' or review_state == 'pending', review_state
        self.state = state
        self.review_state = review_state
        self.source_sha = source_sha
        self.target_sha = target_sha
        self.pr_number = pr_number
        self.job_id = job_id

    def github_state_up_to_date(self, github_state):
        return (self.state == github_state or
                self.state == 'running' and github_state == 'pending')

    def to_json(self):
        return {
            'state': self.state,
            'review_state': self.review_state,
            'source_sha': self.source_sha,
            'target_sha': self.target_sha,
            'pr_number': self.pr_number,
            'job_id': self.job_id
        }

target_source_pr = collections.defaultdict(dict)
source_target_pr = collections.defaultdict(dict)

def update_pr_status(source_url, source_ref, target_url, target_ref, status):
    target_source_pr[(target_url, target_ref)][(source_url, source_ref)] = status
    source_target_pr[(source_url, source_ref)][(target_url, target_ref)] = status

def get_pr_status(source_url, source_ref, target_url, target_ref, default=None):
    x = source_target_pr[(source_url, source_ref)].get((target_url, target_ref), default)
    y = target_source_pr[(target_url, target_ref)].get((source_url, source_ref), default)
    assert x == y, str(x) + str(y)
    return x

def pop_prs_for_target(target_url, target_ref):
    prs = target_source_pr.pop((target_url, target_ref))
    for (source_url, source_ref), status in prs.items():
        x = source_target_pr[(source_url, source_ref)]
        del x[(target_url, target_ref)]
    return prs

def pop_prs_for_target(target_url, target_ref, default):
    prs = target_source_pr.pop((target_url, target_ref), None)
    if prs is None:
        return default
    for (source_url, source_ref), status in prs.items():
        x = source_target_pr[(source_url, source_ref)]
        del x[(target_url, target_ref)]
    return prs

def get_pr_status_by_source(source_url, source_ref):
    return source_target_pr[(source_url, source_ref)]

def get_pr_status_by_target(target_url, target_ref):
    return target_source_pr[(target_url, target_ref)]

batch_client = BatchClient(url=BATCH_SERVER_URL)

def cancel_existing_jobs(source_url, source_ref, target_url, target_ref):
    old_status = source_target_pr.get((source_url, source_ref), {}).get((target_url, target_ref), None)
    if old_status and old_status.state == 'running':
        id = old_status.job_id
        assert id is not None
        log.info(f'cancelling existing job {id} due to pr status update')
        try_to_cancel_job(batch_client.get_job(id))

@app.route('/status')
def status():
    return jsonify(
        [{ 'source_url': source_url,
           'source_ref': source_ref,
           'target_url': target_url,
           'target_ref': target_ref,
           'status': status.to_json()
        } for ((source_url, source_ref), prs) in source_target_pr.items()
         for ((target_url, target_ref), status) in prs.items()])

###############################################################################
### post and get helpers

def post_repo(url, headers=None, json=None, data=None, status_code=None):
    if headers is None:
        headers = {}
    if 'Authorization' in headers:
        raise ValueError(
            'Header already has Authorization? ' + str(headers))
    headers['Authorization'] = 'token ' + oauth_token
    r = requests.post(
        GITHUB_URL + 'repos/' + REPO + url,
        headers=headers,
        json=json,
        data=data
    )
    if status_code and r.status_code != status_code:
        raise BadStatus({
            'method': 'post',
            'endpoint' : GITHUB_URL + 'repos/' + REPO + url,
            'status_code' : r.status_code,
            'data': data,
            'json': json,
            'message': 'github error',
            'github_json': r.json()
        }, r.status_code)
    else:
        return r.json()

def get_repo(url, headers=None, status_code=None):
    return get_github('repos/' + REPO + url, headers, status_code)

def get_github(url, headers=None, status_code=None):
    if headers is None:
        headers = {}
    if 'Authorization' in headers:
        raise ValueError(
            'Header already has Authorization? ' + str(headers))
    headers['Authorization'] = 'token ' + oauth_token
    r = requests.get(
        GITHUB_URL + url,
        headers=headers
    )
    if status_code and r.status_code != status_code:
        raise BadStatus({
            'method': 'get',
            'endpoint' : GITHUB_URL + url,
            'status_code' : r.status_code,
            'message': 'github error',
            'github_json': r.json()
        }, r.status_code)
    else:
        return r.json()

###############################################################################
### Error Handlers

@app.errorhandler(BadStatus)
def handle_invalid_usage(error):
    log.error('ERROR: ' + str(error.status_code) + ': ' + str(error.data) + '\n\nrequest json: ' + str(request.json))
    return jsonify(error.data), error.status_code

###############################################################################
### Endpoints

@app.route('/push', methods=['POST'])
def github_push():
    data = request.json
    ref = data['ref']
    new_sha = data['after']
    if ref.startswith('refs/heads/'):
        target_ref = ref[11:]
        target_url = data['repository']['clone_url']
        pr_statuses = get_pr_status_by_target(target_url, target_ref)
        for (source_url, source_ref), status in pr_statuses.items():
            if (status.target_sha != new_sha):
                post_repo(
                    'statuses/' + status.source_sha,
                    json={
                        'state': 'pending',
                        'description': f'build merged into {new_sha} pending',
                        'context': CONTEXT
                    },
                    status_code=201
                )
                update_pr_status(
                    source_url,
                    source_ref,
                    target_url,
                    target_ref,
                    Status('pending', 'pending', status.source_sha, new_sha, status.pr_number))
        heal()
    else:
        log.info(f'ignoring ref push {ref} because it does not start with refs/heads/')
    return '', 200

@app.route('/pull_request', methods=['POST'])
def github_pull_request():
    data = request.json
    action = data['action']
    if action == 'opened' or action == 'synchronize':
        source_url = data['pull_request']['head']['repo']['clone_url']
        source_ref = data['pull_request']['head']['ref']
        source_sha = data['pull_request']['head']['sha']
        target_url = data['pull_request']['base']['repo']['clone_url']
        target_ref = data['pull_request']['base']['ref']
        pr_number = str(data['number'])
        target_sha = get_sha_for_target_ref(target_ref)
        review_status = review_status_net(pr_number)
        cancel_existing_jobs(source_url, source_ref, target_url, target_ref)
        status = Status('pending', review_status['state'], source_sha, target_sha, pr_number)
        update_pr_status(source_url, source_ref, target_url, target_ref, status)
        post_repo(
            'statuses/' + source_sha,
            json={
                'state': 'pending',
                'description': f'build merged into {target_sha} pending',
                'context': CONTEXT
            },
            status_code=201
        )
        # eagerly trigger a build (even if master has other pending PRs
        # building) since the requester introduced new changes
        test_pr(source_url, source_ref, target_url, target_ref, status)
    else:
        log.info(f'ignoring github pull_request event of type {action} full json: {data}')
    return '', 200

@app.route('/ci_build_done', methods=['POST'])
def ci_build_done():
    data = request.json
    job_id = data['id']
    exit_code = data['exit_code']
    attributes = data['attributes']
    source_url = attributes['source_url']
    source_ref = attributes['source_ref']
    target_url = attributes['target_url']
    target_ref = attributes['target_ref']
    status = get_pr_status(source_url, source_ref, target_url, target_ref)
    source_sha = attributes['source_sha']
    target_sha = attributes['target_sha']
    if status is None:
        log.info(f'ignoring job for pr I did not think existed: {target_ref}:{target_sha} <- {source_ref}:{source_sha}')
    elif status.source_sha == source_sha and status.target_sha == target_sha:
        build_finished(attributes['pr_number'],
                       source_url,
                       source_ref,
                       source_sha,
                       target_url,
                       target_ref,
                       target_sha,
                       job_id,
                       data['log'],
                       exit_code,
                       status)
        heal()
    else:
        log.info(f'ignoring completed job that I no longer care about: {target_ref}:{target_sha} <- {source_ref}:{source_sha}')

    return '', 200

def build_finished(pr_number,
                   source_url,
                   source_ref,
                   source_sha,
                   target_url,
                   target_ref,
                   target_sha,
                   job_id,
                   job_log,
                   exit_code,
                   status):
    upload_public_gs_file_from_string(
        GCS_BUCKET,
        f'{source_sha}/{target_sha}/job-log',
        job_log
    )
    upload_public_gs_file_from_filename(
        GCS_BUCKET,
        f'{source_sha}/{target_sha}/index.html',
        'index.html'
    )
    if exit_code == 0:
        log.info(f'test job {job_id} finished successfully for pr #{pr_number}')
        update_pr_status(
            source_url,
            source_ref,
            target_url,
            target_ref,
            Status('success', status.review_state, status.source_sha, status.target_sha, pr_number, job_id)
        )
        post_repo(
            'statuses/' + source_sha,
            json={
                'state': 'success',
                'description': 'successful build after merge with ' + target_sha,
                'context': CONTEXT,
                'target_url': f'https://storage.googleapis.com/{GCS_BUCKET}/{source_sha}/{target_sha}/index.html'
            },
            status_code=201
        )
    else:
        log.info(f'test job {job_id} failed for pr #{pr_number} ({source_sha}) with exit code {exit_code} after merge with {target_sha}')
        update_pr_status(
            source_url,
            source_ref,
            target_url,
            target_ref,
            Status('failure', status.review_state, status.source_sha, status.target_sha, pr_number, job_id)
        )
        status_message = f'failing build ({exit_code}) after merge with {target_sha}'
        post_repo(
            'statuses/' + source_sha,
            json={
                'state': 'failure',
                'description': status_message,
                'context': CONTEXT,
                'target_url': f'https://storage.googleapis.com/{GCS_BUCKET}/{source_sha}/{target_sha}/index.html'
            },
            status_code=201
        )

@app.route('/heal', methods=['POST'])
def heal_endpoint():
    heal()
    return '', 200

def heal():
    for (target_url, target_ref), prs in target_source_pr.items():
        ready_to_merge = [(source, status)
                          for source, status in prs.items()
                          if status.state == 'success'
                          and status.review_state == 'approved']
        if len(ready_to_merge) != 0:
            # pick oldest one instead
            ((source_url, source_ref), status) = ready_to_merge[0]
            log.info(f'normally I would merge {source_url}:{source_ref} into {target_url}:{target_ref} with status {status.to_json()}')
        # else:
        approved_running = [(source, status)
                            for source, status in prs.items()
                            if status.state == 'running'
                            and status.review_state == 'approved']
        if len(approved_running) != 0:
            approved_running_json = [(source, status.to_json()) for (source, status) in approved_running]
            log.info(f'at least one approved PR is already being tested against {target_url}:{target_ref}, I will not test any others. {approved_running_json}')
        else:
            approved = [(source, status)
                        for source, status in prs.items()
                        if status.state == 'pending'
                        and status.review_state == 'approved']
            if len(approved) != 0:
                # pick oldest one instead
                ((source_url, source_ref), status) = approved[0]
                log.info(f'no approved and running prs, will build: {target_url}:{target_ref} <- {source_url}:{source_ref}; {status.target_sha} <- {status.source_sha}')
                test_pr(source_url, source_ref, target_url, target_ref, status)
            else:
                untested = [(source, status)
                            for source, status in prs.items()
                            if status.state == 'pending']
                if len(untested) != 0:
                    log.info('no approved prs, will build all PRs with out of date statuses')
                    for (source_url, source_ref), status in untested:
                        log.info(f'building: {target_url}:{target_ref} <- {source_url}:{source_ref}; {status.target_sha} <- {status.source_sha}')
                        test_pr(source_url, source_ref, target_url, target_ref, status)
                else:
                    log.info('all prs are tested or running')

@app.route('/force_retest_pr', methods=['POST'])
def force_retest_pr_endpoint():
    data = request.json
    user = data['user']
    source_ref = data['source_ref']
    target_repo = data['target_repo']
    target_ref = data['target_ref']

    pulls = get_github(
        f'repos/{target_repo}/pulls?state=open&head={user}:{source_ref}&base={target_ref}',
        status_code=200
    )

    if len(pulls) != 1:
        return f'too many PRs found: {pulls}', 200

    pull = pulls[0]
    pr_number = str(pull['number'])
    source_url = pull['head']['repo']['clone_url']
    source_ref = pull['head']['ref']
    source_sha = pull['head']['sha']
    target_url = pull['base']['repo']['clone_url']
    target_ref = pull['base']['ref']
    target_sha = get_github(
        f'repos/{target_repo}/git/refs/heads/{target_ref}',
        status_code=200
    )['object']['sha']

    review_status = review_status_net(pr_number)
    cancel_existing_jobs(source_url, source_ref, target_url, target_ref)
    status = Status('pending', review_status['state'], source_sha, target_sha, pr_number)
    update_pr_status(source_url, source_ref, target_url, target_ref, status)
    post_repo(
        'statuses/' + source_sha,
        json={
            'state': 'pending',
            'description': f'(MANUALLY FORCED REBUILD) build merged into {target_sha} pending',
            'context': CONTEXT
        },
        status_code=201
    )
    test_pr(source_url, source_ref, target_url, target_ref, status)
    return '', 200

def test_pr(source_url, source_ref, target_url, target_ref, status):
    assert status.state == 'pending', str(status.to_json())
    attributes = {
        'pr_number': status.pr_number,
        'source_url': source_url,
        'source_ref': source_ref,
        'source_sha': status.source_sha,
        'target_url': target_url,
        'target_ref': target_ref,
        'target_sha': status.target_sha,
        'type': 'hail-ci-' + VERSION
    }
    log.info(f'creating job with attributes {attributes}')
    job=batch_client.create_job(
        PR_IMAGE,
        command=[
            '/bin/bash',
            '-c',
            PR_BUILD_SCRIPT],
        env={
            'SOURCE_REPO_URL': source_url,
            'SOURCE_BRANCH': source_ref,
            'SOURCE_SHA': status.source_sha,
            'TARGET_REPO_URL': target_url,
            'TARGET_BRANCH': target_ref,
            'TARGET_SHA': status.target_sha
        },
        resources={
            'requests': {
                'cpu' : '3.7',
                'memory': '4G'
            }
        },
        tolerations=[{
            'key': 'preemptible',
            'value': 'true'
        }],
        callback=SELF_HOSTNAME + '/ci_build_done',
        attributes=attributes,
        volumes=[{
            'volume': { 'name' : f'hail-ci-{VERSION}-service-account-key',
                        'secret' : { 'optional': False,
                                     'secretName': f'hail-ci-{VERSION}-service-account-key' } },
            'volume_mount': { 'mountPath': '/secrets',
                              'name': f'hail-ci-{VERSION}-service-account-key',
                              'readOnly': True }
        }]
    )
    log.info(f'successfully created job {job.id} with attributes {attributes}')
    post_repo(
        'statuses/' + status.source_sha,
        json={
            'state': 'pending',
            'description': f'build merged into {status.target_sha} running {job.id}',
            'context': CONTEXT
        },
        status_code=201
    )
    update_pr_status(
        source_url,
        source_ref,
        target_url,
        target_ref,
        Status('running',
               status.review_state,
               status.source_sha,
               status.target_sha,
               status.pr_number,
               job_id=job.id))

def get_sha_for_target_ref(ref):
    return get_repo(
        'git/refs/heads/' + ref,
        status_code=200
    )['object']['sha']

@app.route('/refresh_github_state', methods=['POST'])
def refresh_github_state():
    target_url = 'https://github.com/'+REPO[:-1]+'.git'
    pulls = get_repo(
        'pulls?state=open',
        status_code=200
    )
    pulls_by_target = collections.defaultdict(list)
    for pull in pulls:
        target_ref = pull['base']['ref']
        pulls_by_target[target_ref].append(pull)
    for target_ref, pulls in pulls_by_target.items():
        target_sha = get_sha_for_target_ref(target_ref)
        log.info(f'for target {target_ref} ({target_sha}) we found ' + str([pull['title'] for pull in pulls]))
        known_prs = pop_prs_for_target(target_url, target_ref, {})
        for pull in pulls:
            source_url = pull['head']['repo']['clone_url']
            source_ref = pull['head']['ref']
            source_sha = pull['head']['sha']
            pr_number = str(pull['number'])
            status = known_prs.pop((source_url, source_ref), None)
            review_status = review_status_net(pr_number)
            latest_state = status_state_net(source_sha, target_sha)
            latest_review_state = review_status['state']
            if (status and
                status.source_sha == source_sha and
                status.target_sha == target_sha and
                status.github_state_up_to_date(latest_state) and
                status.review_state == latest_review_state):
                log.info(f'no change to knowledge of {target_url}:{target_ref} <- {source_url}:{source_ref}')
                # restore pop'ed status
                update_pr_status(source_url, source_ref, target_url, target_ref, status)
            else:
                log.info(f'updating knowledge of {target_url}:{target_ref} <- {source_url}:{source_ref} '
                         f'to {latest_state} {latest_review_state} {target_sha} <- {source_sha}')
                update_pr_status(
                    source_url,
                    source_ref,
                    target_url,
                    target_ref,
                    Status(latest_state, latest_review_state, source_sha, target_sha, pr_number))

        if len(known_prs) != 0:
            known_prs_json = [(x, status.to_json()) for (x, status) in known_prs.items()]
            log.info(f'some PRs have been invalidated by github state refresh: {known_prs_json}')
            for (source_url, source_ref), status in known_prs.items():
                if status.state == 'running':
                    log.info(f'cancelling job {status.job_id} for {status.to_json()}')
                    try_to_cancel_job(client.get_job(status.job_id))

    return '', 200

# FIXME: have an end point to refresh batch/jobs state, needs a jobs endpoint
@app.route('/refresh_batch_state', methods=['POST'])
def refresh_batch_state():
    jobs = batch_client.list_jobs()
    latest_jobs = {}
    for job in jobs:
        t = job.attributes.get('type', None)
        if t and t == 'hail-ci-' + VERSION:
            key = (job.attributes['source_url'],
                   job.attributes['source_ref'],
                   job.attributes['source_sha'],
                   job.attributes['target_url'],
                   job.attributes['target_ref'],
                   job.attributes['target_sha'])
            job2 = latest_jobs.get(key, None)
            if job2 is None:
                latest_jobs[key] = job
            else:
                job2_state = job2.status()['state']
                job_state = job.status()['state']
                if (batch_job_state_smaller_is_closer_to_complete(job2_state, job_state) < 0 or
                    job2.id < job.id):
                    try_to_cancel_job(job2)
                    latest_jobs[key] = job
                else:
                    try_to_cancel_job(job)

    for job in latest_jobs.values():
        job_id = job.id
        job_status = job.status()
        job_state = job_status['state']
        pr_number = job.attributes['pr_number']
        source_url = job.attributes['source_url']
        source_ref = job.attributes['source_ref']
        source_sha = job.attributes['source_sha']
        target_url = job.attributes['target_url']
        target_ref = job.attributes['target_ref']
        target_sha = job.attributes['target_sha']
        status = get_pr_status(source_url, source_ref, target_url, target_ref)
        if status and status.source_sha == source_sha and status.target_sha == target_sha:
            assert status.pr_number == pr_number, f'{status.pr_number} {pr_number}'
            if job_state == 'Complete':
                exit_code = job_status['exit_code']
                job_log = job_status['log']
                if exit_code == 0:
                    build_state = 'success'
                else:
                    build_state = 'failure'
                if status.state == 'pending' or status.state == 'running' or status.state != build_state:
                    log.info(f'updating knowledge of {target_url}:{target_ref} <- {source_url}:{source_ref} '
                             f'to {build_state} {status.review_state} {target_sha} <- {source_sha}')
                    build_finished(pr_number,
                                   source_url,
                                   source_ref,
                                   source_sha,
                                   target_url,
                                   target_ref,
                                   target_sha,
                                   job_id,
                                   job_log,
                                   exit_code,
                                   status)
                else:
                    log.info(f'already knew {target_url}:{target_ref} <- {source_url}:{source_ref} '
                             f'was {build_state} for {target_sha} <- {source_sha}')
            elif job_state == 'Cancelled':
                if status.state != 'pending':
                    log.info(f'updating knowledge of {target_url}:{target_ref} <- {source_url}:{source_ref} '
                             f'to pending {status.review_state} {target_sha} <- {source_sha}')
                    post_repo(
                        'statuses/' + source_sha,
                        json={
                            'state': 'pending',
                            'description': f'build merged into {target_sha} pending (job {job_id} was cancelled)',
                            'context': CONTEXT
                        },
                        status_code=201
                    )
                    update_pr_status(
                        source_url, source_ref, target_url, target_ref,
                        Status('pending',
                               status.review_state,
                               status.source_sha,
                               status.target_sha,
                               status.pr_number,
                               job_id))
                else:
                    log.info(f'already knew {target_url}:{target_ref} <- {source_url}:{source_ref} '
                             f'was pending for {target_sha} <- {source_sha}')
            else:
                if status.state != 'running' or status.job_id != job_id:
                    log.info(f'updating knowledge of {target_url}:{target_ref} <- {source_url}:{source_ref} '
                             f'to running {status.review_state} {target_sha} <- {source_sha}')
                    assert job_state == 'Created', f'{job_state}'
                    post_repo(
                        'statuses/' + status.source_sha,
                        json={
                            'state': 'pending',
                            'description': f'build merged into {status.target_sha} running {job_id}',
                            'context': CONTEXT
                        },
                        status_code=201
                    )
                    update_pr_status(
                        source_url, source_ref, target_url, target_ref,
                        Status('running',
                               status.review_state,
                               status.source_sha,
                               status.target_sha,
                               status.pr_number,
                               job_id))
                else:
                    log.info(f'already knew {target_url}:{target_ref} <- {source_url}:{source_ref} '
                             f'was running for {target_sha} <- {source_sha}')

        else:
            if status is None:
                log.info(f'batch has job {job_id} for unknown PR, {target_url}:{target_ref} <- {source_url}:{source_ref} ')
            else:
                log.info(f'batch has job {job_id} with unexpected SHAs for {target_url}:{target_ref} <- {source_url}:{source_ref} '
                         f'job SHAs: {target_sha} <- {source_sha}, my SHAs: {status.target_sha} <- {status.source_sha}')
            if job_state == 'Created':
                log.info(f'will cancel undesired batch job {job_id}')
                try_to_cancel_job(job)
    return '', 200

###############################################################################
### Batch Utils

def try_to_cancel_job(job):
    try:
        job.cancel()
    except requests.exceptions.HTTPError as e:
        log.warn(f'could not cancel job {job.id} due to {e}')

def batch_job_state_smaller_is_closer_to_complete(x, y):
    if x == 'Complete':
        if y == 'Complete':
            return 0
        else:
            return -1
    elif x == 'Cancelled':
        if y == 'Cancelled':
            return 0
        else:
            assert y == 'Created' or y == 'Complete', y
            return 1
    else:
        assert x == 'Created', x
        if y == 'Created':
            return 0
        elif y == 'Complete':
            return 1
        else:
            assert y == 'Cancelled', y
            return -1

###############################################################################
### Review Status

@app.route('/pr/<pr_number>/review_status')
def review_status_endpoint(pr_number):
    status = review_status_net(pr_number)
    return jsonify(status), 200

def review_status_net(pr_number):
    reviews = get_repo(
        'pulls/' + pr_number + '/reviews',
        status_code=200
    )
    return review_status(reviews)

def status_state_net(source_sha, target_sha):
    statuses = get_repo(
        'commits/' + source_sha + '/statuses',
        status_code=200
    )
    my_statuses = [s for s in statuses if s['context'] == CONTEXT]
    latest_state = 'pending'
    if len(my_statuses) != 0:
        latest_status = my_statuses[0]
        if target_sha in latest_status['description']:
            latest_state = latest_status['state']
    return latest_state

def review_status(reviews):
    latest_state_by_login = {}
    for review in reviews:
        login = review['user']['login']
        state = review['state']
        # reviews is chronological, so later ones are newer statuses
        latest_state_by_login[login] = state
    at_least_one_approved = False
    for login, state in latest_state_by_login.items():
        if (state == 'CHANGES_REQUESTED'):
            return {
                'state': 'changes_requested',
                'reviews': latest_state_by_login
            }
        elif (state == 'APPROVED'):
            at_least_one_approved = True

    if at_least_one_approved:
        return {
            'state': 'approved',
            'reviews': latest_state_by_login
        }
    else:
        return {
            'state': 'pending',
            'reviews': latest_state_by_login
        }

###############################################################################
### SHA Statuses

@app.route('/pr/<sha>/statuses')
def statuses(sha):
    json = get_repo(
        'commits/' + sha + '/statuses',
        status_code=200
    )
    return jsonify(json), 200

###############################################################################
### GitHub things

@app.route('/github_rate_limits')
def github_rate_limits():
    return jsonify(get_github('/rate_limit')), 200

###############################################################################
### Google Storage

def upload_public_gs_file_from_string(bucket, target_path, string):
    create_public_gs_file(
        bucket,
        target_path,
        lambda f: f.upload_from_string(string)
    )

def upload_public_gs_file_from_filename(bucket, target_path, filename):
    create_public_gs_file(
        bucket,
        target_path,
        lambda f: f.upload_from_filename(filename)
    )

def create_public_gs_file(bucket, target_path, upload):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(target_path)
    f.metadata = {'Cache-Control': 'private, max-age=0, no-transform'}
    upload(f)
    f.acl.all().grant_read()
    f.acl.save()

###############################################################################
### event loops

def flask_event_loop():
    app.run(threaded=False, host='0.0.0.0')

def polling_event_loop():
    time.sleep(5)
    while True:
        try:
           r = requests.post('http://127.0.0.1:5000/refresh_github_state')
           r.raise_for_status()
           r = requests.post('http://127.0.0.1:5000/refresh_batch_state')
           r.raise_for_status()
           r = requests.post('http://127.0.0.1:5000/heal')
           r.raise_for_status()
        except Exception as e:
            log.error(f'Could not poll due to exception: {e}')
            pass
        time.sleep(REFRESH_INTERVAL_IN_SECONDS)

if __name__ == '__main__':
    poll_thread = threading.Thread(target=polling_event_loop)
    poll_thread.start()
    flask_event_loop()
