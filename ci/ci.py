from flask import Flask, request, jsonify
from batch.client import *
import requests
from google.cloud import storage
import os
import collections

# REPO = 'danking/docker-build-test/' # needs trailing slash
REPO = 'hail-is/hail/' # needs trailing slash
REPO_API_URL = 'https://api.github.com/repos/' + REPO
CONTEXT = 'hail-ci'
PR_IMAGE = 'gcr.io/broad-ctsa/hail-pr-builder:latest'
SELF_HOSTNAME = 'http://35.232.159.176:3000'
BATCH_SERVER_URL='http://localhost:8888'
GCP_PROJECT='broad-ctsa'
GCS_BUCKET='hail-ci-0-1'

class NoOAuthToken(Exception):
    pass
class NoSecret(Exception):
    pass
class NoPRBuildScript(Exception):
    pass
class BadStatus(Exception):
    def __init__(self, data, status_code):
        Exception.__init__(self)
        self.data = data
        self.status_code = status_code
class PR(object):
    def __init__(self, job, attributes):
        self.job = job
        self.attributes = attributes

    def to_json(self):
        return {
            'job': {
                'id': self.job.id,
                'client': { 'url' : self.job.client.url }
            },
            'attributes': self.attributes
        }

app = Flask(__name__)

try:
    with open('oauth-token', 'r') as f:
        oauth_token = f.read()
except FileNotFoundError as e:
    raise NoOAuthToken(
        "working directory must contain a file called `oauth-token' "
        "containing a valid GitHub oauth token"
    ) from e

try:
    with open('secret', 'r') as f:
        secret = f.read()
except FileNotFoundError as e:
    raise NoSecret(
        "working directory must contain a file called `secret' "
        "containing a string used to access dangerous endpoints"
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
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'secrets/hail-ci-0-1.key'
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

def get_pr_status(source_url, source_ref, target_url, target_ref):
    x = source_target_pr[(source_url, source_ref)].get((target_url, target_ref), None)
    y = target_source_pr[(target_url, target_ref)].get((source_url, source_ref), None)
    assert x == y, str(x) + str(y)
    return x

def pop_prs_for_target(target_url, target_ref):
    prs = target_source_pr.pop((target_url, target_ref))
    for (source_url, source_ref), status in prs:
        x = source_target_pr[(source_url, source_ref)]
        del x[(target_url, target_ref)]
    return prs

def pop_prs_for_target(target_url, target_ref, default):
    prs = target_source_pr.pop((target_url, target_ref), default)
    for (source_url, source_ref), status in prs:
        x = source_target_pr[(source_url, source_ref)]
        del x[(target_url, target_ref)]
    return prs

def get_pr_status_by_source(source_url, source_ref):
    return source_target_pr[(source_url, source_ref)]

def get_pr_status_by_target(target_url, target_ref):
    return target_source_pr[(target_url, target_ref)]

batch_client = BatchClient(url=BATCH_SERVER_URL)

def cancel_existing_jobs(source_url, source, target_url, target):
    old_status = source_target_pr.get((source_url, source), {}).get((target_url, target), None)
    if old_status and old_status.state == 'running':
        id = old_status.job_id
        assert(id is not None)
        print(f'cancelling existing job {id} due to pr status update')
        batch_client.get_job(id).cancel()

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
        REPO_API_URL + url,
        headers=headers,
        json=json,
        data=data
    )
    if status_code and r.status_code != status_code:
        raise BadStatus({
            'method': 'post',
            'endpoint' : REPO_API_URL + url,
            'status_code' : r.status_code,
            'data': data,
            'json': json,
            'message': 'github error',
            'github_json': r.json()
        }, r.status_code)
    else:
        return r.json()

def get_repo(url, headers=None, status_code=None):
    if headers is None:
        headers = {}
    if 'Authorization' in headers:
        raise ValueError(
            'Header already has Authorization? ' + str(headers))
    headers['Authorization'] = 'token ' + oauth_token
    r = requests.get(
        REPO_API_URL + url,
        headers=headers
    )
    if status_code and r.status_code != status_code:
        raise BadStatus({
            'method': 'get',
            'endpoint' : REPO_API_URL + url,
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
    print('ERROR: ' + str(error.status_code) + ': ' + str(error.data))
    return jsonify(error.data), error.status_code

###############################################################################
### Endpoints

@app.route('/push', methods=['POST'])
def github_push():
    data = request.json
    print(data)
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
        print(f'ignoring ref push {ref} because it does not start with refs/heads/')
    return '', 200

@app.route('/pull_request', methods=['POST'])
def github_pull_request():
    data = request.json
    print(data)
    action = data['action']
    if action == 'opened' or action == 'synchronize':
        pr_number = str(data['number'])
        source_url = data['pull_request']['head']['repo']['clone_url']
        source_ref = data['pull_request']['head']['ref']
        source_sha = data['pull_request']['head']['sha']
        target_url = data['pull_request']['base']['repo']['clone_url']
        target_ref = data['pull_request']['base']['ref']
        pr_number = str(data['number'])
        target_sha = get_repo(
            f'git/refs/heads/{target_ref}',
            status_code=200
        )['object']['sha']
        cancel_existing_jobs(source_url, source_sha, target_url, target)
        update_pr_status(
            source_url,
            source_ref,
            target_url,
            target_ref,
            Status('pending', 'pending', source_sha, target_sha, pr_number)
        )
        post_repo(
            'statuses/' + source_sha,
            json={
                'state': 'pending',
                'description': f'build merged into {target_sha} pending',
                'context': CONTEXT
            },
            status_code=201
        )
        heal()
    # add elif to update review status
    else:
        print(f'ignoring github pull_request event of type {action} full json: {data}')
    return '', 200

@app.route('/ci_build_done', methods=['POST'])
def ci_build_done():
    data = request.json
    print(data)
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
        print(f'ignoring job for pr I did not think existed: {target_ref}:{target_sha} <- {source_ref}:{source_sha}')
    elif status.source_sha == source_sha and status.target_sha == target_sha:
        pr_number = attributes['pr_number']
        upload_public_gs_file_from_string(
            GCS_BUCKET,
            f'{source_sha}/{target_sha}/job-log',
            data['log']
        )
        upload_public_gs_file_from_filename(
            GCS_BUCKET,
            f'{source_sha}/{target_sha}/index.html',
            'index.html'
        )
        if exit_code == 0:
            print(f'test job {job_id} finished successfully for pr #{pr_number}')
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
                    'context': CONTEXT
                },
                status_code=201
            )
        else:
            print(f'test job {job_id} failed for pr #{pr_number} ({source_sha}) with exit code {exit_code} after merge with {target_sha}')
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
                    'target_url': f'https://storage.googleapis.com/hail-ci-0-1/{source_sha}/{target_sha}/index.html'
                },
                status_code=201
            )
        heal()
    else:
        print(f'ignoring completed job that I no longer care about: {target_ref}:{target_sha} <- {source_ref}:{source_sha}')

    return '', 200

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
            print(f'normally I would merge {source_url}:{source_ref} into {target_url}:{target_ref} with status {status.to_json()}')
        else:
            approved = [(source, status)
                        for source, status in prs.items()
                        if status.state == 'pending'
                        and status.review_state == 'approved']
            if len(approved) != 0:
                # pick oldest one instead
                ((source_url, source_ref), status) = approved[0]
                print(f'gonna test {target_url}:{target_ref} <- {source_url}:{source_ref}; {status.target_sha} <- {status.source_sha}')
                test_pr(source_url, source_ref, target_url, target_ref, status)
            else:
                untested = [(source, status)
                            for source, status in prs.items()
                            if status.state == 'pending']
                if len(untested) != 0:
                    # pick oldest one instead
                    ((source_url, source_ref), status) = untested[0]
                    print(f'gonna test {target_url}:{target_ref} <- {source_url}:{source_ref}; {status.target_sha} <- {status.source_sha}')
                    test_pr(source_url, source_ref, target_url, target_ref, status)
                else:
                    print('all prs have up-to-date ci state')

def test_pr(source_url, source_ref, target_url, target_ref, status):
    assert status.state == 'pending', str(status.to_json())
    attributes = {
        'pr_number': status.pr_number,
        'source_url': source_url,
        'source_ref': source_ref,
        'source_sha': status.source_sha,
        'target_url': target_url,
        'target_ref': target_ref,
        'target_sha': status.target_sha
    }
    print('creating job with attributes ' + str(attributes))
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
            'limits': {
                # our k8s seems to have >250mCPU available on each node
                # 'cpu' : '0.5',
                'memory': '2048M'
            }
        },
        callback=SELF_HOSTNAME + '/ci_build_done',
        attributes=attributes,
        volumes=[{
            'volume': { 'name' : 'hail-ci-0-1-service-account-key',
                        'secret' : { 'optional': False,
                                     'secretName': 'hail-ci-0-1-service-account-key' } },
            'volume_mount': { 'mountPath': '/secrets',
                              'name': 'hail-ci-0-1-service-account-key',
                              'readOnly': True }
        }]
    )
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

@app.route('/refresh_base/<target_ref>', methods=['POST'])
def refresh_github_state(target_ref):
    target_url = 'https://github.com/'+REPO[:-1]+'.git'    # send this in json data or something
    target_sha = get_repo(
        'git/refs/heads/'+target_ref,
        status_code=200
    )['object']['sha']
    pulls = get_repo(
        'pulls?state=open&base=' + target_ref,
        status_code=200
    )
    print(f'for target {target_ref} ({target_sha}) we found ' + str([pull['title'] for pull in pulls]))
    known_prs = pop_prs_for_target(target_url, target_ref, {})
    for pull in pulls:
        source_url = pull['head']['repo']['clone_url']
        source_ref = pull['head']['ref']
        source_sha = pull['head']['sha']
        pr_number = str(pull['number'])
        status = known_prs.pop((source_url, source_ref), None)
        if status and status.source_sha == source_sha and status.target_sha == target_sha:
            print(f'no change to knowledge of {target_url}:{target_ref} <- {source_url}:{source_ref}')
        else:
            print(f'updating knowledge of {target_url}:{target_ref} <- {source_url}:{source_ref}')
            review_status = review_status_net(pr_number)
            update_pr_status(
                source_url,
                source_ref,
                target_url,
                target_ref,
                Status('pending', review_status['state'], source_sha, target_sha, pr_number))

    if len(known_prs) != 0:
        print(f'some PRs have been invalidated by github state refresh: {known_prs}')
        for pr in known_prs:
            if pr.state == 'running':
                print(f'cancelling job {pr.job_id} for {pr.to_json()}')
                client.get_job(pr.job_id).cancel()

    return '', 200

# FIXME: have an end point to refresh batch/jobs state, needs a jobs endpoint

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
### SHA Status Manipulation

@app.route('/pr/<sha>/statuses')
def statuses(sha):
    json = get_repo(
        'commits/' + sha + '/statuses',
        status_code=200
    )
    return jsonify(json), 200

# @app.route('/pr/<sha>/fail')
# def fail(sha):
#     if request.args.get('secret') != secret:
#         return '403 Forbidden: bad secret query parameter', 403
#     post_repo(
#         'statuses/' + sha,
#         json={
#             'state': 'failure',
#             'description': 'manual override: fail',
#             'context': CONTEXT
#         },
#         status_code=201
#     )

#     return '', 200

# @app.route('/pr/<sha>/pending')
# def pending(sha):
#     if request.args.get('secret') != secret:
#         return '403 Forbidden: bad secret query parameter', 403
#     post_repo(
#         'statuses/' + sha,
#         json={
#             'state': 'pending',
#             'description': 'manual override: pending',
#             'context': CONTEXT
#         },
#         status_code=201
#     )

#     return '', 200

# @app.route('/pr/<sha>/success')
# def success(sha):
#     if request.args.get('secret') != secret:
#         return '403 Forbidden: bad secret query parameter', 403
#     post_repo(
#         'statuses/' + sha,
#         json={
#             'state': 'success',
#             'description': 'manual override: success',
#             'context': CONTEXT
#         },
#         status_code=201
#     )

#     return '', 200

if __name__ == '__main__':
    app.run()
