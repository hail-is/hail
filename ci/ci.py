from batch.client import *
from flask import Flask, request, jsonify
from google.cloud import storage
from subprocess import run, CalledProcessError
import collections
import json
import logging
import os
import re
import requests
import threading
import time

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

try:
    INITIAL_WATCHED_REPOS = json.loads(os.environ['WATCHED_REPOS'])
except Exception as e:
    raise ValueError(
        'environment variable WATCHED_REPOS should be a json array of repos as '
        f'strings e.g. ["hail-is/hail"], but was: `{os.environ.get("WATCHED_REPOS", None)}`',
    ) from e
assert isinstance(INITIAL_WATCHED_REPOS, list), INITIAL_WATCHED_REPOS
assert all(isinstance(repo, str) for repo in INITIAL_WATCHED_REPOS), INITIAL_WATCHED_REPOS
GITHUB_URL = 'https://api.github.com/'
CONTEXT = 'hail-ci-0-1'
SELF_HOSTNAME = os.environ['SELF_HOSTNAME'] # 'http://35.232.159.176:3000'
BATCH_SERVER_URL = os.environ['BATCH_SERVER_URL'] # 'http://localhost:8888'
REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 5 * 60))
GCP_PROJECT = 'broad-ctsa'
VERSION = '0-1'
GCS_BUCKET = 'hail-ci-' + VERSION

log.info(f'INITIAL_WATCHED_REPOS {INITIAL_WATCHED_REPOS}')
log.info(f'BATCH_SERVER_URL {BATCH_SERVER_URL}')
log.info(f'SELF_HOSTNAME {SELF_HOSTNAME}')
log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')

class NoOAuthToken(Exception):
    pass
class NoPRBuildScript(Exception):
    pass
class BadStatus(Exception):
    def __init__(self, data, status_code):
        Exception.__init__(self, str(data))
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

def implies(antecedent, consequent):
    return not antecedent or consequent

class Status(object):
    states = set(['failure', 'pending', 'success', 'running', 'merged'])
    review_states = set(['approved', 'changes_requested', 'pending'])

    def __init__(self,
                 state,
                 review_state,
                 source_sha,
                 target_sha,
                 pr_number,
                 job_id=None,
                 docker_image=None,
                 gc=0):
        assert state in Status.states, f'{state} should be in {Status.states}'
        assert review_state in Status.review_states, f'{review_state} sohuld be in {Status.review_states}'
        assert implies(state == 'pending', job_id is None), f'{state} {job_id}'
        self.state = state
        self.review_state = review_state
        self.source_sha = source_sha
        self.target_sha = target_sha
        self.pr_number = pr_number
        self.job_id = job_id
        self.docker_image = docker_image
        self.gc = gc

    class Sentinel(object):
        pass
    keep = Sentinel()

    def copy(self,
             state=keep,
             review_state=keep,
             source_sha=keep,
             target_sha=keep,
             pr_number=keep,
             job_id=keep,
             docker_image=keep,
             gc=keep):
        return Status(state=self.state if state is Status.keep else state,
                      review_state=self.review_state if review_state is Status.keep else review_state,
                      source_sha=self.source_sha if source_sha is Status.keep else source_sha,
                      target_sha=self.target_sha if target_sha is Status.keep else target_sha,
                      pr_number=self.pr_number if pr_number is Status.keep else pr_number,
                      job_id=self.job_id if job_id is Status.keep else job_id,
                      docker_image=self.docker_image if docker_image is Status.keep else docker_image,
                      gc=self.gc if gc is Status.keep else gc)

    def github_state_up_to_date(self, github_state):
        return (self.state == github_state or
                self.state == 'running' and github_state == 'pending')

    # but target_url, target_ref, source_url, source_ref on Status so I can
    # compute docker_image myself
    def target_change(self, new_sha, docker_image):
        if self.state == 'merged':
            return self
        else:
            return self.copy(
                state='pending',
                target_sha=new_sha,
                job_id=None,
                docker_image=docker_image
            )

    def build_succeeded(self, pr_number, job_id):
        if self.state == 'merged':
            log.warning(
                f'was notified of succeeding build for already merged PR! '
                f'{self.pr_number}, {self.job_id}, {self.to_json()}')
            return self
        else:
            return self.copy(
                state='success',
                # what does it mean if pr_number is different?
                pr_number=pr_number,
                # what does it mean if job_id is different?
                job_id=job_id)

    def build_failed(self, pr_number, job_id):
        if self.state == 'merged':
            log.error(
                f'was notified of failing build for already merged PR! '
                f'{self.pr_number}, {self.job_id}, {self.to_json()}')
            return self
        else:
            return self.copy(
                state='failure',
                # what does it mean if pr_number is different?
                pr_number=pr_number,
                # what does it mean if job_id is different?
                job_id=job_id)

    def merged(self):
        return self.copy(state='merged')

    def running(self, job_id):
        assert self.state != 'merged', self.to_json()
        return self.copy(state='running', job_id=job_id)

    def pending(self):
        assert self.state != 'merged', self.to_json()
        return self.copy(state='pending', job_id=None)

    def survived_a_gc(self):
        assert self.state == 'merged' and self.gc == 0, self.to_json()
        return self.copy(gc=self.gc+1)

    def to_json(self):
        return {
            'state': self.state,
            'review_state': self.review_state,
            'source_sha': self.source_sha,
            'target_sha': self.target_sha,
            'pr_number': self.pr_number,
            'job_id': self.job_id,
            'docker_image': self.docker_image,
            'gc': self.gc
        }

auto_merge_on = True
watched_repos = INITIAL_WATCHED_REPOS
target_source_pr = {}
source_target_pr = {}

def update_pr_status(source_url, source_ref, target_url, target_ref, status):
    if (target_url, target_ref) not in target_source_pr:
        target_source_pr[(target_url, target_ref)] = {}
    if (source_url, source_ref) not in source_target_pr:
        source_target_pr[(source_url, source_ref)] = {}
    target_source_pr[(target_url, target_ref)][(source_url, source_ref)] = status
    source_target_pr[(source_url, source_ref)][(target_url, target_ref)] = status

def remove_pr(source_url, source_ref, target_url, target_ref):
    target_source_pr[(target_url, target_ref)].pop((source_url, source_ref))
    source_target_pr[(source_url, source_ref)].pop((target_url, target_ref))

def get_pr_status(source_url, source_ref, target_url, target_ref, default=None):
    x = source_target_pr.get((source_url, source_ref), {}).get((target_url, target_ref), default)
    y = target_source_pr.get((target_url, target_ref), {}).get((source_url, source_ref), default)
    assert x == y, str(x) + str(y)
    return x

def pop_prs_for_target(target_url, target_ref, default):
    prs = target_source_pr.pop((target_url, target_ref), None)
    if prs is None:
        return default
    for (source_url, source_ref), status in prs.items():
        x = source_target_pr[(source_url, source_ref)]
        del x[(target_url, target_ref)]
    return prs

def get_pr_status_by_source(source_url, source_ref):
    return source_target_pr.get((source_url, source_ref), {})

def get_pr_status_by_target(target_url, target_ref):
    return target_source_pr.get((target_url, target_ref), {})

def get_pr_targets():
    return target_source_pr.keys()

batch_client = BatchClient(url=BATCH_SERVER_URL)

def cancel_existing_jobs(source_url, source_ref, target_url, target_ref):
    old_status = source_target_pr.get((source_url, source_ref), {}).get((target_url, target_ref), None)
    if old_status and old_status.state == 'running':
        id = old_status.job_id
        assert id is not None
        log.info(f'cancelling existing job {id} due to pr status update')
        try_to_cancel_job_by_id(id)

@app.route('/status')
def status():
    return jsonify({
        'auto_merge_on': auto_merge_on,
        'watched_repos': watched_repos,
        'prs': [{ 'source_url': source_url,
           'source_ref': source_ref,
           'target_url': target_url,
           'target_ref': target_ref,
           'status': status.to_json()
        } for ((source_url, source_ref), prs) in source_target_pr.items()
         for ((target_url, target_ref), status) in prs.items()]
    })

@app.route('/status/enable_auto_merge', methods=['POST'])
def enable_auto_merge():
    global auto_merge_on
    auto_merge_on = True
    return '', 200

@app.route('/status/disable_auto_merge', methods=['POST'])
def disable_auto_merge():
    global auto_merge_on
    auto_merge_on = False
    return '', 200

@app.route('/status/watched_repos')
def get_watched_repos():
    return jsonify(watched_repos)

@app.route('/status/watched_repos/<repo>/remove', methods=['POST'])
def remove_watched_repo(repo):
    global watched_repos
    watched_repos = [r for r in watched_repos if r != repo]
    return jsonify(watched_repos)

@app.route('/status/watched_repos/<repo>/add', methods=['POST'])
def add_watched_repo(repo):
    global watched_repos
    watched_repos = [r for r in watched_repos if r != repo]
    watched_repos.append(repo)
    return jsonify(watched_repos)

###############################################################################
### post and get helpers

def post_repo(repo, url, headers=None, json=None, data=None, status_code=None):
    return verb_repo(
        'post',
        repo,
        url,
        headers=headers,
        json=json,
        data=data,
        status_code=status_code)

def get_repo(repo, url, headers=None, status_code=None):
    return verb_repo(
        'get',
        repo,
        url,
        headers=headers,
        status_code=status_code)

def put_repo(repo, url, headers=None, json=None, data=None, status_code=None):
    return verb_repo(
        'put',
        repo,
        url,
        headers=headers,
        json=json,
        data=data,
        status_code=status_code)

def get_github(url, headers=None, status_code=None):
    return verb_github(
        'get',
        url,
        headers=headers,
        status_code=status_code)

def verb_repo(verb,
              repo,
              url,
              headers=None,
              json=None,
              data=None,
              status_code=None):
    return verb_github(
        verb,
        f'repos/{repo}/{url}',
        headers=headers,
        json=json,
        data=data,
        status_code=status_code)

verbs = set(['post', 'put', 'get'])
def verb_github(verb,
                url,
                headers=None,
                json=None,
                data=None,
                status_code=None):
    if isinstance(status_code, int):
        status_codes = [status_code]
    else:
        status_codes = status_code
    assert verb in verbs
    assert implies(verb == 'post' or verb == 'put', json is not None or data is not None)
    assert implies(verb == 'get', json is None and data is None)
    if headers is None:
        headers = {}
    if 'Authorization' in headers:
        raise ValueError(
            'Header already has Authorization? ' + str(headers))
    headers['Authorization'] = 'token ' + oauth_token
    full_url = f'{GITHUB_URL}{url}'
    if verb == 'get':
        r = requests.get(full_url, headers=headers, timeout=5)
        output = r.json()
        if 'Link' in r.headers:
            assert isinstance(output, list), output
            link = r.headers['Link']
            url = github_link_header_to_maybe_next(link)
            while url is not None:
                r = requests.get(url, headers=headers, timeout=5)
                link = r.headers['Link']
                output.extend(r.json())
                url = github_link_header_to_maybe_next(link)
    elif verb == 'post':
        r = requests.post(full_url, headers=headers, data=data, json=json, timeout=5)
        output = r.json()
    elif verb == 'put':
        r = requests.put(full_url, headers=headers, data=data, json=json, timeout=5)
        output = r.json()
    if status_codes and r.status_code not in status_codes:
        raise BadStatus({
            'method': verb,
            'endpoint' : full_url,
            'status_code' : {
                'actual': r.status_code,
                'expected': status_codes
            },
            'message': 'github error',
            'data': data,
            'json': json,
            'github_json': output
        }, r.status_code)
    else:
        if isinstance(status_code, list):
            return (output, r.status_code)
        else:
            return output

github_link = re.compile('\s*<(http.+page=[0-9]+)>; rel="([A-z]+)"\s*')
def github_link_header_to_maybe_next(link):
    # I cannot find rigorous documentation on the format, but this seems to
    # work?
    link_texts = link.split(',')
    links = {}
    for t in link_texts:
        m = github_link.match(t)
        assert m is not None, f'{m} {t}'
        links[m[2]] = m[1]
    return links.get('next', None)

###############################################################################
### Error Handlers

@app.errorhandler(BadStatus)
def handle_invalid_usage(error):
    log.exception('bad status found when making request')
    return jsonify(error.data), error.status_code

###############################################################################
### Endpoints

clone_url_to_repo = re.compile('https://github.com/([^/]+)/([^/]+).git')
def repo_from_url(url):
    m = clone_url_to_repo.match(url)
    assert m and m.lastindex and m.lastindex == 2, f'{m} {url}'
    return m[1] + '/' + m[2]

def url_from_repo(repo):
    return f'https://github.com/{repo}.git'

@app.route('/push', methods=['POST'])
def github_push():
    data = request.json
    ref = data['ref']
    new_sha = data['after']
    if ref.startswith('refs/heads/'):
        target_ref = ref[11:]
        target_url = data['repository']['clone_url']
        pr_statuses = get_pr_status_by_target(target_url, target_ref)
        log.info(
            f'{target_url}:{target_ref} was updated to {new_sha}, '
            f'{len(pr_statuses)} (possibly recently merged) PRs target this ref')
        for (source_url, source_ref), status in pr_statuses.items():
            if (status.target_sha != new_sha and status.state != 'merged'):
                cancel_existing_jobs(source_url, source_ref, target_url, target_ref)
                try:
                    post_repo(
                        repo_from_url(target_url),
                        'statuses/' + status.source_sha,
                        json={
                            'state': 'pending',
                            'description': f'build merged into {new_sha} pending',
                            'context': CONTEXT
                        },
                        status_code=201
                    )
                except BadStatus as e:
                    if e.status_code == 422:
                        log.exception(
                            f'Too many statuses applied to {source_sha}! This is a '
                            f'dangerous situation because I can no longer block merging '
                            f'of failing PRs.')
                    else:
                        raise e
                docker_image = get_build_image(target_url, target_ref, new_sha,
                                               source_url, source_ref, status.source_sha)
                update_pr_status(
                    source_url,
                    source_ref,
                    target_url,
                    target_ref,
                    status.target_change(new_sha, docker_image))
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
        target_sha = get_sha_for_target_ref(target_url, target_ref)
        docker_image = get_build_image(target_url, target_ref, target_sha,
                                       source_url, source_ref, source_sha)
        review_status = review_status_net(repo_from_url(target_url), pr_number)
        cancel_existing_jobs(source_url, source_ref, target_url, target_ref)
        status = Status('pending',
                        review_status['state'],
                        source_sha,
                        target_sha,
                        pr_number,
                        docker_image=docker_image)
        update_pr_status(source_url, source_ref, target_url, target_ref, status)
        try:
            post_repo(
                repo_from_url(target_url),
                'statuses/' + source_sha,
                json={
                    'state': 'pending',
                    'description': f'build merged into {target_sha} pending',
                    'context': CONTEXT
                },
                status_code=201
            )
        except BadStatus as e:
            if e.status_code == 422:
                log.exception(
                    f'Too many statuses applied to {source_sha}! This is a '
                    f'dangerous situation because I can no longer block merging '
                    f'of failing PRs.')
            else:
                raise e
        # eagerly trigger a build (even if master has other pending PRs
        # building) since the requester introduced new changes
        test_pr(source_url, source_ref, target_url, target_ref, status)
    else:
        log.warning(f'ignoring github pull_request event of type {action}.')
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
            status.build_succeeded(pr_number, job_id)
        )
        try:
            post_repo(
                repo_from_url(target_url),
                'statuses/' + source_sha,
                json={
                    'state': 'success',
                    'description': 'successful build after merge with ' + target_sha,
                    'context': CONTEXT,
                    'target_url': f'https://storage.googleapis.com/{GCS_BUCKET}/{source_sha}/{target_sha}/index.html'
                },
                status_code=201
            )
        except BadStatus as e:
            if e.status_code == 422:
                log.exception(
                    f'Too many statuses applied to {source_sha}! This is a '
                    f'dangerous situation because I can no longer block merging '
                    f'of failing PRs.')
            else:
                raise e
    else:
        log.info(f'test job {job_id} failed for pr #{pr_number} ({source_sha}) with exit code {exit_code} after merge with {target_sha}')
        update_pr_status(
            source_url,
            source_ref,
            target_url,
            target_ref,
            status.build_failed(pr_number, job_id)
        )
        status_message = f'failing build ({exit_code}) after merge with {target_sha}'
        try:
            post_repo(
                repo_from_url(target_url),
                'statuses/' + source_sha,
                json={
                    'state': 'failure',
                    'description': status_message,
                    'context': CONTEXT,
                    'target_url': f'https://storage.googleapis.com/{GCS_BUCKET}/{source_sha}/{target_sha}/index.html'
                },
                status_code=201
            )
        except BadStatus as e:
            if e.status_code == 422:
                log.exception(
                    f'Too many statuses applied to {source_sha}! This is a '
                    f'dangerous situation because I can no longer block merging '
                    f'of failing PRs.')
            else:
                raise e

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
        if not auto_merge_on and len(ready_to_merge) != 0:
            log.info(f'auto merge is disabled, so I will not attempt to merge any of {[(x, y.to_json()) for (x, y) in ready_to_merge]}')
        if auto_merge_on and len(ready_to_merge) != 0:
            # pick oldest one instead
            ((source_url, source_ref), status) = ready_to_merge[0]
            log.info(f'merging {source_url}:{source_ref} into {target_url}:{target_ref} with status {status.to_json()}')
            pr_number = status.pr_number
            (gh_response, status_code) = put_repo(
                repo_from_url(target_url),
                f'pulls/{pr_number}/merge',
                json={
                    'merge_method': 'squash',
                    'sha': status.source_sha
                },
                status_code=[200, 409]
            )
            if status_code == 200:
                log.info(
                    f'successful merge of {source_url}:{source_ref} into '
                    f'{target_url}:{target_ref} with status {status.to_json()}')
            else:
                assert status_code == 409, f'{status_code} {gh_response}'
                log.warning(
                    f'failure to merge {source_url}:{source_ref} into '
                    f'{target_url}:{target_ref} with status {status.to_json()} '
                    f'due to {status_code} {gh_response}, removing PR, github '
                    f'state refresh will recover and retest if necessary')
            update_pr_status(
                source_url,
                source_ref,
                target_url,
                target_ref,
                status.merged()
            )
            # FIXME: eagerly update statuses for all PRs targeting this branch
        else:
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
                    log.info(
                        f'no approved and running prs, will build: '
                        f'{target_url}:{target_ref} <- {source_url}:{source_ref} '
                        f'{status.to_json()}')
                    test_pr(source_url, source_ref, target_url, target_ref, status)
                else:
                    untested = [(source, status)
                                for source, status in prs.items()
                                if status.state == 'pending']
                    if len(untested) != 0:
                        log.info('no approved prs, will build all PRs with out of date statuses')
                        for (source_url, source_ref), status in untested:
                            log.info(
                                f'building: '
                                f'{target_url}:{target_ref} <- {source_url}:{source_ref}'
                                f'{status.to_json()}')
                            test_pr(source_url, source_ref, target_url, target_ref, status)
                    else:
                        log.info(f'all prs are tested or running for {target_url}:{target_ref}')

@app.route('/gc', methods=['POST'])
def gc():
    for (target_url, target_ref), prs in target_source_pr.items():
        log.info(f'attempting to clean up old merged PRs for {target_url}:{target_ref}')
        ready_to_gc = [(source, status)
                       for source, status in prs.items()
                       if status.state == 'merged' and status.gc > 0]
        if len(ready_to_gc) > 0:
            ready_to_gc_message = [f'{source_url}:{source_ref} {status.to_json()}'
                                   for ((source_url, source_ref), status) in ready_to_gc]
            log.info(f'removing {len(ready_to_gc)} old merged PRs for {target_url}:{target_ref}: {ready_to_gc_message}')
            for ((source_url, source_ref), status) in ready_to_gc:
                remove_pr(source_url, source_ref, target_url, target_ref)
        merged_and_old = [(source, status)
                          for source, status in prs.items()
                          if status.state == 'merged' and status.gc == 0]
        for ((source_url, source_ref), status) in merged_and_old:
            update_pr_status(
                source_url,
                source_ref,
                target_url,
                target_ref,
                status.survived_a_gc()
            )
    return '', 200

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

    review_status = review_status_net(repo_from_url(target_url), pr_number)
    cancel_existing_jobs(source_url, source_ref, target_url, target_ref)
    docker_image = get_build_image(target_url, target_ref, target_sha,
                                   source_url, source_ref, source_sha)
    status = Status('pending', review_status['state'], source_sha, target_sha, pr_number, docker_image=docker_image)
    update_pr_status(source_url, source_ref, target_url, target_ref, status)
    try:
        post_repo(
            repo_from_url(target_url),
            'statuses/' + source_sha,
            json={
                'state': 'pending',
                'description': f'(MANUALLY FORCED REBUILD) build merged into {target_sha} pending',
                'context': CONTEXT
            },
            status_code=201
        )
    except BadStatus as e:
        if e.status_code == 422:
            log.exception(
                f'Too many statuses applied to {source_sha}! This is a '
                f'dangerous situation because I can no longer block merging '
                f'of failing PRs.')
        else:
            raise e
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
    assert status.docker_image is not None, ((target_url, target_ref), (source_url, source_ref), status.to_json())
    job=batch_client.create_job(
        status.docker_image,
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
    try:
        post_repo(
            repo_from_url(target_url),
            'statuses/' + status.source_sha,
            json={
                'state': 'pending',
                'description': f'build merged into {status.target_sha} running {job.id}',
                'context': CONTEXT
            },
            status_code=201
        )
    except BadStatus as e:
        if e.status_code == 422:
            log.exception(
                f'Too many statuses applied to {source_sha}! This is a '
                f'dangerous situation because I can no longer block merging '
                f'of failing PRs.')
        else:
            raise e
    log.info(f'successfully updated github #{status.pr_number} ({status.source_sha}) about job {job.id}')
    update_pr_status(
        source_url,
        source_ref,
        target_url,
        target_ref,
        status.running(job.id)
    )
    log.info(f'successfully updated status about job {job.id}')

def get_sha_for_target_ref(url, ref):
    return get_repo(
        repo_from_url(url),
        'git/refs/heads/' + ref,
        status_code=200
    )['object']['sha']

@app.route('/refresh_github_state', methods=['POST'])
def refresh_github_state():
    for repo in watched_repos:
        log.info(f'refreshing state for {repo}')
        try:
            target_url = url_from_repo(repo)
            pulls = get_repo(
                repo_from_url(target_url),
                'pulls?state=open',
                status_code=200
            )
            log.info(f'found {len(pulls)} open PRs in this repo')
            pulls_by_target = collections.defaultdict(list)
            for pull in pulls:
                target_ref = pull['base']['ref']
                pulls_by_target[target_ref].append(pull)
            log.info(f'found {len(pulls_by_target)} target branches with open PRs in this repo: {pulls_by_target.keys()}')
            gh_targets = set([(target_url, ref) for ref in pulls_by_target.keys()])
            for (dead_target_url, dead_target_ref) in set(get_pr_targets()) - gh_targets:
                prs = pop_prs_for_target(dead_target_url, dead_target_ref, {})
                if len(prs) != 0:
                    log.info(
                        f'no open PRs for {dead_target_url}:{dead_target_ref} on GitHub, '
                        f'forgetting the {len(prs)} PRs I was tracking')
            for target_ref, pulls in pulls_by_target.items():
                target_sha = get_sha_for_target_ref(target_url, target_ref)
                log.info(f'for target {target_ref} ({target_sha}) we found ' + str([pull['title'] for pull in pulls]))
                known_prs = pop_prs_for_target(target_url, target_ref, {})
                for pull in pulls:
                    source_url = pull['head']['repo']['clone_url']
                    source_ref = pull['head']['ref']
                    source_sha = pull['head']['sha']
                    pr_number = str(pull['number'])
                    status = known_prs.pop((source_url, source_ref), None)
                    review_status = review_status_net(repo_from_url(target_url), pr_number)
                    latest_state = status_state_net(repo_from_url(target_url), source_sha, target_sha)
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
                        try:
                            docker_image = get_build_image(target_url, target_ref, target_sha,
                                                           source_url, source_ref, source_sha)
                            if status and status.state == 'running' and latest_state == 'pending':
                                latest_state = 'running'
                                job_id = status.job_id
                            else:
                                job_id = None
                            new_status = Status(
                                latest_state,
                                latest_review_state,
                                source_sha,
                                target_sha,
                                pr_number,
                                job_id=job_id,
                                docker_image=docker_image)
                            log.info(f'updating knowledge of {target_url}:{target_ref} <- {source_url}:{source_ref} '
                                     f'to {new_status.to_json()} from {status.to_json() if status else status})')
                            update_pr_status(
                                source_url,
                                source_ref,
                                target_url,
                                target_ref,
                                new_status)
                        except (FileNotFoundError, CalledProcessError) as e:
                            log.exception(
                                f'could not get docker image due to {e}, will '
                                f'ignore this PR for now '
                                f'{target_url}:{target_ref} <- {source_url}:{source_ref}')
                if len(known_prs) != 0:
                    known_prs_json = [(x, status.to_json()) for (x, status) in known_prs.items()]
                    log.info(f'some PRs have been invalidated by github state refresh: {known_prs_json}')
                    for (source_url, source_ref), status in known_prs.items():
                        if status.state == 'running':
                            log.info(f'cancelling job {status.job_id} for {status.to_json()}')
                            try_to_cancel_job_by_id(status.job_id)
        except Exception as e:
            log.exception(f'could not refresh state from {repo} due to {e}')

    return '', 200

# FIXME: have an end point to refresh batch/jobs state, needs a jobs endpoint
@app.route('/refresh_batch_state', methods=['POST'])
def refresh_batch_state():
    jobs = batch_client.list_jobs()
    latest_jobs = {}
    for job in jobs:
        t = job.attributes.get('type', None)
        if t and t == 'hail-ci-' + VERSION:
            assert 'target_url' in job.attributes, job.attributes
            repo = repo_from_url(job.attributes['target_url'])
            if repo in watched_repos:
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
                    job2_state = job2.cached_status()['state']
                    job_state = job.cached_status()['state']
                    if (batch_job_state_smaller_is_closer_to_complete(job2_state, job_state) < 0 or
                        job2.id < job.id):
                        log.info(f'cancelling {job2.id}, preferring {job.id}, {job2.attributes} {job.attributes} ')
                        try_to_cancel_job(job2)
                        latest_jobs[key] = job
                    else:
                        log.info(f'cancelling {job.id}, preferring {job2.id}, {job2.attributes} {job.attributes} ')
                        try_to_cancel_job(job)

    for job in latest_jobs.values():
        job_id = job.id
        job_status = job.cached_status()
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
                             f'to {build_state} {status.review_state} {target_sha} <- {source_sha} (was: {status.to_json() if status else status})')
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
                    try:
                        post_repo(
                            repo_from_url(target_url),
                            'statuses/' + source_sha,
                            json={
                                'state': 'pending',
                                'description': f'build merged into {target_sha} pending (job {job_id} was cancelled)',
                                'context': CONTEXT
                            },
                            status_code=201
                        )
                    except BadStatus as e:
                        if e.status_code == 422:
                            log.exception(
                                f'Too many statuses applied to {source_sha}! This is a '
                                f'dangerous situation because I can no longer block merging '
                                f'of failing PRs.')
                        else:
                            raise e
                    update_pr_status(
                        source_url, source_ref, target_url, target_ref,
                        status.pending()
                    )
                else:
                    log.info(f'already knew {target_url}:{target_ref} <- {source_url}:{source_ref} '
                             f'was pending for {target_sha} <- {source_sha}')
            else:
                if status.state != 'running' or status.job_id != job_id:
                    log.info(f'updating knowledge of {target_url}:{target_ref} <- {source_url}:{source_ref} '
                             f'to running {status.review_state} {target_sha} <- {source_sha}')
                    assert job_state == 'Created', f'{job_state}'
                    try:
                        post_repo(
                            repo_from_url(target_url),
                            'statuses/' + status.source_sha,
                            json={
                                'state': 'pending',
                                'description': f'build merged into {status.target_sha} running {job_id}',
                                'context': CONTEXT
                            },
                            status_code=201
                        )
                    except BadStatus as e:
                        if e.status_code == 422:
                            log.exception(
                                f'Too many statuses applied to {source_sha}! This is a '
                                f'dangerous situation because I can no longer block merging '
                                f'of failing PRs.')
                        else:
                            raise e
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

def try_to_cancel_job_by_id(id):
    try:
        job = batch_client.get_job(id)
        try_to_cancel_job(job)
    except requests.exceptions.HTTPError as e:
        log.warning(f'while trying to cancel a job, could not get job {id} due to {e}')

def try_to_cancel_job(job):
    try:
        job.cancel()
    except requests.exceptions.HTTPError as e:
        log.warning(f'could not cancel job {job.id} due to {e}')

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

def get_build_image(source_url, source_ref, source_sha,
                    target_url, target_ref, target_sha):
    d = os.getcwd()
    try:
        target_repo = repo_from_url(target_url)
        if not os.path.isdir(target_repo):
            os.makedirs(target_repo, exist_ok=True)
            os.chdir(target_repo)
            run(['git', 'clone', target_url, '.'], check=True)
        else:
            os.chdir(target_repo)
        source_repo = repo_from_url(source_url)
        if run(['/bin/sh', '-c', f'git remote | grep -q {source_repo}']).returncode != 0:
            run(['git', 'remote', 'add', source_repo, source_url], check=True)
        run(['git', 'fetch', 'origin'], check=True)
        run(['git', 'fetch', source_repo], check=True)
        run(['git', 'checkout', target_sha], check=True)
        run(['git', 'config', '--global', 'user.email', 'hail-ci-leader@example.com'], check=True)
        run(['git', 'config', '--global', 'user.name', 'hail-ci-leader'], check=True)
        run(['git', 'merge', source_sha, '-m', 'foo'], check=True)
        # a force push that removes refs could fail us... not sure what we
        # should do in that case. maybe 500'ing is OK?
        with open('hail-ci-build-image', 'r') as f:
            return f.read().strip()
    finally:
        run(['git', 'reset', '--merge'], check=True)
        os.chdir(d)


###############################################################################
### Review Status

@app.route('/pr/<user>/<repo>/<pr_number>/review_status')
def review_status_endpoint(user, repo, pr_number):
    status = review_status_net(f'{user}/{repo}', pr_number)
    return jsonify(status), 200

def review_status_net(repo, pr_number):
    reviews = get_repo(
        repo,
        'pulls/' + pr_number + '/reviews',
        status_code=200
    )
    return review_status(reviews)

# NB: a SHA that is used by a PR may have a status in *either* the target or the
# source repo. We always use the target_repo.
def status_state_net(repo, source_sha, target_sha):
    statuses = get_repo(
        repo,
        'commits/' + source_sha + '/statuses',
        status_code=200
    )
    my_statuses = [s for s in statuses if s['context'] == CONTEXT]
    latest_state = 'pending'
    if len(my_statuses) != 0:
        latest_status = my_statuses[0]
        if target_sha in latest_status['description']:
            latest_state = latest_status['state']
            log.info(f'latest status for {repo}:{source_sha} is {latest_state} because of {latest_status}')
        else:
            log.info(f'latest status for {repo}:{source_sha} does not include {target_sha} in description: {latest_status}')
    else:
        log.info(f'no status found for {repo}:{source_sha}')
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

@app.route('/pr/<user>/<repo>/<sha>/statuses')
def statuses(user, repo, sha):
    json = get_repo(
        f'{user}/{repo}',
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
    time.sleep(1)
    while True:
        try:
           r = requests.post('http://127.0.0.1:5000/refresh_github_state', timeout=120)
           r.raise_for_status()
           r = requests.post('http://127.0.0.1:5000/refresh_batch_state', timeout=120)
           r.raise_for_status()
           r = requests.post('http://127.0.0.1:5000/heal', timeout=120)
           r.raise_for_status()
           r = requests.post('http://127.0.0.1:5000/gc', timeout=120)
           r.raise_for_status()
        except Exception as e:
            log.error(f'Could not poll due to exception: {e}')
            pass
        time.sleep(REFRESH_INTERVAL_IN_SECONDS)

if __name__ == '__main__':
    poll_thread = threading.Thread(target=polling_event_loop)
    poll_thread.start()
    flask_event_loop()
