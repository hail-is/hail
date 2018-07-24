from flask import Flask, request, jsonify
from batch.client import *
import requests
from google.cloud import storage
import os

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

bases = {}
prs = {}
batch_client = BatchClient(url=BATCH_SERVER_URL)

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
    print('---')
    print([pr.to_json() for pr in prs])
    ref = data['ref']
    new_sha = data['after']
    if not ref.startswith('refs/heads/'):
        return '', 200
    else:
        ref = ref[11:]
        return refresh_github_state(ref, new_sha)

@app.route('/refresh_base/<ref>', methods=['POST'])
def refresh_github_state(ref, new_sha=None):
    if not new_sha:
        new_sha = update_base_ref(ref)
        if not new_sha:
            return 'no updates', 200
    pulls = get_repo(
        'pulls?state=open&base=' + ref,
        status_code=200
    )
    print(f'for target {ref} we found ' + str([pull['title'] for pull in pulls]))
    if len(pulls) == 0:
        if ref in bases:
            del bases[ref]
        print(f'no prs, nothing to do on {ref} push')
        return '', 200
    else:
        for pull in pulls:
            pr_number = pull['number']
            pr = prs.pop(pr_number, None)
            if pr is not None:
                job = pr.job
                print('cancelling job ' + str(job.id))
                job.cancel()
            # FIXME: The source hash isn't necessarily in the main repo.  Do I have
            # permission to slam statuses on third-party repos?
            post_repo(
                'statuses/' + pull['head']['sha'],
                json={
                    'state': 'pending',
                    'description': 'target branch commit changed, CI job was cancelled',
                    'context': CONTEXT
                },
                status_code=201
            )
        # FIXME: start with PRs that *were* passing
        pr_and_status = [{
            'number': str(pull['number']),
            'review': review_status(str(pull['number'])),
            'pull': pull
        } for pull in pulls]
        approved_prs = [x for x in pr_and_status if x['review']['state'] == 'APPROVED']
        if len(approved_prs) == 0:
            print('no approved prs, testing first unapproved pr: ' + str(pulls[0]['number']))
            test_pr(pulls[0])
        else:
            print('testing first approved pr: ' + approved_prs[0]['number'])
            test_pr(approved_prs[0]['pull'])
        return '', 200

@app.route('/pull_request', methods=['POST'])
def github_pull_request():
    data = request.json
    print(data)
    print('---')
    print(prs)
    action = data['action']
    pr_number = str(data['number'])
    if action == 'opened' or action == 'synchronize':
        existing_pr = prs.get(pr_number, None)
        if existing_pr is not None:
            id = existing_pr.job.id
            print(f'cancelling existing job {id}')
            batch_client.get_job(id).cancel()
        pr = data['pull_request']
        target_ref = pr['base']['ref']
        if target_ref not in bases:
            update_base_ref(target_ref)
        test_pr(pr)
    else:
        print('ignoring github pull_request event of type ' + action +
              ' full json: ' + str(data))
    return '', 200

def update_base_ref(ref):
    new_sha = get_repo(
        f'git/refs/heads/{ref}',
        status_code=200
    )['object']['sha']
    if bases[ref] != new_sha:
        bases[ref] = new_sha
        return new_sha
    else:
        return None

# expects bases[gh_pr_json['base']['ref]] to exist
def test_pr(gh_pr_json):
    print(gh_pr_json)
    pr_number = str(gh_pr_json['number'])
    source_repo_url = gh_pr_json['head']['repo']['clone_url']
    source_branch = gh_pr_json['head']['ref']
    source_hash = gh_pr_json['head']['sha']
    target_repo_url = gh_pr_json['base']['repo']['clone_url']
    target_branch = gh_pr_json['base']['ref']
    pr_target_hash = gh_pr_json['base']['sha']
    target_hash = bases[target_branch]
    if target_hash != pr_target_hash:
        print(f'target_hash {pr_target_hash} from PR is not the same as what we believe to be newest hash: {target_hash}')
    attributes = {
        'pr_number': pr_number,
        'source_repo_url': source_repo_url,
        'source_branch': source_branch,
        'source_hash': source_hash,
        'target_repo_url': target_repo_url,
        'target_branch': target_branch,
        'target_hash': target_hash
    }
    print('creating job with attributes ' + str(attributes))
    post_repo(
        'statuses/' + source_hash,
        json={
            'state': 'pending',
            'description': 'build merged into {target_hash} pending'.format(
                target_hash=target_hash),
            'context': CONTEXT
        },
        status_code=201
    )
    prs[pr_number] = PR(
        job=batch_client.create_job(
            PR_IMAGE,
            command=[
                '/bin/bash',
                '-c',
                PR_BUILD_SCRIPT],
            env={
                'SOURCE_REPO_URL': source_repo_url,
                'SOURCE_BRANCH': source_branch,
                'SOURCE_HASH': source_hash,
                'TARGET_REPO_URL': target_repo_url,
                'TARGET_BRANCH': target_branch,
                'TARGET_HASH': target_hash
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
        ),
        attributes=attributes
    )
    print('created PR job with id ' + str(prs[pr_number].job.id))

@app.route('/ci_build_done', methods=['POST'])
def ci_build_done():
    data = request.json
    print(data)
    jobid = data['id']
    pr_number = str(data['attributes']['pr_number'])
    pr = prs.pop(pr_number, None)
    if pr is None:
        print('test job finished for pr we no longer care about: ' + str(data))
    else:
        exit_code = data['exit_code']
        attributes = data['attributes']
        source_hash = attributes['source_hash']
        target_hash = attributes['target_hash']
        upload_public_gs_file_from_string(
            GCS_BUCKET,
            f'{source_hash}/{target_hash}/job-log',
            data['log']
        )
        upload_public_gs_file_from_filename(
            GCS_BUCKET,
            f'{source_hash}/{target_hash}/index.html',
            'index.html'
        )
        if exit_code == 0:
            print(f'test job {jobid} finished successfully for pr #{pr_number}')
            post_repo(
                'statuses/' + source_hash,
                json={
                    'state': 'success',
                    'description': 'successful build after merge with ' + target_hash,
                    'context': CONTEXT
                },
                status_code=201
            )
        else:
            print(f'test job {jobid} failed for pr #{pr_number} with exit code {exit_code}')
            status_message = f'failing build after merge with {target_hash}, exit code: {exit_code}'
            post_repo(
                'statuses/' + source_hash,
                json={
                    'state': 'failure',
                    'description': status_message,
                    'context': CONTEXT,
                    'target_url': f'https://storage.googleapis.com/hail-ci-0-1/{source_hash}/{target_hash}/index.html'
                },
                status_code=201
            )
    return '', 200

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
    upload(f)
    f.acl.all().grant_read()
    f.acl.save()
    if f.metadata:
        f.metadata['Cache-Control'] = 'private, max-age=0, no-transform'
    else:
        f.metadata = {'Cache-Control': 'private, max-age=0, no-transform'}

@app.route('/pr/<pr_number>/retest')
def retest(pr_number):
    return pr_number, 200

@app.route('/pr/<pr_number>/review_status')
def review_status_endpoint(pr_number):
    status = review_status(pr_number)
    return jsonify(status), 200

def review_status(pr_number):
    reviews = get_repo(
        'pulls/' + pr_number + '/reviews',
        status_code=200
    )
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
                'state': 'CHANGES_REQUESTED',
                'reviews': latest_state_by_login
            }
        elif (state == 'APPROVED'):
            at_least_one_approved = True

    if at_least_one_approved:
        return {
            'state': 'APPROVED',
            'reviews': latest_state_by_login
        }
    else:
        return {
            'state': 'PENDING',
            'reviews': latest_state_by_login
        }

@app.route('/pr/<pr_number>/mergeable')
def mergeable_endpoint(pr_number):
    m = mergeable(pr_number)
    return jsonify(m), 200

def mergeable(pr_number):
    pr = get_repo(
        'pulls/' + pr_number,
        status_code=200
    )
    status = get_repo(
        'commits/' + pr['head']['sha'] + '/status',
        status_code=200
    )
    ci_status = None
    for status in status['statuses']:
        if status['context'] == CONTEXT:
            ci_status = status
            break
    if ci_status is None:
        print(f'no ci_status found for {CONTEXT} assuming pending')
        ci_success = 'pending'
    else:
        ci_success = ci_status['state'] == 'success'
    status = review_status(pr_number)
    approved = status['state'] == 'APPROVED'
    if (ci_success and approved):
        return {
            'mergeable': True,
            'ci_success': ci_status,
            'review_status': status
        }
    else:
        return {
            'mergeable': False,
            'ci_success': ci_status,
            'review_status': status
        }

@app.route('/status')
def status():
    return jsonify({
        'bases': bases,
        'prs': { pr_number: pr.to_json() for pr_number, pr in prs.items() }
    }), 200

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
