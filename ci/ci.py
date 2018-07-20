from flask import Flask, request, jsonify
from batch.client import *
import requests

REPO = 'danking/docker-build-test/'
REPO_API_URL = 'https://api.github.com/repos/' + REPO
CONTEXT = 'hail-ci'
PR_IMAGE = 'gcr.io/broad-ctsa/hail-pr-builder:latest'
SELF_HOSTNAME = 'http://35.232.159.176:3000'
BATCH_SERVER_URL='http://localhost:8888'

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


###############################################################################
### Global State & Setup

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
    return jsonify(error.data), error.status_code

###############################################################################
### Endpoints

@app.route('/push', methods=['POST'])
def github_push():
    data = request.json
    print(data)
    ref = data['ref']
    if not ref.startswith('refs/heads/'):
        return '', 200
    else:
        ref = ref[11:]
        pulls = get_repo(

            'pulls?state=open&base=' + ref,
            status_code=200
        )
        print('for target ' + ref + ' we found ' + str(pulls))
        if len(pulls) == 0:
            print('no prs, nothing to do on ' + ref + ' push')
            return '', 200
        else:
            for pull in pulls:
                pr_number = pull['number']
                pr = prs.get(pr_number, None)
                if pr is not None:
                    del prs[pr_number]
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
            pr_and_status = [(str(pull['number']), review_status(str(pull['number']))) for pull in pulls]
            approved_prs = [x for x in pr_and_status if x[1]['state'] == 'APPROVED']
            if len(approved_prs) == 0:
                print('no approved prs, testing first unapproved pr: ' + str(pulls[0]['number']))
                test_pr(pulls[0])
            else:
                print('testing first approved pr: ' + str(approved_prs[0]['number']))
                test_pr(approved_prs[0])
            return '', 200

@app.route('/pull_request', methods=['POST'])
def github_pull_request():
    data = request.json
    print(data)
    action = data['action']
    pr_number = str(data['number'])
    if action == 'opened' or action == 'synchronize':
        existing_pr = prs.get(pr_number, None)
        if existing_pr is not None:
            id = existing_pr.job.id
            print('cancelling existing job {id}'.format(id=id))
            batch_client.get_job(id).cancel()
        test_pr(data['pull_request'])
    else:
        print('ignoring github pull_request event of type ' + action +
              ' full json: ' + str(data))
    return '', 200

def test_pr(gh_pr_json):
    pr_number = str(gh_pr_json['number'])
    source_repo_url = gh_pr_json['head']['repo']['clone_url']
    source_branch = gh_pr_json['head']['ref']
    source_hash = gh_pr_json['head']['sha']
    target_repo_url = gh_pr_json['base']['repo']['clone_url']
    target_branch = gh_pr_json['base']['ref']
    target_hash = gh_pr_json['base']['sha']
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
            callback=SELF_HOSTNAME + '/ci_build_done',
            attributes=attributes
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
    pr = prs.get(pr_number, None)
    if pr is None:
        print('test job finished for pr we no longer care about: ' + str(data))
    else:
        del prs[pr_number]
        exit_code = data['exit_code']
        attributes = data['attributes']
        source_hash = attributes['source_hash']
        target_hash = attributes['target_hash']
        if exit_code == 0:
            print('test job {jobid} finished successfully for pr #{pr_number}'.format(
                jobid=jobid, pr_number=pr_number))
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
            print(('test job {jobid} failed for pr #{pr_number} with exit code'
                   '{exit_code}').format(
                       jobid=jobid, pr_number=pr_number, exit_code=exit_code))
            post_repo(
                'statuses/' + source_hash,
                json={
                    'state': 'failure',
                    'description': ('failing build after merge with ' +
                                    target_hash +
                                    ', exit code: ' + exit_code),
                    'context': CONTEXT
                },
                status_code=201
            )
    return '', 200

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
        print('no ci_status found for ' + CONTEXT + ' assuming pending')
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
    return jsonify({ pr_number: pr.to_json() for pr_number, pr in prs.items() }), 200

###############################################################################
### SHA Status Manipulation

@app.route('/pr/<sha>/statuses')
def statuses(sha):
    json = get_repo(
        'commits/' + sha + '/statuses',
        status_code=200
    )
    return jsonify(json), 200

@app.route('/pr/<sha>/fail')
def fail(sha):
    if request.args.get('secret') != secret:
        return '403 Forbidden: bad secret query parameter', 403
    post_repo(
        'statuses/' + sha,
        json={
            'state': 'failure',
            'description': 'manual override: fail',
            'context': CONTEXT
        },
        status_code=201
    )

    return '', 200

@app.route('/pr/<sha>/pending')
def pending(sha):
    if request.args.get('secret') != secret:
        return '403 Forbidden: bad secret query parameter', 403
    post_repo(
        'statuses/' + sha,
        json={
            'state': 'pending',
            'description': 'manual override: pending',
            'context': CONTEXT
        },
        status_code=201
    )

    return '', 200

@app.route('/pr/<sha>/success')
def success(sha):
    if request.args.get('secret') != secret:
        return '403 Forbidden: bad secret query parameter', 403
    post_repo(
        'statuses/' + sha,
        json={
            'state': 'success',
            'description': 'manual override: success',
            'context': CONTEXT
        },
        status_code=201
    )

    return '', 200

if __name__ == '__main__':
    app.run()
