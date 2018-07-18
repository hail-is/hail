from flask import Flask, request, jsonify
import batch
import requests

repo = 'danking/docker-build-test/'
repo_api_url = 'https://api.github.com/repos/' + repo

class NoOAuthToken(Exception):
    pass
class NoSecret(Exception):
    pass
class BadStatus(Exception):
    def __init__(self, data, status_code):
        Exception.__init__(self)
        self.data = data
        self.status_code = status_code

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

###############################################################################
### Global State & Setup

prs = {}

def post_repo(url, headers=None, json=None, data=None, status_code=None):
    if headers is None:
        headers = {}
    if 'Authorization' in headers:
        raise ValueError(
            'Header already has Authorization? ' + str(headers))
    headers['Authorization'] = 'token ' + oauth_token
    r = requests.post(
        repo_api_url + url,
        headers=headers,
        json=json,
        data=data
    )
    if status_code and r.status_code != status_code:
        raise BadStatus({
            'method': 'post',
            'endpoint' : repo_api_url + url,
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
        repo_api_url + url,
        headers=headers
    )
    if status_code and r.status_code != status_code:
        raise BadStatus({
            'method': 'get',
            'endpoint' : repo_api_url + url,
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

@app.route('/github', methods=['POST'])
def github():
    print(request.data)
    return '', 200

@app.route('/status')
def status():
    return jsonify(prs), 200

@app.route('/pr/<pr_number>/retest')
def retest(pr_number):
    return pr_number, 200

@app.route('/pr/<pr_number>/review_status')
def review_status(pr_number):
    reviews = requests.get(
        repo_api_url + 'pulls/' + pr_number + '/reviews'
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
            return jsonify({
                'state': 'CHANGES_REQUESTED',
                'reviews': latest_state_by_login
            }), 200
        elif (state == 'APPROVED'):
            at_least_one_approved = True

    if at_least_one_approved:
        return jsonify({
            'state': 'APPROVED',
            'reviews': latest_state_by_login
        }), 200
    else:
        return jsonify({
            'state': 'PENDING',
            'reviews': latest_state_by_login
        }), 200

@app.route('/pr/<sha>/fail')
def fail(sha):
    if request.args.get('secret') != secret:
        return '403 Forbidden: bad secret query parameter', 403
    post_repo(
        'statuses/' + sha,
        json={
            'state': 'failure',
            'description': 'manual override: fail',
            'context': 'hail-ci'
        },
        status_code=201
    )

    return 'success', 200

@app.route('/pr/<sha>/pending')
def pending(sha):
    if request.args.get('secret') != secret:
        return '403 Forbidden: bad secret query parameter', 403
    post_repo(
        'statuses/' + sha,
        json={
            'state': 'pending',
            'description': 'manual override: pending',
            'context': 'hail-ci'
        },
        status_code=201
    )

    return 'success', 200

@app.route('/pr/<sha>/success')
def success(sha):
    if request.args.get('secret') != secret:
        return '403 Forbidden: bad secret query parameter', 403
    post_repo(
        'statuses/' + sha,
        json={
            'state': 'success',
            'description': 'manual override: success',
            'context': 'hail-ci'
        },
        status_code=201
    )

    return 'success', 200

@app.route('/pr/<sha>/statuses')
def statuses(sha):
    json = get_repo(
        'commits/' + sha + '/statuses',
        status_code=200
    )
    return jsonify(json), 200

if __name__ == '__main__':
    app.run()
