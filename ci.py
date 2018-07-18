from flask import Flask, request, jsonify
import batch
import requests

repo = 'hail-is/hail/'
repo_api_url = 'https://api.github.com/repos/' + repo

class NoOAuthToken(Exception):
    pass

app = Flask(__name__)

try:
    with open('oauth-token', 'r') as f:
        oauth_token = f.read()
except FileNotFoundError as e:
    raise NoOAuthToken(
        "working directory must contain a file called `oauth-token' "
        "containing a valid GitHub oauth token"
    ) from e

prs = {}

@app.route('/')
def hello_world():
    return 'Hello Bhavana!'

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
    r = requests.get(
        repo_api_url + '/pulls/' + pr_number + '/reviews',
        headers={ 'access_token' : oauth_token }
    )
    if r.status_code != 200:
        return jsonify({
            'message': 'github error',
            'github_json': r.json()
        }), r.status_code

    reviews = r.json()
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

if __name__ == '__main__':
    app.run()
