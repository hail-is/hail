import subprocess
from subprocess import call, run
import os
import time
import requests
import tempfile
import unittest

CI_URL='http://localhost:5000'
GITHUB_URL = 'https://api.github.com/'

class NoOAuthToken(Exception):
    pass
class BadStatus(Exception):
    def __init__(self, data, status_code):
        Exception.__init__(self, str(data))
        self.data = data
        self.status_code = status_code

try:
    with open('oauth-token/oauth-token', 'r') as f:
        oauth_token = f.read()
except FileNotFoundError as e:
    raise NoOAuthToken(
        "working directory must contain `oauth-token/oauth-token' "
        "containing a valid GitHub oauth token"
    ) from e

def ci_post(endpoint, json=None, status_code=None, json_response=True):
    r = requests.post(CI_URL + endpoint, json = json)
    if status_code and r.status_code != status_code:
        raise ValueError(
            'bad status_code from pull_request: {status_code}\n{message}'.format(
                endpoint=endpoint, status_code=r.status_code, message=r.text))
    if json_response:
        return r.json()
    else:
        return r.text

def ci_get(endpoint, status_code=None, json_response=True):
    r = requests.get(CI_URL + endpoint)
    if status_code and r.status_code != status_code:
        raise ValueError(
            'bad status_code from {endpoint}: {status_code}\n{message}'.format(
                endpoint=endpoint, status_code=r.status_code, message=r.text))
    if json_response:
        return r.json()
    else:
        return r.text

def post_repo(repo, url, headers=None, json=None, data=None, status_code=None):
    return modify_repo('post', repo, url, headers, json, data, status_code)

def patch_repo(repo, url, headers=None, json=None, data=None, status_code=None):
    return modify_repo('patch', repo, url, headers, json, data, status_code)

def modify_repo(verb, repo, url, headers=None, json=None, data=None, status_code=None):
    if headers is None:
        headers = {}
    if 'Authorization' in headers:
        raise ValueError(
            'Header already has Authorization? ' + str(headers))
    headers['Authorization'] = 'token ' + oauth_token
    if verb == 'post':
        r = requests.post(
            f'{GITHUB_URL}repos/{repo}/{url}',
            headers=headers,
            json=json,
            data=data
        )
    else:
        assert verb == 'patch', verb
        r = requests.patch(
            f'{GITHUB_URL}repos/{repo}/{url}',
            headers=headers,
            json=json,
            data=data
        )
    if status_code and r.status_code != status_code:
        raise BadStatus({
            'method': verb,
            'endpoint' : f'{GITHUB_URL}repos/{repo}/{url}',
            'status_code' : r.status_code,
            'data': data,
            'json': json,
            'message': 'github error',
            'github_json': r.json()
        }, r.status_code)
    else:
        return r.json()

def get_repo(repo, url, headers=None, status_code=None):
    return get_github(f'repos/{repo}/{url}', headers, status_code)

def get_github(url, headers=None, status_code=None):
    if headers is None:
        headers = {}
    if 'Authorization' in headers:
        raise ValueError(
            'Header already has Authorization? ' + str(headers))
    headers['Authorization'] = 'token ' + oauth_token
    r = requests.get(
        f'{GITHUB_URL}{url}',
        headers=headers
    )
    if status_code and r.status_code != status_code:
        raise BadStatus({
            'method': 'get',
            'endpoint' : f'{GITHUB_URL}{url}',
            'status_code' : r.status_code,
            'message': 'github error',
            'github_json': r.json()
        }, r.status_code)
    else:
        return r.json()

###############################################################################

class TestCI(unittest.TestCase):
    def test_pull_request_trigger(self):
        with tempfile.TemporaryDirectory() as d:
            pr_number = None
            try:
                status = ci_get('/status', status_code=200)
                self.assertIn('watched_repos', status)
                self.assertEqual(status['watched_repos'], ['hail-is/ci-test'])
                os.chdir(d)
                call(['git', 'clone', 'git@github.com:hail-is/ci-test.git'])
                os.chdir('ci-test')
                call(['git', 'remote', '-v'])

                call(['git', 'checkout', '-b', 'foo'])
                call(['touch', 'foo'])
                call(['git', 'add', 'foo'])
                call(['git', 'commit', '-m', 'foo'])
                call(['git', 'push', 'origin', 'foo'])
                source_sha = run(['git', 'rev-parse', 'foo'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                target_sha = run(['git', 'rev-parse', 'master'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                data = post_repo(
                    'hail-is/ci-test',
                    'pulls',
                    json={ "title" : "foo", "head": "foo", "base": "master" },
                    status_code=201
                )
                pr_number = data['number']
                time.sleep(5)
                status = ci_get('/status', status_code=200)
                self.assertIn('prs', status)
                self.assertIn('watched_repos', status)
                prs = status['prs']
                pr_goodnesses = [(pr['source_ref'] == 'foo',
                                  pr['source_url'] == 'https://github.com/hail-is/ci-test.git',
                                  pr['target_url'] == 'https://github.com/hail-is/ci-test.git',
                                  pr['target_ref'] == 'master',
                                  (pr['status']['state'] == 'pending' and pr['status']['job_id'] is None or
                                   pr['status']['state'] != 'pending' and pr['status']['job_id'] is not None),
                                  pr['status']['review_state'] == 'pending',
                                  pr['status']['source_sha'] == source_sha,
                                  pr['status']['target_sha'] == target_sha,
                                  pr['status']['pr_number'] == str(pr_number),
                                  pr['status']['docker_image'] == 'gcr.io/broad-ctsa/alpine-bash:latest')
                                 for pr in prs]
                self.assertEqual(
                    [all(x) for x in pr_goodnesses].count(True),
                    1, f'expected a pr to have: "source_sha": "{source_sha}", "target_sha": "{target_sha}", "pr_number": "{pr_number}", actual prs and goodnesses: {list(zip(prs, pr_goodnesses))}')
            finally:
                call(['git', 'push', 'origin', ':foo'])
                if pr_number is not None:
                    patch_repo(
                        'hail-is/ci-test',
                        f'pulls/{pr_number}',
                        json={ "state" : "closed" },
                        status_code=200
                    )



