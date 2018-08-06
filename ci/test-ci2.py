import subprocess
from subprocess import call, run
import os
import time
import random
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
                time.sleep(7) # plenty of time to start a pod and run a simple command
                status = ci_get('/status', status_code=200)
                self.assertIn('prs', status)
                self.assertIn('watched_repos', status)
                prs = status['prs']
                pr_goodnesses = [(pr['source_ref'] == 'foo',
                                  pr['source_url'] == 'https://github.com/hail-is/ci-test.git',
                                  pr['target_url'] == 'https://github.com/hail-is/ci-test.git',
                                  pr['target_ref'] == 'master',
                                  pr['status']['state'] == 'success',
                                  pr['status']['job_id'] is not None,
                                  pr['status']['review_state'] == 'pending',
                                  pr['status']['source_sha'] == source_sha,
                                  pr['status']['target_sha'] == target_sha,
                                  pr['status']['pr_number'] == str(pr_number),
                                  pr['status']['docker_image'] == 'google/cloud-sdk:alpine')
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

    def test_push_while_building(self):
        BRANCH_NAME='test_push_while_building'
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

                call(['git', 'checkout', '-b', BRANCH_NAME])
                with open('hail-ci-build.sh', 'w') as f:
                    f.write('sleep 45')
                call(['git', 'add', 'hail-ci-build.sh'])
                call(['git', 'commit', '-m', 'foo'])
                call(['git', 'push', 'origin', BRANCH_NAME])
                source_sha = run(['git', 'rev-parse', BRANCH_NAME], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                first_target_sha = run(['git', 'rev-parse', 'master'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                data = post_repo(
                    'hail-is/ci-test',
                    'pulls',
                    json={ "title" : "foo", "head": BRANCH_NAME, "base": "master" },
                    status_code=201
                )
                pr_number = data['number']
                time.sleep(7) # plenty of time to start a pod
                status = ci_get('/status', status_code=200)
                self.assertIn('prs', status)
                self.assertIn('watched_repos', status)
                prs = status['prs']
                prs = [pr for pr in prs if pr['source_ref'] == BRANCH_NAME]
                self.assertEqual(len(prs), 1)
                pr = prs[0]
                self.assertEqual(pr['source_url'], 'https://github.com/hail-is/ci-test.git')
                self.assertEqual(pr['target_url'], 'https://github.com/hail-is/ci-test.git')
                self.assertEqual(pr['target_ref'], 'master')
                self.assertEqual(pr['status']['state'], 'running')
                first_job_id = pr['status']['job_id']
                self.assertTrue(first_job_id is not None)
                self.assertEqual(pr['status']['review_state'], 'pending')
                self.assertEqual(pr['status']['source_sha'], source_sha)
                self.assertEqual(pr['status']['target_sha'], first_target_sha)
                self.assertEqual(pr['status']['pr_number'], str(pr_number))
                self.assertEqual(pr['status']['docker_image'], 'google/cloud-sdk:alpine')
                call(['git', 'checkout', 'master'])
                call(['git', 'commit', '--allow-empty', '-m', 'foo'])
                call(['git', 'push', 'origin', 'master'])
                second_target_sha = run(['git', 'rev-parse', 'master'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                time.sleep(15) # plenty of time for github to notify ci and to start a new pod
                status = ci_get('/status', status_code=200)
                self.assertIn('prs', status)
                self.assertIn('watched_repos', status)
                prs = status['prs']
                prs = [pr for pr in prs if pr['source_ref'] == BRANCH_NAME]
                self.assertEqual(len(prs), 1)
                pr = prs[0]
                self.assertEqual(pr['source_url'], 'https://github.com/hail-is/ci-test.git')
                self.assertEqual(pr['target_url'], 'https://github.com/hail-is/ci-test.git')
                self.assertEqual(pr['target_ref'], 'master')
                self.assertEqual(pr['status']['state'], 'running')
                second_job_id = pr['status']['job_id']
                self.assertNotEqual(second_job_id, first_job_id)
                self.assertEqual(pr['status']['review_state'], 'pending')
                self.assertEqual(pr['status']['source_sha'], source_sha)
                self.assertEqual(pr['status']['target_sha'], second_target_sha)
                self.assertEqual(pr['status']['pr_number'], str(pr_number))
                self.assertEqual(pr['status']['docker_image'], 'google/cloud-sdk:alpine')
                time.sleep(45) # build should be done now
                status = ci_get('/status', status_code=200)
                self.assertIn('prs', status)
                self.assertIn('watched_repos', status)
                prs = status['prs']
                prs = [pr for pr in prs if pr['source_ref'] == BRANCH_NAME]
                self.assertEqual(len(prs), 1)
                pr = prs[0]
                self.assertEqual(pr['source_url'], 'https://github.com/hail-is/ci-test.git')
                self.assertEqual(pr['target_url'], 'https://github.com/hail-is/ci-test.git')
                self.assertEqual(pr['target_ref'], 'master')
                self.assertEqual(pr['status']['state'], 'success')
                self.assertEqual(second_job_id, pr['status']['job_id'])
                self.assertEqual(pr['status']['review_state'], 'pending')
                self.assertEqual(pr['status']['source_sha'], source_sha)
                self.assertEqual(pr['status']['target_sha'], second_target_sha)
                self.assertEqual(pr['status']['pr_number'], str(pr_number))
                self.assertEqual(pr['status']['docker_image'], 'google/cloud-sdk:alpine')
            finally:
                call(['git', 'push', 'origin', ':'+BRANCH_NAME])
                if pr_number is not None:
                    patch_repo(
                        'hail-is/ci-test',
                        f'pulls/{pr_number}',
                        json={ "state" : "closed" },
                        status_code=200
                    )

    def test_merges_approved_pr(self):
        BRANCH_NAME='test_merges_approved_pr'
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

                call(['git', 'checkout', '-b', BRANCH_NAME])
                call(['git', 'commit', '--allow-empty', '-m', 'foo'])
                call(['git', 'push', 'origin', BRANCH_NAME])
                source_sha = run(['git', 'rev-parse', BRANCH_NAME], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                target_sha = run(['git', 'rev-parse', 'master'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
                pr = post_repo(
                    'hail-is/ci-test',
                    'pulls',
                    json={ "title" : "foo", "head": BRANCH_NAME, "base": "master" },
                    status_code=201
                )
                post_repo(
                    'hail-is/ci-test',
                    f'pulls/{pr["number"]}/reviews',
                    json={ "commit_id": source_sha, "event": "APPROVE" },
                    status_code=200
                )
                time.sleep(10) # enough time to run the test pod
                prs = ci_get('/status', status_code=200)['prs']
                prs = [pr for pr in prs if pr['source_ref'] == BRANCH_NAME]
                assert len(prs) == 0
                pr = prs[0]
                assert pr.items() >= {
                    'source_url': 'https://github.com/hail-is/ci-test.git',
                    'target_url': 'https://github.com/hail-is/ci-test.git',
                    'target_ref': 'master',
                    'status': {
                        'state': 'success',
                        'review_state': 'approved',
                        'source_sha': source_sha,
                        'target_sha': target_sha,
                        'pr_number': str(data['pr_number'])
                    }
                }.items()
            finally:
                call(['git', 'push', 'origin', ':'+BRANCH_NAME])
                if pr_number is not None:
                    patch_repo(
                        'hail-is/ci-test',
                        f'pulls/{pr_number}',
                        json={ "state" : "closed" },
                        status_code=200
                    )
