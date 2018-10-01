from http_helper import get_repo, post_repo, patch_repo
from environment import INSTANCE_ID
from pr import PR
from subprocess import call, run
import inspect
import json
import os
import pkg_resources
import requests
import subprocess
import tempfile
import time
import unittest

CI_URL = 'http://localhost:5000'
GITHUB_URL = 'https://api.github.com/'


class NoOAuthToken(Exception):
    pass


class BadStatus(Exception):
    def __init__(self, data, status_code):
        Exception.__init__(self, str(data))
        self.data = data
        self.status_code = status_code


def read_oauth_token_or_fail(path):
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError as e:
        raise NoOAuthToken(f"working directory must contain `{path}' "
                           "containing a valid GitHub oauth token") from e


oauth_tokens = {}
oauth_tokens['user1'] = read_oauth_token_or_fail('github-tokens/user1')
oauth_tokens['user2'] = read_oauth_token_or_fail('github-tokens/user2')


def ci_get(endpoint, status_code=None, json_response=True):
    r = requests.get(CI_URL + endpoint)
    if status_code and r.status_code != status_code:
        raise ValueError(
            'bad status_code from {endpoint}: {status_code}\n{message}'.format(
                endpoint=endpoint,
                status_code=r.status_code,
                message=r.text))
    if json_response:
        return r.json()
    else:
        return r.text


###############################################################################


class Match(object):
    def __init__(self, pred, msg):
        self.pred = pred
        self.msg = msg

    def matches(self, x):
        return self.pred(x)

    def __str__(self):
        return self.msg

    def any(*args):
        s = set(args)
        return Match(lambda x: x in s, f'any{args}')

    def notEqual(x):
        return Match(lambda y: x != y, f'not equal to {x}')


def assertDictHasKVs(actual, kvs):
    d = dictKVMismatches(actual, kvs)
    assert len(d) == 0, f'{d}, actual: {actual}'


def dictKVMismatches(actual, kvs):
    assert isinstance(actual, dict), actual
    errors = {}
    for k, v in kvs.items():
        if k not in actual:
            errors[k] = f'had nothing, should have had {v}'
        elif isinstance(v, dict):
            sub_errors = dictKVMismatches(actual[k], v)
            if len(sub_errors) != 0:
                errors[k] = sub_errors
        else:
            actual_v = actual[k]
            if isinstance(v, Match):
                if not v.matches(actual_v):
                    errors[k] = f'{actual_v} does not match {v}'
            elif actual_v != v:
                errors[k] = f'{actual_v} != {v}'
    return errors


###############################################################################


REPO_NAME = os.environ['REPO_NAME']
FQ_REPO = 'hail-ci-test/' + os.environ['REPO_NAME']
DELAY_IN_SECONDS = 15
MAX_POLLS = 30


def test_pull_request_comment_does_not_overwrite_approval():
    data = pkg_resources.resource_string(
        __name__,
        'comment-after-approve-pull-request-review.json')
    reviews = json.loads(data)

    from github import overall_review_state

    review_state = overall_review_state(reviews)

    assert review_state['state'] == 'approved'
    assert review_state['reviews']['cseed'] == 'APPROVED'
    assert 'tpoterba' not in review_state['reviews']


def get_pr(source_ref):
    status = ci_get('/status', status_code=200)
    assert 'prs' in status
    assert '_watched_targets' in status
    all_prs = [PR.from_json(x) for x in status['prs']]
    prs = [pr for pr in all_prs if pr.source.ref.name == source_ref]
    assert len(prs) == 1, [str(x.source.ref) for x in all_prs]
    return prs[0]


def poll_github_until_merged(pr_number,
                             delay_in_seconds=DELAY_IN_SECONDS,
                             max_polls=MAX_POLLS):
    r, status_code = get_repo(
        FQ_REPO,
        f'pulls/{pr_number}/merge',
        # 204 NO CONTENT means merged, 404 means not merged
        status_code=[204, 404],
        json_response=False,
        token=oauth_tokens['user1'])
    polls = 0
    while status_code != 204 and polls < max_polls:
        time.sleep(delay_in_seconds)
        polls = polls + 1
        _, status_code = get_repo(
            FQ_REPO,
            f'pulls/{pr_number}/merge',
            # 204 NO CONTENT means merged, 404 means not merged
            status_code=[204, 404],
            json_response=False,
            token=oauth_tokens['user1'])
    assert polls < max_polls, f'{polls} {max_polls}'
    assert status_code == 204


def poll_until_finished_pr(source_ref,
                           delay_in_seconds=DELAY_IN_SECONDS,
                           max_polls=MAX_POLLS):
    return poll_pr(
        source_ref,
        lambda pr: pr.is_running() or pr.is_pending_build(),
        delay_in_seconds=delay_in_seconds,
        max_polls=max_polls)


def poll_until_running_pr(source_ref,
                          delay_in_seconds=DELAY_IN_SECONDS,
                          max_polls=MAX_POLLS):
    return poll_pr(
        source_ref,
        lambda pr: pr.is_pending_build(),
        delay_in_seconds=delay_in_seconds,
        max_polls=max_polls)


def poll_pr(source_ref,
            poll_until_false,
            delay_in_seconds=DELAY_IN_SECONDS,
            max_polls=MAX_POLLS):
    pr = get_pr(source_ref)
    polls = 0
    while poll_until_false(pr):
        assert polls < max_polls
        time.sleep(delay_in_seconds)
        pr = get_pr(source_ref)
        polls = polls + 1
    return pr


def poll_until_pr_exists_and(source_ref,
                             poll_until_true,
                             delay_in_seconds=DELAY_IN_SECONDS,
                             max_polls=MAX_POLLS):
    prs = []
    polls = 0
    while (len(prs) == 0
           or not poll_until_true(prs[0])) and polls < max_polls:
        time.sleep(delay_in_seconds)
        status = ci_get('/status', status_code=200)
        assert 'prs' in status
        assert '_watched_targets' in status
        all_prs = [PR.from_json(x) for x in status['prs']]
        prs = [pr for pr in all_prs if pr.source.ref.name == source_ref]
        assert len(prs) <= 1, [str(x.source.ref) for x in all_prs]
        polls = polls + 1
    assert len(prs) == 1
    return prs[0]


def test_pull_request_trigger(tmpdir):
    BRANCH_NAME = 'test_pull_request_trigger'
    pr_number = None
    try:
        status = ci_get('/status', status_code=200)
        assert '_watched_targets' in status
        assert status['_watched_targets'] == [[{
            'repo': {
                'name': REPO_NAME,
                'owner': 'hail-ci-test'},
            'name': 'master'}, True]]
        os.chdir(tmpdir)
        call(['git', 'clone', f'https://{oauth_tokens["user1"]}@github.com/hail-ci-test/{REPO_NAME}.git'])
        os.chdir(REPO_NAME)
        call(['git', 'config', 'user.email', 'ci-automated-tests@broadinstitute.org'])
        call(['git', 'config', 'user.name', 'ci-automated-tests'])
        call(['git', 'remote', '-v'])

        call(['git', 'checkout', '-b', BRANCH_NAME])
        call(['git', 'commit', '--allow-empty', '-m', 'foo'])
        call(['git', 'push', 'origin', BRANCH_NAME])
        source_sha = rev_parse(BRANCH_NAME)
        target_sha = rev_parse('master')
        data = post_repo(
            FQ_REPO,
            'pulls',
            json={
                "title": "foo",
                "head": BRANCH_NAME,
                "base": "master"
            },
            status_code=201,
            token=oauth_tokens['user1'])
        pr_number = str(data['number'])
        time.sleep(7)
        pr = poll_until_finished_pr(BRANCH_NAME)
        assertDictHasKVs(
            pr.to_json(),
            {
                "target": {
                    "ref": {
                        "repo": {
                            "owner": "hail-ci-test",
                            "name": REPO_NAME
                        },
                        "name": "master"
                    },
                    "sha": target_sha
                },
                "source": {
                    "ref": {
                        "repo": {
                            "owner": "hail-ci-test",
                            "name": REPO_NAME
                        },
                        "name": BRANCH_NAME
                    },
                    "sha": source_sha
                },
                "review": "pending",
                "build": {
                    "type": "Mergeable",
                    "target_sha": target_sha
                },
                "number": pr_number
            })
    finally:
        call(['git', 'push', 'origin', ':' + BRANCH_NAME])
        if pr_number is not None:
            patch_repo(
                FQ_REPO,
                f'pulls/{pr_number}',
                json={"state": "closed"},
                status_code=200,
                token=oauth_tokens['user1'])


def create_pull_request(title, ref, base="master"):
    return post_repo(
        FQ_REPO,
        'pulls',
        json={
            "title": title,
            "head": ref,
            "base": base
        },
        status_code=201,
        token=oauth_tokens['user1'])


def create_and_push_empty_commit(source_ref, target_ref='master'):
    call(['git', 'checkout', target_ref])
    call(['git', 'checkout', '-b', source_ref])
    return push_empty_commit(source_ref)


def push_empty_commit(ref):
    call(['git', 'checkout', ref])
    call(['git', 'commit', '--allow-empty', '-m' 'foo'])
    return push(ref)


def push(ref):
    call(['git', 'push', 'origin', ref])
    return rev_parse(ref)


def approve(pr_number, sha):
    return post_repo(
        FQ_REPO,
        f'pulls/{pr_number}/reviews',
        json={
            "commit_id": sha,
            "event": "APPROVE"
        },
        status_code=200,
        token=oauth_tokens['user2'])


def rev_parse(ref):
    return run(
        ['git', 'rev-parse', ref],
        stdout=subprocess.PIPE).stdout.decode('utf-8').strip()


def test_push_while_building(tmpdir):
    BRANCH_NAME = 'test_push_while_building'
    SLOW_BRANCH_NAME = 'test_push_while_building_slow'
    pr_number = {}
    source_sha = {}
    gh_pr = {}
    pr = {}
    try:
        status = ci_get('/status', status_code=200)
        assert '_watched_targets' in status
        assert status['_watched_targets'] == [[{
            'repo': {
                'name': REPO_NAME,
                'owner': 'hail-ci-test'},
            'name': 'master'}, True]]
        os.chdir(tmpdir)

        call(['git', 'clone', f'https://{oauth_tokens["user1"]}@github.com/hail-ci-test/{REPO_NAME}.git'])
        os.chdir(REPO_NAME)
        call(['git', 'config', 'user.email', 'ci-automated-tests@broadinstitute.org'])
        call(['git', 'config', 'user.name', 'ci-automated-tests'])
        call(['git', 'remote', '-v'])

        # start slow branch
        call(['git', 'checkout', 'master'])
        first_target_sha = rev_parse('master')
        call(['git', 'checkout', '-b', SLOW_BRANCH_NAME])
        with open('hail-ci-build.sh', 'w') as f:
            f.write('sleep 45')
        call(['git', 'add', 'hail-ci-build.sh'])
        call(['git', 'commit', '-m', 'foo'])
        source_sha[SLOW_BRANCH_NAME] = push(SLOW_BRANCH_NAME)
        gh_pr[SLOW_BRANCH_NAME] = create_pull_request(
            'foo',
            SLOW_BRANCH_NAME)
        pr_number[SLOW_BRANCH_NAME] = gh_pr[SLOW_BRANCH_NAME]['number']

        # get details on first job of slow branch
        pr[SLOW_BRANCH_NAME] = poll_until_pr_exists_and(
            SLOW_BRANCH_NAME,
            lambda x: x.is_running())
        assertDictHasKVs(
            pr[SLOW_BRANCH_NAME].to_json(),
            {
                "target": {
                    "ref": {
                        "repo": {
                            "owner": "hail-ci-test",
                            "name": REPO_NAME
                        },
                        "name": "master"
                    },
                    "sha": first_target_sha
                },
                "source": {
                    "ref": {
                        "repo": {
                            "owner": "hail-ci-test",
                            "name": REPO_NAME
                        },
                        "name": SLOW_BRANCH_NAME
                    },
                    "sha": source_sha[SLOW_BRANCH_NAME]
                },
                "review": "pending",
                "build": {
                    "type": "Building",
                    "target_sha": first_target_sha
                },
                "number": str(pr_number[SLOW_BRANCH_NAME])
            })
        first_slow_job_id = pr[SLOW_BRANCH_NAME].build.job.id
        assert first_slow_job_id is not None

        # start fast branch
        source_sha[BRANCH_NAME] = create_and_push_empty_commit(
            BRANCH_NAME)
        gh_pr[BRANCH_NAME] = create_pull_request(
            'foo',
            BRANCH_NAME)
        pr_number[BRANCH_NAME] = gh_pr[BRANCH_NAME]['number']
        approve(pr_number[BRANCH_NAME], source_sha[BRANCH_NAME])

        poll_github_until_merged(pr_number[BRANCH_NAME])

        call(['git', 'fetch', 'origin'])
        second_target_sha = rev_parse('origin/master')

        time.sleep(10)  # allow deploy job to run

        deploy_artifact = run(['gsutil', 'cat', f'gs://hail-ci-test/{second_target_sha}'], stdout=subprocess.PIPE)
        deploy_artifact = deploy_artifact.stdout.decode('utf-8').strip()
        assert f'commit {second_target_sha}' in deploy_artifact

        time.sleep(5)  # allow github push notification to be sent

        pr[SLOW_BRANCH_NAME] = poll_until_finished_pr(
            SLOW_BRANCH_NAME)
        assertDictHasKVs(
            pr[SLOW_BRANCH_NAME].to_json(),
            {
                "target": {
                    "ref": {
                        "repo": {
                            "owner": "hail-ci-test",
                            "name": REPO_NAME
                        },
                        "name": "master"
                    },
                    "sha": second_target_sha
                },
                "source": {
                    "ref": {
                        "repo": {
                            "owner": "hail-ci-test",
                            "name": REPO_NAME
                        },
                        "name": SLOW_BRANCH_NAME
                    },
                    "sha": source_sha[SLOW_BRANCH_NAME]
                },
                "review": "pending",
                "build": {
                    "type": "Mergeable",
                    "target_sha": second_target_sha
                },
                "number": str(pr_number[SLOW_BRANCH_NAME])
            })
    finally:
        call(['git', 'push', 'origin', ':' + SLOW_BRANCH_NAME])
        call(['git', 'push', 'origin', ':' + BRANCH_NAME])
        for pr_number in pr_number.values():
            patch_repo(
                FQ_REPO,
                f'pulls/{pr_number}',
                json={"state": "closed"},
                status_code=200,
                token=oauth_tokens['user1'])


def test_merges_approved_pr(tmpdir):
    BRANCH_NAME = 'test_merges_approved_pr'
    pr_number = None
    try:
        status = ci_get('/status', status_code=200)
        assert '_watched_targets' in status
        assert status['_watched_targets'] == [[{
            'repo': {
                'name': REPO_NAME,
                'owner': 'hail-ci-test'},
            'name': 'master'}, True]]
        os.chdir(tmpdir)
        call(['git', 'clone', f'https://{oauth_tokens["user1"]}@github.com/hail-ci-test/{REPO_NAME}.git'])
        os.chdir(REPO_NAME)
        call(['git', 'config', 'user.email', 'ci-automated-tests@broadinstitute.org'])
        call(['git', 'config', 'user.name', 'ci-automated-tests'])
        call(['git', 'remote', '-v'])

        call(['git', 'checkout', '-b', BRANCH_NAME])
        call(['git', 'commit', '--allow-empty', '-m', 'foo'])
        call(['git', 'push', 'origin', BRANCH_NAME])
        source_sha = rev_parse(BRANCH_NAME)
        gh_pr = post_repo(
            FQ_REPO,
            'pulls',
            json={
                "title": "foo",
                "head": BRANCH_NAME,
                "base": "master"
            },
            status_code=201)
        pr_number = str(gh_pr['number'])
        post_repo(
            FQ_REPO,
            f'pulls/{pr_number}/reviews',
            json={
                "commit_id": source_sha,
                "event": "APPROVE"
            },
            status_code=200,
            token=oauth_tokens['user2'])
        get_repo(
            FQ_REPO,
            f'pulls/{pr_number}/reviews',
            status_code=200,
            token=oauth_tokens['user1'])
        time.sleep(7)
        poll_github_until_merged(pr_number)

        call(['git', 'fetch', 'origin'])
        merged_sha = rev_parse('origin/master')

        # deploy job takes some time
        time.sleep(7)

        deploy_artifact = run(['gsutil', 'cat', f'gs://hail-ci-test/{merged_sha}'], stdout=subprocess.PIPE, check=True)
        deploy_artifact = deploy_artifact.stdout.decode('utf-8').strip()
        assert f'commit {merged_sha}' in deploy_artifact
    finally:
        call(['git', 'push', 'origin', ':' + BRANCH_NAME])
        if pr_number is not None:
            patch_repo(
                FQ_REPO,
                f'pulls/{pr_number}',
                json={"state": "closed"},
                status_code=200)
