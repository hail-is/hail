import os
import sys
import pytest
from gidgethub import aiohttp as gh_aiohttp
import aiohttp
import subprocess as sp
import asyncio

from hailtop.config import get_deploy_config

pytestmark = pytest.mark.asyncio

deploy_config = get_deploy_config()

org = os.environ['ORGANIZATION']
repo = os.environ['REPO_NAME']
namespace = os.environ['NAMESPACE']

with open('/secret/ci-secrets/user1', 'r') as f:
    user1_token = f.read()

with open('/secret/ci-secrets/user2', 'r') as f:
    user2_token = f.read()


def wait_for_hello():
    wait_cmd = f'python3 wait-for.py 900 {namespace} Service --location k8s hello'
    result = sp.run(wait_cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    if result.returncode != 0:
        raise Exception(f'hello service was not deployed: {result!r}')

    url = deploy_config.url('hello', f'/sha')
    sha = sp.check_output(f"curl {url}", shell=True)
    return sha


async def wait_for_redeployment(old_sha):
    wait_interval = 10
    elapsed_time = 0
    while elapsed_time < 300:
        try:
            new_sha = wait_for_hello()
            if new_sha != old_sha:
                print('hello was redeployed', file=sys.stderr)
                return
            elapsed_time += wait_interval
        except Exception:
            pass
        await asyncio.sleep(wait_interval)
    raise Exception(f'hello service was not redeployed in 300 seconds')


@pytest.fixture
async def gh_client1():
    session = aiohttp.ClientSession(
        raise_for_status=True,
        timeout=aiohttp.ClientTimeout(total=60))

    gh_client = gh_aiohttp.GitHubAPI(session, 'test-ci-1', oauth_token=user1_token)
    yield gh_client
    await session.close()


@pytest.fixture
async def gh_client2():
    session = aiohttp.ClientSession(
        raise_for_status=True,
        timeout=aiohttp.ClientTimeout(total=60))

    gh_client = gh_aiohttp.GitHubAPI(session, 'test-ci-2', oauth_token=user2_token)
    yield gh_client
    await session.close()


async def test_deploy():
    wait_for_hello()


# FIXME: This test requires either putting user1 as an authorized user
#        or having a fake developer who can authorize the sha
# async def test_pr_merge(gh_client1, gh_client2):
#     sha = wait_for_hello()
#
#     script = f'''
# git clone https://{user1_token}@github.com/{org}/{repo}.git
# cd {repo}
# git config --global user.email ci@hail.is
# git config --global user.name ci
# git checkout -b benign-changes
# echo "hello" > hello.txt
# git add hello.txt && git commit -m "add hello.txt"
# git push --set-upstream origin benign-changes
# '''
#     sp.run(script, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
#
#     data = {
#         'title': 'Benign changes',
#         'head': 'benign-changes',
#         'base': 'master'
#     }
#
#     result = await gh_client1.post(f'/repos/{org}/{repo}/pulls',
#                                    data=data)
#     pull_number = result['number']
#
#     await gh_client2.post(f'/repos/{org}/{repo}/pulls/{pull_number}/reviews',
#                           data={'event': 'APPROVE'})
#
#     await wait_for_redeployment(sha)
