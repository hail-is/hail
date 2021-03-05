import logging
import pytest

from hailtop.config import get_deploy_config
from hailtop.auth import service_auth_headers
from hailtop.httpx import client_session
import hailtop.utils as utils

pytestmark = pytest.mark.asyncio

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

sha = 'd626f793ad700c45a878d192652a0378818bbd8b'


async def test_update_commits():
    deploy_config = get_deploy_config()
    headers = service_auth_headers(deploy_config, 'benchmark')
    commit_benchmark_url = deploy_config.url('benchmark', f'/api/v1alpha/benchmark/commit/{sha}')

    async with client_session() as session:
        await utils.request_retry_transient_errors(
            session, 'DELETE', f'{commit_benchmark_url}', headers=headers, json={'sha': sha})

        resp_status = await utils.request_retry_transient_errors(
            session, 'GET', f'{commit_benchmark_url}', headers=headers)
        commit = await resp_status.json()
        assert commit['status'] is None, commit

        resp_commit = await utils.request_retry_transient_errors(
            session, 'POST', f'{commit_benchmark_url}/update', headers=headers, json={'sha': sha})
        commit = await resp_commit.json()
        assert 'status' in commit, commit

        delay = 0.1
        while not commit['status']['complete']:
            resp = await utils.request_retry_transient_errors(
                session, 'GET', f'{commit_benchmark_url}', headers=headers)
            commit = await resp.json()
            print(commit)
            assert 'status' in commit, commit
            delay = utils.utils.sync_sleep_and_backoff(delay)
