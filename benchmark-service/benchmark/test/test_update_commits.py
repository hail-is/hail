import json
import logging
import asyncio
import pytest
import aiohttp

from hailtop.config import get_deploy_config
from hailtop.auth import service_auth_headers
from hailtop.tls import in_cluster_ssl_client_session, get_context_specific_ssl_client_session
import hailtop.utils as utils

pytestmark = pytest.mark.asyncio

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

sha = 'd626f793ad700c45a878d192652a0378818bbd8b'

# What I was expecting was a delete, then a get asserting the commit status is None,
# followed by a POST to update the commit where it says the batch was submitted.
# Then wait forever asserting the batch is running until the GET returns a commit status that is complete / file exists.


async def test_update_commits():
    deploy_config = get_deploy_config()
    headers = service_auth_headers(deploy_config, 'benchmark')
    create_benchmark_url = deploy_config.url('benchmark', f'/api/v1alpha/benchmark/commit/{sha}')

    async with get_context_specific_ssl_client_session(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60)) as session:

        await utils.request_retry_transient_errors(
            session, 'DELETE', f'{create_benchmark_url}', headers=headers, json={'sha': sha})

        resp_status = await utils.request_retry_transient_errors(
            session, 'GET', f'{create_benchmark_url}', headers=headers, json={'sha': sha})
        resp_status_text = await resp_status.text()
        status = json.loads(resp_status_text)
        assert status['status'] is None, status

        resp_commit = await utils.request_retry_transient_errors(
            session, 'POST', f'{create_benchmark_url}', headers=headers, json={'sha': sha})
        resp_commit_text = await resp_commit.text()
        commit = json.loads(resp_commit_text)

        async def wait_forever():
            commit_status = None
            while commit_status is None:
                resp = await utils.request_retry_transient_errors(
                    session, 'GET', f'{create_benchmark_url}', headers=headers, json={'sha': sha})
                resp_text = await resp.text()
                commit_status = json.loads(resp_text)
            return commit_status['status']

        commit_status = await wait_forever()
        #assert commit['commit']['status'] is not None, commit
        assert commit_status == 'complete', commit_status
