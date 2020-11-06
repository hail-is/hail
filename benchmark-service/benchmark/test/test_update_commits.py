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


async def test_update_commits():
    deploy_config = get_deploy_config()
    headers = service_auth_headers(deploy_config, 'benchmark')
    create_benchmark_url = deploy_config.url('benchmark', '/api/v1alpha/benchmark/update_commit')

    async with get_context_specific_ssl_client_session(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60)) as session:

        async def wait_forever():
            commit_info = None
            while commit_info is None:
                resp = await utils.request_retry_transient_errors(
                    session, 'POST', f'{create_benchmark_url}', headers=headers, json={'sha': sha})
                resp_text = await resp.text()
                commit_info = json.loads(resp_text)
                # batch_status = commit_info['batch_status']
                # case = commit_info['case']
            return commit_info

        commit_info = await asyncio.wait_for(wait_forever(), timeout=30 * 60)
        assert commit_info['case'] == 'has_results_file'
        assert commit_info['batch_status'] is not None, commit_info
