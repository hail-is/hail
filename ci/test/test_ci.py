import pytest
import aiohttp
import asyncio

from hailtop.config import get_deploy_config
from hailtop.auth import service_auth_headers
import hailtop.utils as utils

pytestmark = pytest.mark.asyncio


async def test_deploy():
    deploy_config = get_deploy_config()
    ci_deploy_status_url = deploy_config.url('ci', '/api/v1alpha/deploy_status')
    headers = service_auth_headers(deploy_config, 'ci')
    async with aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60)) as session:

        async def wait_forever():
            deploy_state = None
            while deploy_state is None:
                resp = await utils.request_retry_transient_errors(
                    session, 'GET', f'{ci_deploy_status_url}', headers=headers)
                deploy_statuses = await resp.json()
                assert len(deploy_statuses) == 1, deploy_statuses
                deploy_status = deploy_statuses[0]
                deploy_state = deploy_status['deploy_state']
                await asyncio.sleep(5)
            return deploy_state

        deploy_state = await asyncio.wait_for(wait_forever(), timeout=20 * 60)
        assert deploy_state == 'success', deploy_state
