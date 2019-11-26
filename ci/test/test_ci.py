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
        deploy_state = None
        tries = 100
        while deploy_state is None and tries > 0:
            resp = await utils.request_retry_transient_errors(
                session, 'GET', f'{ci_deploy_status_url}', headers=headers)
            deploy_statuses = resp.json()
            assert len(deploy_statuses) == 1, deploy_statuses
            deploy_status = deploy_statuses[0]
            deploy_state = deploy_status['deploy_state']
            await asyncio.sleep(5)
            tries -= 1
        assert tries > 0
        assert deploy_state == 'success', deploy_state
