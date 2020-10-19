import pytest
import aiohttp
from hailtop.auth import service_auth_headers
from hailtop.config import get_deploy_config
from hailtop import httpx


deploy_config = get_deploy_config()


@pytest.mark.asyncio
async def test_connect_to_address_on_pod_ip():
    async with httpx.client_session(
            headers=service_auth_headers(deploy_config, 'address'),
            timeout=aiohttp.ClientTimeout(total=60)) as session:
        await session.get(deploy_config.url('address', '/api/address'))
