import pytest
import aiohttp
from hailtop.auth import service_auth_headers
from hailtop.config import get_deploy_config
from hailtop.httpx import in_cluster_ssl_client_session
from hailtop.utils import request_retry_transient_errors


deploy_config = get_deploy_config()


@pytest.mark.asyncio
async def test_connect_to_address_on_pod_ip():
    async with in_cluster_ssl_client_session(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60),
            headers=service_auth_headers(deploy_config, 'address')) as session:
        await request_retry_transient_errors(
            session,
            'GET',
            f'https://address/{deploy_config.base_path("address")}/api/address')
