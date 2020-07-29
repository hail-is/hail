import pytest
import aiohttp
from hailtop.auth import service_auth_headers
from hailtop.config import get_deploy_config
from hailtop.tls import get_in_cluster_no_hostname_checks_client_ssl_context
from hailtop.utils import request_retry_transient_errors


deploy_config = get_deploy_config()


@pytest.mark.asyncio
async def test_connect_to_address_on_pod_ip():
    async with get_in_cluster_no_hostname_checks_client_ssl_context(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60),
            headers=service_auth_headers(deploy_config, 'address'),
            check_hostname=False) as session:
        address, port = await deploy_config.address('address')
        await request_retry_transient_errors(
            session,
            'GET',
            f'https://{address}:{port}{deploy_config.base_path("address")}/api/address')
