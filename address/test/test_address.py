import pytest
import aiohttp
from hailtop.config import get_deploy_config
from hailtop.tls import ssl_client_session
from hailtop.utils import request_retry_transient_errors

@pytest.mark.asyncio
async def test_connect_to_address_on_pod_ip():
    async with ssl_client_session(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
        address = await get_deploy_config().address('address')
        await request_retry_transient_errors(session, 'GET', f'https://{address}/')
