import ssl
import pytest
import aiohttp
from hailtop.auth import service_auth_headers
from hailtop.config import get_deploy_config
from hailtop.tls import _get_ssl_config
from hailtop.utils import retry_transient_errors


deploy_config = get_deploy_config()


@pytest.mark.asyncio
async def test_connect_to_address_on_pod_ip():
    ssl_config = _get_ssl_config()
    client_ssl_context = ssl.create_default_context(
        purpose=ssl.Purpose.SERVER_AUTH, cafile=ssl_config['outgoing_trust']
    )
    client_ssl_context.load_default_certs()
    client_ssl_context.load_cert_chain(ssl_config['cert'], keyfile=ssl_config['key'], password=None)
    client_ssl_context.verify_mode = ssl.CERT_REQUIRED
    client_ssl_context.check_hostname = False

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=client_ssl_context),
        raise_for_status=True,
        timeout=aiohttp.ClientTimeout(total=5),
        headers=service_auth_headers(deploy_config, 'address'),
    ) as session:

        async def get():
            address, port = await deploy_config.address('address')
            session.get(f'https://{address}:{port}{deploy_config.base_path("address")}/api/address')

        await retry_transient_errors(get)
