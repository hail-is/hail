from typing import Dict
import aiohttp

from hailtop.utils import request_retry_transient_errors


async def azure_worker_access_token(session: aiohttp.ClientSession) -> Dict[str, str]:
    # https://docs.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/how-to-use-vm-token#get-a-token-using-http
    params = {
        'api-version': '2018-02-01',
        'resource': 'https://management.azure.com/'
    }
    async with await request_retry_transient_errors(
            session,
            'POST',
            'http://169.254.169.254/metadata/identity/oauth2/token',
            headers={'Metadata': 'true'},
            params=params,
            timeout=aiohttp.ClientTimeout(total=60)
    ) as resp:
        access_token = (await resp.json())['access_token']

    # https://docs.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli#az-acr-login-with---expose-token
    return {'username': '00000000-0000-0000-0000-000000000000', 'password': access_token}
