import os
from typing import Dict
import aiohttp

from hailtop.utils import request_retry_transient_errors, time_msecs

acr_refresh_token = None
expiration_time = None


async def get_aad_access_token(session: aiohttp.ClientSession) -> str:
    # https://docs.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/how-to-use-vm-token#get-a-token-using-http
    params = {
        'api-version': '2018-02-01',
        'resource': 'https://management.azure.com/'
    }
    async with await request_retry_transient_errors(
            session,
            'GET',
            'http://169.254.169.254/metadata/identity/oauth2/token',
            headers={'Metadata': 'true'},
            params=params,
            timeout=aiohttp.ClientTimeout(total=60)
    ) as resp:
        access_token = (await resp.json())['access_token']
        return access_token


async def get_acr_refresh_token(session: aiohttp.ClientSession, acr_url: str, aad_access_token: str) -> str:
    # https://github.com/Azure/acr/blob/main/docs/AAD-OAuth.md#calling-post-oauth2exchange-to-get-an-acr-refresh-token
    data = {
        'grant_type': 'access_token',
        'service': acr_url,
        'access_token': aad_access_token
    }

    async with await request_retry_transient_errors(
            session,
            'POST',
            f'https://{acr_url}/oauth2/exchange',
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data=data,
            timeout=aiohttp.ClientTimeout(total=60)
    ) as resp:
        refresh_token = (await resp.json())['refresh_token']
        return refresh_token


async def azure_worker_access_token(session: aiohttp.ClientSession) -> Dict[str, str]:
    global acr_refresh_token, expiration_time
    if acr_refresh_token is None or time_msecs() >= expiration_time:
        acr_url = os.environ['DOCKER_PREFIX']
        assert acr_url.endswith('azurecr.io'), acr_url
        aad_access_token = await get_aad_access_token(session)
        acr_refresh_token = await get_acr_refresh_token(session, acr_url, aad_access_token)
        expiration_time = time_msecs() + 60 * 60 * 1000  # token expires in 3 hours so we refresh after 1 hour
    return {'username': '00000000-0000-0000-0000-000000000000', 'password': acr_refresh_token}
