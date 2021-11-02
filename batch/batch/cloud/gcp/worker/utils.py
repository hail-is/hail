from typing import Dict
import aiohttp

from hailtop.utils import request_retry_transient_errors


async def gcp_worker_access_token(session: aiohttp.ClientSession) -> Dict[str, str]:
    async with await request_retry_transient_errors(
            session,
            'POST',
            'http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token',
            headers={'Metadata-Flavor': 'Google'},
            timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        access_token = (await resp.json())['access_token']
        return {'username': 'oauth2accesstoken', 'password': access_token}
