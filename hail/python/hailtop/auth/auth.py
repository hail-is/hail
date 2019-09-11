import aiohttp
from hailtop.config import get_deploy_config
from hailtop.utils import async_to_blocking

from .tokens import get_tokens


async def async_get_userinfo():
    deploy_config = get_deploy_config()
    headers = auth_headers('auth')
    async with aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
        async with session.get(
                deploy_config.url('auth', '/api/v1alpha/userinfo'), headers=headers) as resp:
            return await resp.json()


def get_userinfo():
    return async_to_blocking(async_get_userinfo())


def auth_headers(service, authorize_target=True):
    deploy_config = get_deploy_config()
    tokens = get_tokens()
    ns = deploy_config.service_ns(service)
    headers = {}
    if authorize_target:
        headers['Authorization'] = f'Bearer {tokens[ns]}'
    if deploy_config.location() == 'external' and ns != 'default':
        headers['X-Hail-Internal-Authorization'] = f'Bearer {tokens["default"]}'
    return headers
