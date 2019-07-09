import asyncio
import aiohttp

from hailtop.gear import get_deploy_config
from hailtop.gear.auth import get_tokens, set_credentials


def init_parser(parser):  # pylint: disable=unused-argument
    pass


async def async_main():
    deploy_config = get_deploy_config()

    async with aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
        set_credentials(session, 'auth')
        async with session.post(deploy_config.url('auth', '/api/v1alpha/logout')):
            pass
    tokens = get_tokens()
    auth_ns = deploy_config.service_ns('auth')
    del tokens[auth_ns]
    tokens.write()

    print('Logged out.')


def main(args):  # pylint: disable=unused-argument
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main())
