import asyncio

from hailtop.config import get_deploy_config
from hailtop.auth import get_tokens, hail_credentials
from hailtop.httpx import client_session


def init_parser(parser):  # pylint: disable=unused-argument
    pass


async def async_main():
    deploy_config = get_deploy_config()

    auth_ns = deploy_config.service_ns('auth')
    tokens = get_tokens()
    if auth_ns not in tokens:
        print('Not logged in.')
        return

    headers = await hail_credentials().auth_headers()
    async with client_session(headers=headers) as session:
        async with session.post(deploy_config.url('auth', '/api/v1alpha/logout')):
            pass
    auth_ns = deploy_config.service_ns('auth')

    del tokens[auth_ns]
    tokens.write()

    print('Logged out.')


def main(args, pass_through_args):  # pylint: disable=unused-argument
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main())
