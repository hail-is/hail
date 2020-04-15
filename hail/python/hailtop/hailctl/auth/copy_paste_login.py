import os
import asyncio
import aiohttp

from hailtop.config import get_deploy_config
from hailtop.auth import get_tokens, namespace_auth_headers


def init_parser(parser):
    parser.add_argument("copy_paste_token", type=str,
                        help="Copy paste token.")
    parser.add_argument("--namespace", "-n", type=str,
                        help="Specify namespace for auth server.  (default: from deploy configuration)")


async def async_main(args):
    deploy_config = get_deploy_config()
    if args.namespace:
        auth_ns = args.namespace
        deploy_config = deploy_config.with_service('auth', auth_ns)
    else:
        auth_ns = deploy_config.service_ns('auth')
    headers = namespace_auth_headers(deploy_config, auth_ns, authorize_target=False)

    async with aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60), headers=headers) as session:
        async with session.get(deploy_config.url('auth', '/api/v1alpha/copy-paste-login'),
                               params={'copy_paste_token': args.copy_paste_token}) as resp:
            resp = await resp.json()
    token = resp['token']
    username = resp['username']

    tokens = get_tokens()
    tokens[auth_ns] = token
    dot_hail_dir = os.path.expanduser('~/.hail')
    if not os.path.exists(dot_hail_dir):
        os.mkdir(dot_hail_dir, mode=0o700)
    tokens.write()

    if auth_ns == 'default':
        print(f'Logged in as {username}.')
    else:
        print(f'Logged into namespace {auth_ns} as {username}.')


def main(args, pass_through_args):  # pylint: disable=unused-argument
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(args))
