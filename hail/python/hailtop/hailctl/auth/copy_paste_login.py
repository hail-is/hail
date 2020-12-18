import asyncio
import click

from hailtop.auth import async_copy_paste_login

from .auth import auth


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'copy-paste-login',
        help='Obtain Hail credentials with a copy paste token.',
        description='Obtain Hail credentials with a copy paste token.')
    parser.set_defaults(module='hailctl auth copy-paste-login')

    parser.add_argument("copy_paste_token", type=str,
                        help="Copy paste token.")
    parser.add_argument("--namespace", "-n", type=str,
                        help="Specify namespace for auth server.  (default: from deploy configuration)")


async def async_main(copy_paste_token, namespace):
    auth_ns, username = await async_copy_paste_login(copy_paste_token, namespace)

    if auth_ns == 'default':
        print(f'Logged in as {username}.')
    else:
        print(f'Logged into namespace {auth_ns} as {username}.')


@auth.command()
@click.argument('copy_paste_token')
@click.option('--namespace', '-n')
def copy_paste_login(copy_paste_token, namespace):  # pylint: disable=unused-argument
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(copy_paste_token, namespace))
