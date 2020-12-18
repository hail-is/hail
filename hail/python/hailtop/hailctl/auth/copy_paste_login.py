import asyncio
import click

from hailtop.auth import async_copy_paste_login

from .auth import auth


async def async_main(copy_paste_token, namespace):
    auth_ns, username = await async_copy_paste_login(copy_paste_token, namespace)

    if auth_ns == 'default':
        print(f'Logged in as {username}.')
    else:
        print(f'Logged into namespace {auth_ns} as {username}.')


@auth.command(
    help="Obtain Hail credentials with a copy paste token.")
@click.argument('copy_paste_token')
@click.option('--namespace', '-n')
def copy_paste_login(copy_paste_token, namespace):  # pylint: disable=unused-argument
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(copy_paste_token, namespace))
