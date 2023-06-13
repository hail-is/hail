import asyncio
import sys
import typer
from typer import Option as Opt, Argument as Arg
import json

from typing import Optional
from typing_extensions import Annotated as Ann


app = typer.Typer(
    name='auth',
    no_args_is_help=True,
    help='Manage Hail credentials.',
)


NamespaceOption = Ann[
    Optional[str],
    Opt('--namespace', '-n', help='Namespace for the auth server (default: from deploy configuration).'),
]


@app.command()
def login(namespace: NamespaceOption = None):
    '''Obtain Hail credentials.'''
    from .login import async_login  # pylint: disable=import-outside-toplevel
    asyncio.run(async_login(namespace))


@app.command()
def copy_paste_login(copy_paste_token: str, namespace: NamespaceOption = None):
    '''Obtain Hail credentials with a copy paste token.'''
    from hailtop.auth import copy_paste_login  # pylint: disable=import-outside-toplevel

    auth_ns, username = copy_paste_login(copy_paste_token, namespace)
    if auth_ns == 'default':
        print(f'Logged in as {username}.')
    else:
        print(f'Logged into namespace {auth_ns} as {username}.')


@app.command()
def logout():
    '''Revoke Hail credentials.'''
    from hailtop.auth import async_logout  # pylint: disable=import-outside-toplevel

    asyncio.run(async_logout())


@app.command()
def list():
    '''List Hail credentials.'''
    from hailtop.config import get_deploy_config  # pylint: disable=import-outside-toplevel
    from hailtop.auth import get_tokens  # pylint: disable=import-outside-toplevel

    deploy_config = get_deploy_config()
    auth_ns = deploy_config.service_ns('auth')
    tokens = get_tokens()
    for ns in tokens:
        if ns == auth_ns:
            s = '*'
        else:
            s = ' '
        print(f'{s}{ns}')


@app.command()
def user():
    '''Get Hail user information.'''
    from hailtop.auth import get_userinfo  # pylint: disable=import-outside-toplevel

    userinfo = get_userinfo()
    if userinfo is None:
        print('not logged in')
        sys.exit(1)
    result = {
        'username': userinfo['username'],
        'email': userinfo['login_id'],  # deprecated - backwards compatibility
        'gsa_email': userinfo['hail_identity'],  # deprecated - backwards compatibility
        'hail_identity': userinfo['hail_identity'],
        'login_id': userinfo['login_id'],
        'display_name': userinfo['display_name'],
    }
    print(json.dumps(result, indent=4))


@app.command()
def create_user(
    username: str,
    login_id: Ann[str, Arg(help="In Azure, the user's object ID in AAD. In GCP, the Google email")],
    developer: bool = False,
    service_account: bool = False,
    namespace: NamespaceOption = None,
    wait: bool = False,
):
    '''
    Create a new Hail user with username USERNAME and login ID LOGIN_ID.
    '''
    from .create_user import polling_create_user  # pylint: disable=import-outside-toplevel

    asyncio.run(polling_create_user(username, login_id, developer, service_account, namespace, wait))


@app.command()
def delete_user(
    username: str,
    namespace: NamespaceOption = None,
    wait: bool = False,
):
    '''
    Delete the Hail user with username USERNAME.
    '''
    from .delete_user import polling_delete_user  # pylint: disable=import-outside-toplevel

    asyncio.run(polling_delete_user(username, namespace, wait))
