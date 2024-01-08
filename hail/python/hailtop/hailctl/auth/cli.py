import asyncio
import json
import sys
from typing import Annotated as Ann
from typing import Optional

import typer
from typer import Argument as Arg

app = typer.Typer(
    name='auth',
    no_args_is_help=True,
    help='Manage Hail credentials.',
    pretty_exceptions_show_locals=False,
)


@app.command()
def login():
    '''Obtain Hail credentials.'''
    from .login import async_login  # pylint: disable=import-outside-toplevel

    asyncio.run(async_login())


@app.command()
def copy_paste_login(copy_paste_token: str):
    '''Obtain Hail credentials with a copy paste token.'''
    from hailtop.auth import copy_paste_login  # pylint: disable=import-outside-toplevel
    from hailtop.config import get_deploy_config  # pylint: disable=import-outside-toplevel

    username = copy_paste_login(copy_paste_token)
    print(f'Logged into {get_deploy_config().base_url("auth")} as {username}.')


@app.command()
def logout():
    '''Revoke Hail credentials.'''
    from hailtop.auth import async_logout  # pylint: disable=import-outside-toplevel

    asyncio.run(async_logout())


@app.command()
def list():
    '''List Hail credentials.'''
    from hailtop.auth import get_tokens  # pylint: disable=import-outside-toplevel
    from hailtop.config import get_deploy_config  # pylint: disable=import-outside-toplevel

    deploy_config = get_deploy_config()
    tokens = get_tokens()
    for ns in tokens:
        if ns == deploy_config.default_namespace():
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
    hail_identity: Optional[str] = None,
    hail_credentials_secret_name: Optional[str] = None,
    wait: bool = False,
):
    '''
    Create a new Hail user with username USERNAME and login ID LOGIN_ID.
    '''
    from .create_user import polling_create_user  # pylint: disable=import-outside-toplevel

    asyncio.run(
        polling_create_user(
            username, login_id, developer, service_account, hail_identity, hail_credentials_secret_name, wait=wait
        )
    )


@app.command()
def delete_user(
    username: str,
    wait: bool = False,
):
    '''
    Delete the Hail user with username USERNAME.
    '''
    from .delete_user import polling_delete_user  # pylint: disable=import-outside-toplevel

    asyncio.run(polling_delete_user(username, wait))
