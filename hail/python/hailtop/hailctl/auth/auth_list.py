from hailtop.config import get_deploy_config
from hailtop.auth import get_tokens

from .auth import auth


@auth.command(
    name='list',
    help='List Hail credentials.')
def auth_list():
    deploy_config = get_deploy_config()
    auth_ns = deploy_config.service_ns('auth')
    tokens = get_tokens()
    for ns in tokens:
        if ns == auth_ns:
            s = '*'
        else:
            s = ' '
        print(f'{s}{ns}')
