from hailtop.config import get_deploy_config
from hailtop.auth import get_tokens


def init_parser(parent_subparsers):
    list_parser = parent_subparsers.add_parser(
        'list',
        help='List Hail credentials.',
        description='List Hail credentials.')
    list_parser.set_defaults(module='hailctl auth list')


def main(args):
    deploy_config = get_deploy_config()
    auth_ns = deploy_config.service_ns('auth')
    tokens = get_tokens()
    for ns in tokens:
        if ns == auth_ns:
            s = '*'
        else:
            s = ' '
        print(f'{s}{ns}')
