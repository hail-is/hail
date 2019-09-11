from hailtop.config import get_deploy_config
from hailtop.auth import get_tokens


def init_parser(parser):  # pylint: disable=unused-argument
    pass


def main(args, pass_through_args):  # pylint: disable=unused-argument
    deploy_config = get_deploy_config()
    auth_ns = deploy_config.service_ns('auth')
    tokens = get_tokens()
    for ns in tokens:
        if ns == auth_ns:
            s = '*'
        else:
            s = ' '
        print(f'{s}{ns}')
