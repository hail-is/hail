from hailtop.config import get_deploy_config


def init_parser(parser):  # pylint: disable=unused-argument
    pass


def main(args):  # pylint: disable=unused-argument
    deploy_config = get_deploy_config()
    print(f'  location: {deploy_config.location()}')
    print(f'  default_namespace: {deploy_config._default_namespace}')
    print(f'  domain: {deploy_config._domain}')
