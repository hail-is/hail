from hailtop.config import get_deploy_config

from .config import config


@config.command()
def show():
    deploy_config = get_deploy_config()
    print(f'  location: {deploy_config.location()}')
    print(f'  default: {deploy_config._default_namespace}')
    print(f'  domain: {deploy_config._domain}')
