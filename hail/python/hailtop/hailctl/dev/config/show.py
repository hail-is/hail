from hailtop.config import get_deploy_config

def init_parser(parser):
    pass

def main(args):
    deploy_config = get_deploy_config()
    print(f'  location: {deploy_config.location()}')
    print(f'  default: {deploy_config._default_namespace}')
    print(f'  domain: {deploy_config._domain}')
