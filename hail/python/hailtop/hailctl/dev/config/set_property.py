import os
import json

from hailtop.config import get_deploy_config


def init_parser(parser):
    parser.add_argument("property", type=str,
                        help="Property to set.",
                        choices=['location', 'default_namespace', 'domain'])
    parser.add_argument("value", type=str,
                        help="Value to set property to.")


def main(args):
    deploy_config = get_deploy_config()
    config = deploy_config.get_config()

    p = args.property
    config[p] = args.value

    config_file = os.environ.get(
        'HAIL_DEPLOY_CONFIG_FILE', os.path.expanduser('~/.hail/deploy-config.json'))
    with open(config_file, 'w') as f:
        json.dump(config, f)
