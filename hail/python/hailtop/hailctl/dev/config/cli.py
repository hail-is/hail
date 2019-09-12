import os
import json
from hailtop.config import get_deploy_config


def init_parser(parser):
    parser.add_argument("namespace", type=str, nargs='?',
                        help="Default namespace.  Show the current configuration if not specified.")
    parser.add_argument("--location", "-l", type=str, default='external',
                        choices=['external', 'gce', 'k8s'],
                        help="Location.  (default: external)")
    parser.add_argument("--override", "-o", type=str, default='',
                        help="List of comma-separated service=namespace overrides.  (default: none)")


def main(args):
    if not args.namespace:
        deploy_config = get_deploy_config()
        print(f'  location: {deploy_config.location()}')
        print(f'  default: {deploy_config._default_ns}')
        if deploy_config._service_namespace:
            print('  overrides:')
            for service, ns in deploy_config._service_namespace.items():
                print(f'    {service}: {ns}')
        return

    override = args.override.split(',')
    override = [o.split('=') for o in override if o]
    service_namespace = {o[0]: o[1] for o in override}

    config = {
        'location': args.location,
        'default_namespace': args.namespace,
        'service_namespace': service_namespace
    }

    config_file = os.environ.get(
        'HAIL_DEPLOY_CONFIG_FILE', os.path.expanduser('~/.hail/deploy-config.json'))
    with open(config_file, 'w') as f:
        f.write(json.dumps(config))
