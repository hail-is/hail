import os
import json


def init_parser(parser):
    parser.add_argument("--location", "-l", type=str, default='external',
                        choices=['external', 'gce', 'k8s'],
                        help="Location.  (default: external)")
    parser.add_argument("--namespace", "-n", type=str,
                        help="Default namespace.", required=True)
    parser.add_argument("--override", "-o", type=str, default='',
                        help="List of comma-separated service=namespace overrides.  (default: none)")


def main(args):
    override = args.override.split(',')
    override = [o.split('=') for o in override if o]
    print(override)
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
