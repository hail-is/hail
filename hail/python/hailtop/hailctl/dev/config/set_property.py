import os
import json
import click

from hailtop.config import get_deploy_config

from .config import config


@config.command(name='set',
                help="Set deploy configuration property.")
@click.argument('property')
@click.argument('value')
def set_property(property, value):
    deploy_config = get_deploy_config()
    config = deploy_config.get_config()

    config[property] = value

    config_file = os.environ.get(
        'HAIL_DEPLOY_CONFIG_FILE', os.path.expanduser('~/.hail/deploy-config.json'))
    with open(config_file, 'w') as f:
        json.dump(config, f)
