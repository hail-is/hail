from enum import Enum
import os
import orjson
import typer

app = typer.Typer(
    name='config',
    no_args_is_help=True,
    help='Configure deployment.',
)


class DevConfigProperty(str, Enum):
    LOCATION = 'location'
    DEFAULT_NAMESPACE = 'default_namespace'
    DOMAIN = 'domain'


@app.command()
def set(property: DevConfigProperty, value: str):
    """Set dev config property PROPERTY to value VALUE."""
    config_file = os.environ.get('HAIL_DEPLOY_CONFIG_FILE', os.path.expanduser('~/.hail/deploy-config.json'))
    with open(config_file, 'r', encoding='utf-8') as old_config_f:
        config = orjson.loads(old_config_f.read())

    config[property] = value
    with open(config_file, 'w', encoding='utf-8') as new_config_f:
        new_config_f.write(orjson.dumps(config).decode('utf-8'))


@app.command()
def list():
    """List the settings in the dev config."""
    from hailtop.config import get_deploy_config  # pylint: disable=import-outside-toplevel

    deploy_config = get_deploy_config()
    print(f'  location: {deploy_config.location()}')
    print(f'  default_namespace: {deploy_config._default_namespace}')
    print(f'  domain: {deploy_config._domain}')
