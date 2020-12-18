import os
import click

from hailtop.auth import namespace_auth_headers
from hailtop.config import get_deploy_config

from .dataproc import dataproc


@dataproc.command(
    help="Issue authenticated curl requests to Hail infrastructure.")
@click.argument('namespace')
@click.argument('service')
@click.argument('path')
@click.argument('curl_args')
def curl(namespace, service, path, curl_args):
    deploy_config = get_deploy_config()
    deploy_config = deploy_config.with_default_namespace(namespace)
    headers = namespace_auth_headers(deploy_config, namespace)
    headers = [x
               for k, v in headers.items()
               for x in ['-H', f'{k}: {v}']]
    path = deploy_config.url(service, path)
    os.execvp('curl', ['curl', *headers, *curl_args, path])
