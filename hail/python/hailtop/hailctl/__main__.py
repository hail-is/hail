import os

import typer

from .auth import cli as auth_cli
from .batch import cli as batch_cli
from .config import cli as config_cli
from .dataproc import cli as dataproc_cli
from .describe import describe
from .dev import cli as dev_cli
from .hdinsight import cli as hdinsight_cli

app = typer.Typer(
    help='Manage and monitor hail deployments.',
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

for cli in (
    auth_cli.app,
    batch_cli.app,
    config_cli.app,
    dataproc_cli.app,
    dev_cli.app,
    hdinsight_cli.app,
):
    app.add_typer(cli)


@app.command()
def version():
    '''Print version information and exit.'''
    import hailtop  # pylint: disable=import-outside-toplevel

    print(hailtop.version())


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def curl(
    namespace: str,
    service: str,
    path: str,
    ctx: typer.Context,
):
    '''Issue authenticated curl requests to Hail infrastructure.'''
    from hailtop.utils import async_to_blocking  # pylint: disable=import-outside-toplevel

    async_to_blocking(_curl(namespace, service, path, ctx))


async def _curl(
    namespace: str,
    service: str,
    path: str,
    ctx: typer.Context,
):
    from hailtop.auth import hail_credentials  # pylint: disable=import-outside-toplevel
    from hailtop.config import get_deploy_config  # pylint: disable=import-outside-toplevel

    deploy_config = get_deploy_config().with_default_namespace(namespace)
    async with hail_credentials(deploy_config=deploy_config) as credentials:
        headers_dict = await credentials.auth_headers()
    headers = [x for k, v in headers_dict.items() for x in ['-H', f'{k}: {v}']]
    path = deploy_config.url(service, path)
    os.execvp('curl', ['curl', *headers, *ctx.args, path])


app.command(help='Describe Hail Matrix Table and Table files.')(describe)


def main():
    app()
