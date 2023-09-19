import asyncio
import typer
import webbrowser

from typing import List, Optional, Annotated as Ann
from typer import Option as Opt


from . import config


app = typer.Typer(
    name='dev',
    no_args_is_help=True,
    help='Manage Hail development utilities.',
    pretty_exceptions_show_locals=False,
)
app.add_typer(
    config.app,
)


@app.command()
def deploy(
    branch: Ann[str, Opt('--branch', '-b', help='Fully-qualified branch, e.g., hail-is/hail:feature')],
    steps: Ann[List[str], Opt('--steps', '-s', help='Comma-separated list of steps to run.')],
    excluded_steps: Ann[
        Optional[List[str]],
        Opt(
            '--excluded_steps',
            '-e',
            help='Comma-separated list of steps to forcibly exclude. Use with caution!',
        ),
    ] = None,
    extra_config: Ann[
        Optional[List[str]],
        Opt(
            '--extra-config',
            '-c',
            help='Comma-separated list of key=value pairs to add as extra config parameters.',
        ),
    ] = None,
    open: Ann[bool, Opt('--open', '-o', help='Open the deploy batch page in a web browser.')] = False,
):
    '''Deploy a branch.'''
    asyncio.run(_deploy(branch, steps, excluded_steps or [], extra_config or [], open))


async def _deploy(branch: str, steps: List[str], excluded_steps: List[str], extra_config: List[str], open: bool):
    from hailtop.config import get_deploy_config  # pylint: disable=import-outside-toplevel
    from hailtop.utils import unpack_comma_delimited_inputs, unpack_key_value_inputs  # pylint: disable=import-outside-toplevel
    from .ci_client import CIClient  # pylint: disable=import-outside-toplevel

    deploy_config = get_deploy_config()
    steps = unpack_comma_delimited_inputs(steps)
    excluded_steps = unpack_comma_delimited_inputs(excluded_steps)
    extra_config_dict = unpack_key_value_inputs(extra_config)
    async with CIClient(deploy_config) as ci_client:
        batch_id = await ci_client.dev_deploy_branch(branch, steps, excluded_steps, extra_config_dict)
        url = deploy_config.url('ci', f'/batches/{batch_id}')
        print(f'Created deploy batch, see {url}')
        if open:
            webbrowser.open(url)
