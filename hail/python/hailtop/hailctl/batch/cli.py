import asyncio
from enum import Enum
import typer
from typer import Option as Opt, Argument as Arg
import json

from typing import Optional, List, Annotated as Ann

from . import list_batches
from . import billing
from .initialize import async_basic_initialize
from . import submit as _submit
from .batch_cli_utils import (
    get_batch_if_exists,
    get_job_if_exists,
    make_formatter,
    StructuredFormat,
    StructuredFormatOption,
    StructuredFormatPlusText,
    StructuredFormatPlusTextOption,
    ExtendedOutputFormat,
    ExtendedOutputFormatOption,
)


app = typer.Typer(
    name='batch',
    no_args_is_help=True,
    help='Manage batches running on the batch service managed by the Hail team.',
    pretty_exceptions_show_locals=False,
)
app.add_typer(billing.cli.app)


@app.command()
def list(
    query: str = '',
    limit: int = 50,
    before: Optional[int] = None,
    full: bool = False,
    output: ExtendedOutputFormatOption = ExtendedOutputFormat.GRID,
):
    '''List batches.'''
    list_batches.list(query, limit, before, full, output)


@app.command()
def get(batch_id: int, output: StructuredFormatOption = StructuredFormat.YAML):
    '''Get information on the batch with id BATCH_ID.'''
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        batch = get_batch_if_exists(client, batch_id)
        if batch:
            print(make_formatter(output)([batch.last_known_status()]))
        else:
            print(f"Batch with id {batch_id} not found")


@app.command()
def cancel(batch_id: int):
    '''Cancel the batch with id BATCH_ID.'''
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        batch = get_batch_if_exists(client, batch_id)
        if batch:
            batch.cancel()
            print(f"Batch with id {batch_id} was cancelled successfully")
        else:
            print(f"Batch with id {batch_id} not found")


@app.command()
def delete(batch_id: int):
    '''Delete the batch with id BATCH_ID.'''
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        batch = get_batch_if_exists(client, batch_id)
        if batch:
            batch.delete()
            print(f"Batch with id {batch_id} was deleted successfully")
        else:
            print(f"Batch with id {batch_id} not found")


class JobContainer(str, Enum):
    INPUT = 'input'
    MAIN = 'main'
    OUTPUT = 'output'


@app.command()
def log(
    batch_id: int,
    job_id: int,
    container: Ann[Optional[JobContainer], Opt(help='Container name of the desired job')] = None,
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    '''Get the log for the job with id JOB_ID in the batch with id BATCH_ID.'''
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        maybe_job = get_job_if_exists(client, batch_id, job_id)
        if maybe_job is None:
            print(f"Job with ID {job_id} on batch {batch_id} not found")
            return

        if container:
            print(maybe_job.container_log(container))
        else:
            print(make_formatter(output)(maybe_job.log()))


@app.command()
def wait(
    batch_id: int,
    quiet: Ann[bool, Opt('--quiet', '-q', help='Do not print a progress bar for the batch.')] = False,
    output: StructuredFormatPlusTextOption = StructuredFormatPlusText.TEXT,
):
    '''Wait for the batch with id BATCH_ID to complete, then print status.'''
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        batch = get_batch_if_exists(client, batch_id)
        if batch is None:
            print(f"Batch with id {batch_id} not found")
            raise typer.Exit(1)

        quiet = quiet or output != StructuredFormatPlusText.TEXT
        out = batch.wait(disable_progress_bar=quiet)
        if output == StructuredFormatPlusText.JSON:
            print(json.dumps(out))
        else:
            print(out)


@app.command()
def job(batch_id: int, job_id: int, output: StructuredFormatOption = StructuredFormat.YAML):
    '''Get the status and specification for the job with id JOB_ID in the batch with id BATCH_ID.'''
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        job = get_job_if_exists(client, batch_id, job_id)

        if job is not None:
            assert job._status
            print(make_formatter(output)([job._status]))
        else:
            print(f"Job with ID {job_id} on batch {batch_id} not found")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def submit(
    ctx: typer.Context,
    script: str,
    arguments: Ann[Optional[List[str]], Arg(help='You should use -- if you want to pass option-like arguments through.')] = None,
    files: Ann[
        Optional[List[str]], Opt(help='Files or directories to add to the working directory of the job.')
    ] = None,
    name: Ann[str, Opt(help='The name of the batch.')] = '',
    image_name: Ann[Optional[str], Opt(help='Name of Docker image for the job (default: hailgenetics/hail)')] = None,
    output: StructuredFormatPlusTextOption = StructuredFormatPlusText.TEXT,
):
    '''Submit a batch with a single job that runs SCRIPT with the arguments ARGUMENTS.

    If you wish to pass option-like arguments you should use "--". For example:



    $ hailctl batch submit --image-name docker.io/image my_script.py -- some-argument --animal dog
    '''
    asyncio.run(_submit.submit(name, image_name, files or [], output, script, [*(arguments or []), *ctx.args]))


@app.command('init', help='Initialize a Hail Batch environment.')
def initialize(
        verbose: Ann[bool, Opt('--verbose', '-v', help='Print gcloud commands being executed')] = False
):
    asyncio.get_event_loop().run_until_complete(async_basic_initialize(verbose=verbose))
