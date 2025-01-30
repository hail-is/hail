import asyncio
from enum import Enum
from pathlib import Path
from typing import Annotated as Ann
from typing import (
    Any,
    Dict,
    List,
    Optional,
    cast,
)

import orjson
import typer
from typer import Argument as Arg
from typer import Option as Opt

from hailtop import __pip_version__

from . import billing, list_batches
from .batch_cli_utils import (
    ExtendedOutputFormat,
    ExtendedOutputFormatOption,
    StructuredFormat,
    StructuredFormatOption,
    StructuredFormatPlusText,
    StructuredFormatPlusTextOption,
    get_batch_if_exists,
    get_job_if_exists,
    make_formatter,
)
from .initialize import async_basic_initialize

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
    """List batches."""
    list_batches.list(query, limit, before, full, output)


@app.command()
def get(batch_id: int, output: StructuredFormatOption = StructuredFormat.YAML):
    """Get information on the batch with id BATCH_ID."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        batch = get_batch_if_exists(client, batch_id)
        if batch:
            print(make_formatter(output)([batch.last_known_status()]))
        else:
            print(f"Batch with id {batch_id} not found")


@app.command()
def cancel(batch_id: int):
    """Cancel the batch with id BATCH_ID."""
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
    """Delete the batch with id BATCH_ID."""
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
    """Get the log for the job with id JOB_ID in the batch with id BATCH_ID."""
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
    """Wait for the batch with id BATCH_ID to complete, then print status."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        batch = get_batch_if_exists(client, batch_id)
        if batch is None:
            print(f"Batch with id {batch_id} not found")
            raise typer.Exit(1)

        quiet = quiet or output != StructuredFormatPlusText.TEXT
        out = batch.wait(disable_progress_bar=quiet)
        if output == StructuredFormatPlusText.JSON:
            print(orjson.dumps(out).decode('utf-8'))
        else:
            print(out)


@app.command()
def job(batch_id: int, job_id: int, output: StructuredFormatOption = StructuredFormat.YAML):
    """Get the status and specification for the job with id JOB_ID in the batch with id BATCH_ID."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        job = get_job_if_exists(client, batch_id, job_id)

        if job is not None:
            assert job._status
            print(
                make_formatter(output)(
                    [cast(Dict[str, Any], job._status)]  # https://stackoverflow.com/q/71986632/6823256
                )
            )
        else:
            print(f"Job with ID {job_id} on batch {batch_id} not found")


@app.command('init', help='Initialize a Hail Batch environment.')
def initialize(verbose: Ann[bool, Opt('--verbose', '-v', help='Print gcloud commands being executed')] = False):
    asyncio.run(async_basic_initialize(verbose=verbose))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def submit(
    ctx: typer.Context,
    script: Ann[Path, Arg(help='File to execute', show_default=False, exists=True, resolve_path=True, dir_okay=False)],
    arguments: Ann[
        Optional[List[str]], Arg(help='You should use -- if you want to pass option-like arguments through.')
    ] = None,
    *,
    name: Ann[Optional[str], Opt(help='The name of the batch.')] = None,
    image: Ann[
        Optional[str],
        Opt(
            help='Name of Docker image for the job',
            envvar='HAIL_GENETICS_HAIL_IMAGE',
            show_default=f'hailgenetics/hail:{__pip_version__}',
        ),
    ] = None,
    files: Ann[
        Optional[List[str]],
        Opt(help='Extra files or folders to add to the working directory of the job.'),
    ] = None,
    output: StructuredFormatPlusTextOption = StructuredFormatPlusText.TEXT,
    wait: Ann[bool, Opt(help='Wait for the batch to complete.')] = False,
    quiet: Ann[bool, Opt('--quiet', '-q', help='Do not show progress bar for the batch.')] = False,
):
    """Submit a batch with a single job that runs SCRIPT, optionally with ARGUMENTS.

    Use '--' to pass additional arguments and switches to SCRIPT:

    $ hailctl batch submit [OPTIONS] SCRIPT [-- ARGUMENTS]



    Specify a docker image to use for the job:

    $ hailctl batch submit SCRIPT --image docker.io/image



    Specify the name of the batch to submit:

    $ hailctl batch submit SCRIPT --name my-batch



    Add additional files to your job using the --files SRC[:DST] option as follows:



    Copy a local file or folder into the working directory of the job:

    $ hailctl batch submit SCRIPT --files a-file-or-folder



    Copy the local working directory to the working directory of the job:

    $ hailctl batch submit --files .

    $ hailctl batch submit --files .:.



    Copy a local file or folder DRC to an absolute path or a path relative to the job's working directory:

    $ hailctl batch submit SCRIPT --files src:dst



    Copy a local file or folder to DST, using environment variables in the SRC path

    $ hailctl batch submit SCRIPT --files "${HOME}/foo":dst



    Copy the result of globbing a local folder SRC with PATTERN into DST on the worker:

    $ hailctl batch submit SCRIPT --files src/[pattern]:dst



    Notes

    -----

    SCRIPTs ending in '.py' will be invoked with `python3`, or as an executable otherwise.



    Relative DST paths are relative to the worker's working directory



    If DST does not exist, SRC will be copied to DST, otherwise

    If SRC is a file and DST is a file, DST will be replaced by SRC, otherwise

    If SRC is a file and DST is a folder, SRC will be copied into DST, otherwise

    If SRC is a folder and DST is a folder, the contents of SRC will to DST, otherwise

    If DST is a file, DST will be overwritten by SRC if SRC is a file, otherwise

    An error will be raised.



    Environment variables are permitted in SRC paths only



    Recursive glob patterns are not supported
    """
    from .submit import HailctlBatchSubmitError  # pylint: disable=import-outside-toplevel
    from .submit import submit as _submit  # pylint: disable=import-outside-toplevel

    try:
        asyncio.run(_submit(script, name, image, files or [], output, wait, quiet, *ctx.args, *(arguments or [])))
    except HailctlBatchSubmitError as err:
        print(err.message)
        raise typer.Exit(err.exit_code)
