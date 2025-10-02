import asyncio
from enum import Enum
from typing import Annotated as Ann
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

import orjson
import typer
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


CloudFuseEntry = Tuple[str, str, bool]


def parse_cloudfuse_entry(value: str) -> CloudFuseEntry:
    parts = value.split(':')
    if len(parts) != 3:
        raise typer.BadParameter(
            f"Invalid cloudfuse entry format: '{value}'. Expected 'bucket_name:mount_point:read_only'."
        )
    bucket_name = parts[0]
    mount_point = parts[1]

    read_only_str = parts[2].lower()
    if read_only_str == 'true':
        flag = True
    elif read_only_str == 'false':
        flag = False
    else:
        raise typer.BadParameter(
            f"Invalid read only flag in cloudfuse entry: '{parts[2]}'. Expected 'true' or 'false'."
        )

    return (bucket_name, mount_point, flag)


def parse_key_value_pair(value: str) -> Tuple[str, str]:
    if '=' not in value:
        raise typer.BadParameter(f"Invalid format for key-value pair: '{value}'. Expected 'KEY=VALUE'.")
    key, val = value.split('=', 1)
    return (key, val)


def parse_file_mount(value: str) -> Tuple[str, str]:
    if ':' not in value:
        raise typer.BadParameter(f"Invalid format for file mount: '{value}'. Expected 'src:dest'.")
    src, dest = value.split(':', 1)
    if not src or not dest:
        raise typer.BadParameter(f"Invalid format for file mount: '{value}'. Expected 'src:dest'.")
    if not dest.startswith('/'):
        raise typer.BadParameter(f"Invalid format for file mount: '{value}'. dest must be an absolute path.")
    return (src, dest)


@app.command(
    name='submit',
    help='Submit a job using files mounted from the local file system.',
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def submit(
    ctx: typer.Context,
    command_and_args: List[str] = typer.Argument(
        ...,
        help="The command to execute inside the container, followed by its arguments. "
        "Example: python3 /script.py a b --bar 5 --baz 7",
    ),
    *,
    image: Ann[
        str,
        Opt(
            help='Name of Docker image for the job',
            envvar='HAIL_GENETICS_HAIL_IMAGE',
        ),
    ] = f'hailgenetics/hail:{__pip_version__}',
    volume_mounts: Ann[
        Optional[List[str]],
        Opt(
            ...,
            "-v",
            help="Volume mounts in the format 'source:destination'. Can be specified multiple times.",
            parser=parse_file_mount,
        ),
    ] = None,
    name: Ann[Optional[str], Opt(help='The name of the batch.')] = None,
    cpu: Ann[str, Opt(help='CPU for the job.')] = '1',
    memory: Ann[str, Opt(help='Memory for the job.')] = 'standard',
    storage: Ann[str, Opt(help='Storage for the job.')] = '0Gi',
    machine_type: Ann[Optional[str], Opt(help='Use a specific job-private machine. Example: n1-standard-1.')] = None,
    spot: Ann[bool, Opt(help='Use a spot instance.')] = True,
    workdir: Ann[str, Opt(help='Working directory for the job.')] = '/',
    cloudfuse: Ann[
        Optional[List[str]],
        Opt(
            help="Specify a cloudfuse binding 'bucket_name:mount_point:read_only'. Can be specified multiple times.",
            parser=parse_cloudfuse_entry,
        ),
    ] = None,
    env: Ann[
        Optional[List[str]],
        Opt(
            help="Specify an environment variable in KEY=VALUE format. Can be specified multiple times.",
            parser=parse_key_value_pair,
        ),
    ] = None,
    billing_project: Ann[Optional[str], Opt(help='The billing project to use.')] = None,
    remote_tmpdir: Ann[Optional[str], Opt(help='The remote tmpdir to use.')] = None,
    requester_pays_project: Ann[Optional[str], Opt(help='The requester pays project to use.')] = None,
    regions: Ann[
        Optional[List[str]],
        Opt(..., "--region", help="Specify a region to run a job in. Can be provided multiple times."),
    ] = None,
    attributes: Ann[
        Optional[List[str]],
        Opt(
            ...,
            "--attr",
            help="Specify an attribute in KEY=VALUE format. Can be specified multiple times.",
            parser=parse_key_value_pair,
        ),
    ] = None,
    shell: Ann[str, Opt(help='Shell to use when running the job.')] = '/bin/bash',
    output: StructuredFormatPlusTextOption = StructuredFormatPlusText.TEXT,
    wait: Ann[bool, Opt(help='Wait for the batch to complete.')] = False,
    quiet: Ann[bool, Opt(help='Do not show progress bar for the batch.')] = False,
):
    """Submit a batch with a single job that runs SCRIPT, optionally with ARGUMENTS.

    $ hailctl batch submit [OPTIONS] COMMAND [ARGUMENTS]


    Specify the command and arguments:

    $ hailctl batch submit --image docker.io/image python3 my_script.py 1 2 --output my_output.txt


    Specify the working directory:

    $ hailctl batch submit --workdir /workdir/ python3 my_script.py 1 2 --output my_output.txt


    Specify a docker image to use for the job:

    $ hailctl batch submit --image docker.io/image python3 my_script.py


    Specify the name of the batch to submit:

    $ hailctl batch submit --name my-batch python3 my_script.py


    Add additional files to your job using the `-v` option where the value is "src:dest" where dest must be a fully qualified path:


    Copy a local file into the root directory of the container:

    $ hailctl batch submit -v a-file:/ python3 my_script.py


    Copy a local file into the root directory of the container and rename it:

    $ hailctl batch submit -v a-file:/a-file-renamed python3 my_script.py


    Copy the local working directory to the root directory of the container:

    $ hailctl batch submit --files ./:/ python3 my_script.py


    Copy the local working directory to a new subdirectory of the container:

    $ hailctl batch submit --files ./:/workdir/ python3 /workdir/my_script.py


    Copy the contents of a local directory into a new directory inside the container:

    $ hailctl batch submit --workdir /myproject/ --files /Users/jdoe/my-project:/myproject python3 my_script.py


    Notes

    -----

    If SRC is a file and DST is a file, SRC will be copied to DST, otherwise

    IF SRC is a file and DST is a directory, SRC will be copied into DST, otherwise

    IF SRC is a directory and DST has a trailing slash, the contents of SRC will be recursively copied into DST, otherwise

    IF SRC is a directory without a trailing slash and DST has no trailing slash, the contents of SRC will be recursively copied into DST, otherwise

    IF SRC is a directory without a trailing slash and DST has a trailing slash, the directory SRC will be moved into DST, otherwise

    An error will be raised.


    Hail Configuration:

    Local Hail configuration settings are automatically uploaded and stored in `/hail-config/hail/`.

    """

    from .submit import HailctlBatchSubmitError  # pylint: disable=import-outside-toplevel
    from .submit import submit as _submit  # pylint: disable=import-outside-toplevel

    assert command_and_args

    try:
        _submit(
            image,
            entrypoint=command_and_args,
            name=name,
            cpu=cpu,
            memory=memory,
            storage=storage,
            machine_type=machine_type,
            spot=spot,
            workdir=workdir,
            cloudfuse=cloudfuse,  # pyright: ignore
            env=env,  # pyright: ignore
            billing_project=billing_project,
            remote_tmpdir=remote_tmpdir,
            regions=regions,
            requester_pays_project=requester_pays_project,
            attributes=attributes,  # pyright: ignore
            volume_mounts=volume_mounts,  # pyright: ignore
            shell=shell,
            output=output,
            wait=wait,
            quiet=quiet,
        )
    except HailctlBatchSubmitError as err:
        print(err.message)
        raise typer.Exit(err.exit_code)
