import asyncio
import sys
from typing import List, Optional, Tuple, cast

import click
import typer

from hailtop import uvloopx
from hailtop.aiotools.plan import PlanError, plan
from hailtop.aiotools.sync import SyncError
from hailtop.aiotools.sync import sync as aiotools_sync

app_without_click = typer.Typer(
    name='fs',
    no_args_is_help=True,
    help='Use cloud object stores and local filesystems.',
    pretty_exceptions_show_locals=False,
)


@app_without_click.callback()
def callback():
    """
    Typer app, including Click subapp
    """


@click.command()
@click.option(
    '--copy-to',
    help='Pairs of source and destination URL. May be specified multiple times. The destination is always treated as a file. See --copy-into to copy into a directory',
    type=(str, str),
    required=False,
    multiple=True,
    default=(),
)
@click.option(
    '--copy-into',
    help='Copies the source path into the target path. The target must not be a file.',
    type=(str, str),
    required=False,
    multiple=True,
    default=(),
)
@click.option(
    '-v',
    '--verbose',
    help='Enable extra logging information including a progress bar.',
    is_flag=True,
    required=False,
    default=False,
)
@click.option(
    '--max-parallelism', help='The maximum number of concurrent requests.', type=int, required=False, default=75
)
@click.option(
    '--make-plan',
    help='The folder in which to create a new synchronization plan. Must not exist.',
    type=str,
    required=False,
)
@click.option('--use-plan', help='The plan to execute. Must exist.', type=str, required=False)
@click.option(
    '--gcs-requester-pays-project', help='The Google project to which to charge egress costs.', type=str, required=False
)
def sync(
    copy_to: List[Tuple[str, str]],
    copy_into: List[Tuple[str, str]],
    verbose: bool,
    max_parallelism: int,
    make_plan: Optional[str] = None,
    use_plan: Optional[str] = None,
    gcs_requester_pays_project: Optional[str] = None,
):
    """Synchronize files between one or more pairs of locations.

    If a corresponding file already exists at the destination with the same size in bytes, this
    command will not copy it. If you want to replace files that have the exact same size in bytes,
    delete the destination files first. THIS COMMAND DOES NOT CHECK MD5s OR SHAs!

    First generate a plan with --make-plan, then use the plan with --use-plan.

    -----

    Plans

    -----

    The command, `hailctl fs sync --make-plan` creates a "plan". A plan is a folder containing six
    files:

    1. matches: files whose names and sizes match. Two columns: source URL, destination URL.

    2. differs: files or folders whose names match but either differ in size or differ in type.
       Four columns: source URL, destination URL, source state, destination state. The states
       are either: file, dif, or a size. If either state is a size, both states are sizes.

    3. srconly: files only present in the source. One column: source URL.

    4. dstonly: files only present in the destination. One column: destination URL.

    5. plan: a proposed set of object-to-object copies. Two columns: source URL, destination URL.

    6. summary: a one-line file containing the total number of copies in plan and the total number of bytes which would be copied.


    --------

    Examples

    --------

    Copy all the files under gs://gcs-bucket/a to s3://s3-bucket/b. For example,
    gs://gcs-bucket/a/b/c will appear at s3://s3-bucket/b/b/c:



    $ hailctl fs sync --make-plan plan1 --copy-to gs://gcs-bucket/a s3://s3-bucket/b



    $ hailctl fs sync --use-plan plan1




    Copy all the files under gs://gcs-bucket/a into the (possibly not existing) s3://s3-bucket/b folder.
    For example, gs://gcs-bucket/a/b/c will appear at s3://s3-bucket/b/a/b/c:



    $ hailctl fs sync --make-plan plan1 --copy-into gs://gcs-bucket/a s3://s3-bucket/b



    $ hailctl fs sync --use-plan plan1




    Sync to cloud locations and then regenerate a plan folder to verify that all copies completed
    successfully:



    $ hailctl fs sync --make-plan plan1 --copy-to gs://gcs-bucket/a s3://s3-bucket/b



    $ hailctl fs sync --use-plan plan1



    $ hailctl fs sync --make-plan plan1 --copy-to gs://gcs-bucket/a s3://s3-bucket/b



    $ cat plan1/summary



    $ less plan1/plan
    """
    uvloopx.install()

    if (make_plan is None and use_plan is None) or (make_plan is not None and use_plan is not None):
        print('Must specify one of --make-plan or --use-plan. See hailctl fs sync --help.')
        raise typer.Exit(1)

    if make_plan:
        try:
            asyncio.run(plan(make_plan, copy_to, copy_into, gcs_requester_pays_project, verbose, max_parallelism))
        except PlanError as err:
            print('ERROR: ' + err.args[0])
            sys.exit(err.args[1])
    if use_plan:
        if copy_to or copy_into:
            print(
                'Do not specify --copy-to or --copy-into with --use-plan. Create the plan with --make-plan then call --use-plan without any --copy-to and --copy-into.'
            )
            raise typer.Exit(1)
        try:
            asyncio.run(aiotools_sync(use_plan, gcs_requester_pays_project, verbose, max_parallelism))
        except SyncError as err:
            print('ERROR: ' + err.args[0])
            sys.exit(err.args[1])


app = cast(click.Group, typer.main.get_command(app_without_click))
app.add_command(sync)
