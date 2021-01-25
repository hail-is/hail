import sys
from concurrent.futures import ThreadPoolExecutor
import asyncio
import click

from hailtop.aiotools import RouterAsyncFS, LocalAsyncFS, Transfer
from hailtop.aiogoogle import GoogleStorageAsyncFS
from ..hailctl import hailctl


async def async_main(srcs, dest, treat_dest_as):
    with ThreadPoolExecutor() as thread_pool:
        async with RouterAsyncFS(
                'file', [LocalAsyncFS(thread_pool), GoogleStorageAsyncFS()]) as fs:
            print(srcs, dest, treat_dest_as)
            await fs.copy(Transfer(srcs, dest, treat_dest_as=treat_dest_as))


@hailctl.command(
    help="""Copy files.  'hailctl cp' can be invoked three ways:

    cp SOURCE... DIRECTORY

    cp --no-target-directory/-T SOURCE DEST

    cp --target-directory/-t DIRECTORY SOURCE...

The -t and -T options are incompatible.
""")
@click.option('--no-target-directory', '-T', is_flag=True,
              help="treat DEST as a normal file")
@click.option('--target-directory', '-t',
              metavar='DIRECTORY',
              help="copy all SOURCE arguments into DIRECTORY")
@click.argument('args', nargs=-1)
def cp(no_target_directory, target_directory, args):  # pylint: disable=invalid-name
    if no_target_directory and target_directory:
        print('error: cannot combine --no-target-directory/-T and --target-directory/-t', file=sys.stderr)
        sys.exit(1)

    treat_dest_as = None
    if target_directory is not None:
        if len(args) < 1:
            print('error: no sources specified')
            sys.exit(1)
        srcs = list(args)
        dest = target_directory
        treat_dest_as = Transfer.TARGET_DIR
    else:
        if len(args) < 2:
            print('error: too few arguments, at least two expected, a source and destination')
            sys.exit(1)
        srcs = list(args[:-1])
        dest = args[-1]

        if no_target_directory:
            treat_dest_as = Transfer.TARGET_FILE

    asyncio.run(async_main(srcs, dest, treat_dest_as))
