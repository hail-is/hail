import asyncio

from datetime import datetime
from typing import List, Optional, Annotated as Ann

import humanize
import typer

from typer import Option as Opt, Argument as Arg

app = typer.Typer(
    name='fs',
    no_args_is_help=True,
    help='Object and File utilites',
    pretty_exceptions_show_locals=False,
)


async def async_du(fs, path: str, human_readable: bool, summarize: bool) -> int:
    total_size = 0
    async for item in await fs.listfiles(path):
        url, is_dir = await asyncio.gather(item.url(), item.is_dir())
        if is_dir:
            size = await async_du(fs, url, human_readable, summarize)
        else:
            stat = await item.status()
            size = await stat.size()
        total_size += size

        if not summarize:
            if human_readable:
                size = humanize.naturalsize(size, gnu=True)
            print(f'{size:>15}\t{url}')
    return total_size


@app.command()
def du(
    paths: Ann[Optional[List[str]], Arg()] = None,
    human_readable: Ann[
        bool, Opt('-h', '--human-readable', help='print sizes in human readable format (base 10)')
    ] = False,
    summarize: Ann[bool, Opt('-s', '--summarize', help='display only a total for each argument')] = False,
):
    '''
    Display storage resourse usage
    '''
    from hailtop.aiotools.router_fs import RouterAsyncFS  # pylint: disable=import-outside-toplevel
    from hailtop.utils import async_to_blocking  # pylint: disable=import-outside-toplevel

    if not paths:
        paths = ['.']
    fs = RouterAsyncFS()
    try:
        for path in paths:
            size = async_to_blocking(async_du(fs, path, human_readable, summarize))
            if human_readable:
                size = humanize.naturalsize(size, gnu=True)
            print(f'{size:>15}\t{path}')
    finally:
        async_to_blocking(fs.close())


@app.command()
def ls(
    paths: Ann[Optional[List[str]], Arg()] = None,
    long: Ann[bool, Opt('-l', '--long', help='use long listing format')] = False,
    human_readable: Ann[bool, Opt('-h', '--human-readable', help='print human readable file sizes (base 10)')] = False,
):
    '''
    List objects
    '''
    import tabulate  # pylint: disable=import-outside-toplevel
    from hailtop.fs.router_fs import RouterFS  # pylint: disable=import-outside-toplevel
    from hailtop.utils import async_to_blocking  # pylint: disable=import-outside-toplevel

    def display_bytes(size):
        if size is None or size <= 0:
            return None
        if human_readable:
            return humanize.naturalsize(size, gnu=True)
        return size

    fs = RouterFS()
    try:
        if not paths:
            paths = ['.']
        for path in paths:
            listing = fs.ls(path)
            listing.sort(key=lambda item: item.path)
            if long:
                listing = [
                    (
                        item.typ,
                        display_bytes(item.size),
                        datetime.utcfromtimestamp(item.modification_time)
                        if item.modification_time is not None
                        else None,
                        item.path,
                    )
                    for item in listing
                ]
                print(tabulate.tabulate(listing, tablefmt='plain', colalign=('global', 'right', 'global', 'global')))
            else:
                print(*(item.path for item in listing), sep='\n')
    finally:
        async_to_blocking(fs.afs.close())
