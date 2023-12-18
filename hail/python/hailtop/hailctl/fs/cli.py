import asyncio

from datetime import datetime
from typing import List, Optional, Annotated as Ann

import humanize
import tabulate
import typer

from typer import Option as Opt, Argument as Arg

import hailtop.fs

from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.utils import async_to_blocking

app = typer.Typer(
    name='fs',
    no_args_is_help=True,
    help='Object and File utilites',
    pretty_exceptions_show_locals=False,
)


async def async_du(fs: RouterAsyncFS, path: str, human_readable: bool, summarize: bool) -> int:
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
def du(ctx: typer.Context,
       paths: Ann[Optional[List[str]], Arg()] = None,
       human_readable: Ann[bool, Opt('-h', '--human-readable', help='print sizes in human readable format (base 10)')] = False,
       summarize: Ann[bool, Opt('-s', '--summarize', help='display only a total for each argument')] = False,
       ):
    '''
    Display storage resourse usage
    '''
    if not paths:
        paths = ['.']
    fs = RouterAsyncFS()
    for path in paths:
        size = async_to_blocking(async_du(fs, path, human_readable, summarize))
        if human_readable:
            size = humanize.naturalsize(size, gnu=True)
        print(f'{size:>15}\t{path}')


@app.command()
def ls(ctx: typer.Context,
       paths: Ann[Optional[List[str]], Arg()] = None,
       long: Ann[bool, Opt('-l', '--long', help='use long listing format')] = False,
       human_readable: Ann[bool, Opt('-h', '--human-readable', help='print human readable file sizes (base 10)')] = False,
       ):
    '''
    List objects
    '''
    def display_bytes(size):
        if size is None or size <= 0:
            return None
        if human_readable:
            return humanize.naturalsize(size, gnu=True)
        return size
    if not paths:
        paths = ['.']
    for path in paths:
        listing = hailtop.fs.ls(path)
        listing.sort(key=lambda item: item.path)
        if long:
            listing = [(item.typ,
                        display_bytes(item.size),
                        datetime.utcfromtimestamp(item.modification_time)
                        if item.modification_time is not None else None,
                        item.path) for item in listing]
            print(tabulate.tabulate(listing, tablefmt='plain',
                                    colalign=('global', 'right', 'global', 'global')))
        else:
            print(*(item.path for item in listing), sep='\n')
