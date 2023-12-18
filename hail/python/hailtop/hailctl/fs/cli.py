from datetime import datetime
from typing import List, Optional, Annotated as Ann

import humanize
import tabulate
import typer

from typer import Option as Opt, Argument as Arg

import hailtop.fs

app = typer.Typer(
    name='fs',
    no_args_is_help=True,
    help='Object and File utilites',
    pretty_exceptions_show_locals=False,
)


@app.command()
def du(ctx: typer.Context,
       paths: Ann[Optional[List[str]], Arg()] = None,
       human_readable: Ann[bool, Opt('-h', '--human-readable', help='print sizes in human readable format')] = False,
       ):
    '''
    Display storage resourse usage
    '''
    if not paths:
        paths = ['.']
    print(paths, human_readable)


@app.command()
def ls(ctx: typer.Context,
       paths: Ann[Optional[List[str]], Arg()] = None,
       long: Ann[bool, Opt('-l', '--long', help='use long listing format')] = False,
       human_readable: Ann[bool, Opt('-h', '--human-readable', help='print human readable (base 10) file sizes')] = False,
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
