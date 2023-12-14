import tabulate
import typer
import hailtop.fs

from datetime import datetime
from typing import List, Optional, Annotated as Ann

from typer import Option as Opt, Argument as Arg

app = typer.Typer(
    name='fs',
    no_args_is_help=True,
    help='Object and File utilites',
    pretty_exceptions_show_locals=False,
)


@app.command()
def ls(ctx: typer.Context,
       paths: Ann[Optional[List[str]], Arg()] = None,
       long: Ann[bool, Opt('-l', '--long', help='use long listing format')] = False,
       ):
    '''
    List objects
    '''
    if not paths:
        paths = ['.']
    for path in paths:
        listing = hailtop.fs.ls(path)
        if long:
            listing = [(item.typ,
                        item.size if item.size > 0 else None,
                        datetime.utcfromtimestamp(item.modification_time)
                        if item.modification_time is not None else None,
                        item.path) for item in listing]
            print(tabulate.tabulate(listing, tablefmt='plain'))
        else:
            print(*(item.path for item in listing), sep='\n')
