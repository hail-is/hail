import pprint

import typer
import hailtop.fs

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
       ):
    '''
    List objects
    '''
    if not paths:
        paths = ['.']
    for path in paths:
        listing = hailtop.fs.ls(path)
        print(*(item.path for item in listing), sep='\n')
