from typing import Optional

import typer

from ...batch_cli_utils import StructuredFormat, StructuredFormatOption, make_formatter

app = typer.Typer(
    name='quotes',
    no_args_is_help=True,
    help='Manage billing quotes.',
    pretty_exceptions_show_locals=False,
)


@app.command()
def list(output: StructuredFormatOption = StructuredFormat.YAML):
    """List quotes visible to the current user."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        quotes = client.list_quotes()
        print(make_formatter(output.value)(quotes))


@app.command()
def create(
    name: str,
    cost_object: str = typer.Option(..., help='Cost object identifier.'),
    authorized_amount: Optional[str] = typer.Option(None, help='Authorized spending amount, or "unlimited".'),
    pi_name: Optional[str] = typer.Option(None, help='Principal investigator name.'),
    pm_designee: Optional[str] = typer.Option(None, help='Program manager designee.'),
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Create a new billing quote named NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.create_quote(name, cost_object, authorized_amount, pi_name, pm_designee, comment=comment)
        print(make_formatter(output.value)(result))


@app.command()
def describe(name: str, output: StructuredFormatOption = StructuredFormat.YAML):
    """Show details for quote NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        quote = client.get_quote(name)
        print(make_formatter(output.value)(quote))


@app.command()
def edit(
    name: str,
    cost_object: Optional[str] = typer.Option(None),
    authorized_amount: Optional[str] = typer.Option(None, help='New authorized amount, or "unlimited".'),
    pi_name: Optional[str] = typer.Option(None),
    pm_designee: Optional[str] = typer.Option(None),
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Edit fields on quote NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    updates = {}
    if cost_object is not None:
        updates['cost_object'] = cost_object
    if authorized_amount is not None:
        updates['authorized_amount'] = authorized_amount
    if pi_name is not None:
        updates['pi_name'] = pi_name
    if pm_designee is not None:
        updates['pm_designee'] = pm_designee

    if not updates:
        typer.echo('No fields to update.', err=True)
        raise typer.Exit(1)

    with BatchClient('') as client:
        result = client.edit_quote(name, comment=comment, **updates)
        print(make_formatter(output.value)(result))


@app.command()
def add_manager(
    name: str,
    user: str,
    role: str = typer.Option('manager', help='Role: "owner" or "manager".'),
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Add USER as a manager of quote NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.add_quote_manager(name, user, role, comment=comment)
        print(make_formatter(output.value)(result))


@app.command()
def remove_manager(
    name: str,
    user: str,
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Remove USER as a manager of quote NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.remove_quote_manager(name, user, comment=comment)
        print(make_formatter(output.value)(result))


@app.command()
def events(name: str, output: StructuredFormatOption = StructuredFormat.YAML):
    """Show audit events for quote NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.get_quote_events(name)
        print(make_formatter(output.value)(result))
