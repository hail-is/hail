from typing import List, Optional

import typer

from ...batch_cli_utils import StructuredFormat, StructuredFormatOption, make_formatter

app = typer.Typer(
    name='projects',
    no_args_is_help=True,
    help='Manage billing projects.',
    pretty_exceptions_show_locals=False,
)


@app.command()
def list(output: StructuredFormatOption = StructuredFormat.YAML):
    """List billing projects visible to the current user."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        projects = client.list_billing_projects()
        print(make_formatter(output.value)(projects))


@app.command()
def create(
    name: str,
    quote: str = typer.Option('INTERNAL', help='Quote to create the billing project under.'),
    limit: Optional[float] = typer.Option(None, help='Spending limit in dollars.'),
    alert: Optional[float] = typer.Option(None, '--alert', help='Low-budget alert threshold in dollars.'),
    users: Optional[List[str]] = typer.Option(None, '--user', help='Initial users to add (repeatable).'),
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Create billing project NAME under QUOTE."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.create_billing_project_v2(
            name,
            quote_name=quote,
            limit=limit,
            low_budget_alert=alert,
            initial_users=[*users] if users else None,
            comment=comment,
        )
        print(make_formatter(output.value)(result))


@app.command()
def describe(name: str, output: StructuredFormatOption = StructuredFormat.YAML):
    """Show details for billing project NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        project = client.get_billing_project(name)
        print(make_formatter(output.value)(project))


@app.command()
def add_user(
    name: str,
    user: str,
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Add USER to billing project NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.add_user(user, name, comment=comment)
        print(make_formatter(output.value)(result))


@app.command()
def remove_user(
    name: str,
    user: str,
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Remove USER from billing project NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.remove_user(user, name, comment=comment)
        print(make_formatter(output.value)(result))


@app.command()
def set_alert(
    name: str,
    threshold: Optional[float] = typer.Argument(None, help='Alert threshold in dollars, or omit to clear.'),
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Set or clear the low-budget alert threshold for billing project NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.patch_billing_project(name, low_budget_alert=threshold, comment=comment)
        print(make_formatter(output.value)(result))


@app.command()
def set_limit(
    name: str,
    limit: Optional[float] = typer.Argument(None, help='Spending limit in dollars, or omit to clear.'),
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Set or clear the spending limit for billing project NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.patch_billing_project(name, limit=limit, comment=comment)
        print(make_formatter(output.value)(result))


@app.command()
def change_quote(
    name: str,
    quote: str,
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Move billing project NAME to a different QUOTE."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.change_billing_project_quote(name, quote, comment=comment)
        print(make_formatter(output.value)(result))


@app.command()
def events(name: str, output: StructuredFormatOption = StructuredFormat.YAML):
    """Show audit events for billing project NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.get_billing_project_events(name)
        print(make_formatter(output.value)(result))


@app.command()
def close(
    name: str,
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Close billing project NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.close_billing_project(name, comment=comment)
        print(make_formatter(output.value)(result))


@app.command()
def reopen(
    name: str,
    comment: Optional[str] = typer.Option(None, help='Short comment to record with this event.'),
    output: StructuredFormatOption = StructuredFormat.YAML,
):
    """Reopen billing project NAME."""
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        result = client.reopen_billing_project(name, comment=comment)
        print(make_formatter(output.value)(result))
