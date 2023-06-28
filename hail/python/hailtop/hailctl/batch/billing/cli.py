import typer

from ..batch_cli_utils import make_formatter, StructuredFormat, StructuredFormatOption


app = typer.Typer(
    name='billing',
    no_args_is_help=True,
    help='Manage billing on the service managed by the Hail team.',
    pretty_exceptions_show_locals=False,
)


@app.command()
def get(billing_project: str, output: StructuredFormatOption = StructuredFormat.YAML):
    '''Get the billing information for BILLING_PROJECT.'''
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        billing_project_data = client.get_billing_project(billing_project)
        print(make_formatter(output.value)(billing_project_data))


@app.command()
def list(output: StructuredFormatOption = StructuredFormat.YAML):
    '''List billing projects.'''
    from hailtop.batch_client.client import BatchClient  # pylint: disable=import-outside-toplevel

    with BatchClient('') as client:
        billing_projects = client.list_billing_projects()
        format = make_formatter(output.value)
        print(format(billing_projects))
