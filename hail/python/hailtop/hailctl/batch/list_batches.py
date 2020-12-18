import sys
import csv
import tabulate
import click

from hailtop.batch_client.client import BatchClient

from .batch import batch
from .batch_cli_utils import make_formatter


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'list',
        help="List batches",
        description="List batches")
    parser.set_defaults(module='hailctl batch list')


@batch.command(
    help="List batches.")
@click.option('--query', '-q',
              help="See docs at https://batch.hail.is/batches.")
@click.option('--limit', '-l',
              type=int, default=50, show_default=True,
              help='Number of batches to return.')
@click.option('--all', '-a', is_flag=True,
              help='list all batches (overrides --limit)')
@click.option('--before', type=int, help='Start listing before supplied id.', default=None)
@click.option('--full', is_flag=True,
              help='When output is tabular, print more information.')
@click.option('--no-header',
              help='Do not print a table header.')
@click.option('--output-format', '-o',
              default='orgtbl', show_default=True,
              help='Specify output format (json, yaml, csv, tsv, or any tabulate format).')
def list_batches(query, limit, all, before, full, no_header, output_format):
    with BatchClient(None) as client:
        choices = ['json', 'yaml', 'csv', 'tsv', *tabulate.tabulate_formats]
        if output_format not in choices:
            print('invalid output format:', output_format, file=sys.stderr)
            print('must be one of:', *choices, file=sys.stderr)
            sys.exit(1)

        batch_list = client.list_batches(q=query, last_batch_id=before, limit=limit)
        statuses = [batch.last_known_status() for batch in batch_list]
        if output_format in ('json', 'yaml'):
            print(make_formatter(output_format)(statuses))
            return

        for status in statuses:
            status['state'] = status['state'].capitalize()

        if full:
            header = () if no_header else (
                'ID', 'PROJECT', 'STATE', 'COMPLETE', 'CLOSED', 'N_JOBS', 'N_COMPLETED',
                'N_SUCCEDED', 'N_FAILED', 'N_CANCELLED', 'TIME CREATED', 'TIME CLOSED',
                'TIME COMPLETED', 'DURATION', 'MSEC_MCPU', 'COST')
            rows = [[v for k, v in status.items() if k != 'attributes'] for status in statuses]
        else:
            header = () if no_header else ('ID', 'STATE')
            rows = [(status['id'], status['state']) for status in statuses]

        if output_format in ('csv', 'tsv'):
            delim = ',' if output_format == 'csv' else '\t'
            writer = csv.writer(sys.stdout, delimiter=delim)
            if header:
                writer.writerow(header)
            for row in rows:
                writer.writerow(row)
        else:
            print(tabulate.tabulate(rows, headers=header, tablefmt=output_format))
