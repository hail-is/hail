import sys
import tabulate
from ..batch_cli_utils import make_formatter


def init_parser(parser):
    parser.add_argument('-o', type=str, default='grid',
                        help='specify output format (json, yaml, csv, tsv, or any tabulate format)')


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    choices = ['json', 'yaml', *tabulate.tabulate_formats]
    if args.o not in choices:
        print('invalid output format:', args.o, file=sys.stderr)
        print('must be one of:', *choices, file=sys.stderr)
        sys.exit(1)

    billing_projects = client.list_billing_projects()

    if args.o not in ('json', 'yaml'):
        for bp in billing_projects:
            bp['users'] = "\n".join(bp['users'])

    format = make_formatter(args.o)
    format(billing_projects)
