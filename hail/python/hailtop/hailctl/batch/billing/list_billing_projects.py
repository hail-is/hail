from ..batch_cli_utils import make_formatter, TABLE_FORMAT_OPTIONS


def init_parser(parser):
    parser.add_argument('-o', type=str, default='grid', choices=TABLE_FORMAT_OPTIONS)


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    billing_projects = client.list_billing_projects()

    if args.o not in ('json', 'yaml'):
        for bp in billing_projects:
            bp['users'] = "\n".join(bp['users'])

    format = make_formatter(args.o)
    print(format(billing_projects))
