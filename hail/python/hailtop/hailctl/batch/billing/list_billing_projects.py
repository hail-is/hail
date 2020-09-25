from ..batch_cli_utils import make_formatter


def init_parser(parser):
    parser.add_argument('-o', type=str, default='yaml', help="Specify output format",
                        choices=["yaml", "json"])


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    billing_projects = client.list_billing_projects()
    print(make_formatter(args.o)(billing_projects))
