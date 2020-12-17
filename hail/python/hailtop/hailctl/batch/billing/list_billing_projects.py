from ..batch_cli_utils import make_formatter


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'list',
        help="List billing projects",
        description="List billing projects")
    parser.set_defaults(module='hailctl batch billing list')
    parser.add_argument('-o', type=str, default='yaml', help="Specify output format",
                        choices=["yaml", "json"])


def main(args, client):
    billing_projects = client.list_billing_projects()
    print(make_formatter(args.o)(billing_projects))
