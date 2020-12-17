import sys
import argparse

from . import list_billing_projects
from . import get


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'billing',
        help="Manage billing on the service managed by the Hail team.",
        description="Manage billing on the service managed by the Hail team.")
    subparsers = parser.add_subparsers(
        title='hailctl batch billing subcommand',
        dest='hailctl batch billing subcommand',
        required=True)

    list_billing_projects.init_parser(subparsers)
    get.init_parser(subparsers)


def main(args, client):
    if args.module.startswith('hailctl batch billing list'):
        list_billing_projects.main(args, client)
    else:
        assert args.module.startswith('hailctl batch billing get')
        get.main(args, client)
