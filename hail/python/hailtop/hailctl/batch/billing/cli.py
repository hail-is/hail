import sys
import argparse

from . import list_billing_projects
from . import get


def init_parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl batch billing',
        description='Manage billing on the service managed by the Hail team.')
    subparsers = main_parser.add_subparsers()

    list_parser = subparsers.add_parser(
        'list',
        help="List billing projects",
        description="List billing projects")
    get_parser = subparsers.add_parser(
        'get',
        help='Get a particular billing project\'s info',
        description='Get a particular billing project\'s info')

    list_parser.set_defaults(module='list')
    list_billing_projects.init_parser(list_parser)

    get_parser.set_defaults(module='get')
    get.init_parser(get_parser)

    return main_parser


def main(args, pass_through_args, client):
    if not args:
        init_parser().print_help()
        sys.exit(0)
    jmp = {
        'list': list_billing_projects,
        'get': get
    }

    args, pass_through_args = init_parser().parse_known_args(args=pass_through_args)

    if not args or 'module' not in args:
        init_parser().print_help()
        sys.exit(0)

    if args.module not in jmp:
        sys.stderr.write(f"ERROR: no such module: {args.module!r}")
        init_parser().print_help()
        sys.exit(1)

    jmp[args.module].main(args, pass_through_args, client)
