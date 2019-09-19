import sys
import argparse

from . import login
from . import logout
from . import auth_list


def parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl auth',
        description='Manage Hail credentials.')
    subparsers = main_parser.add_subparsers()

    login_parser = subparsers.add_parser(
        'login',
        help='Obtain Hail credentials.',
        description='Obtain Hail credentials.')
    logout_parser = subparsers.add_parser(
        'logout',
        help='Revoke Hail credentials.',
        description='Revoke Hail credentials.')
    list_parser = subparsers.add_parser(
        'list',
        help='List Hail credentials.',
        description='List Hail credentials.')

    login_parser.set_defaults(module='login')
    login.init_parser(login_parser)

    logout_parser.set_defaults(module='logout')
    logout.init_parser(logout_parser)

    list_parser.set_defaults(module='list')
    auth_list.init_parser(list_parser)

    return main_parser


def main(args):
    if not args:
        parser().print_help()
        sys.exit(0)
    jmp = {
        'login': login,
        'logout': logout,
        'list': auth_list,
    }

    args, pass_through_args = parser().parse_known_args(args=args)
    jmp[args.module].main(args, pass_through_args)
