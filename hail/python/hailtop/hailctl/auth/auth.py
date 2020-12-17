import sys
import argparse

from . import login
from . import logout
from . import auth_list
from . import copy_paste_login
from . import user


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'auth',
        help='Manage Hail credentials.',
        description='Manage Hail credentials.')
    subparsers = parser.add_subparsers(
        title='hailctl auth subcommand',
        dest='hailctl auth subcommand',
        required=True)

    login.init_parser(subparsers)
    logout.init_parser(subparsers)
    auth_list.init_parser(subparsers)
    copy_paste_login.init_parser(subparsers)
    user.init_parser(subparsers)


def main(args):
    if args.module.startswith('hailctl auth login'):
        login.main(args)
    elif args.module.startswith('hailctl auth logout'):
        logout.main(args)
    elif args.module.startswith('hailctl auth list'):
        auth_list.main(args)
    elif args.module.startswith('hailctl auth copy-paste-login'):
        copy_paste_login.main(args)
    else:
        args.module.startswith('hailctl auth user')
        user.main(args)
