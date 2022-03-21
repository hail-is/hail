import sys
import argparse

from . import login
from . import logout
from . import auth_list
from . import copy_paste_login
from . import user
from . import create_user
from . import delete_user


def parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl auth',
        description='Manage Hail credentials.')
    subparsers = main_parser.add_subparsers()

    login_parser = subparsers.add_parser(
        'login',
        help='Obtain Hail credentials.',
        description='Obtain Hail credentials.')
    copy_paste_login_parser = subparsers.add_parser(
        'copy-paste-login',
        help='Obtain Hail credentials with a copy paste token.',
        description='Obtain Hail credentials with a copy paste token.')
    logout_parser = subparsers.add_parser(
        'logout',
        help='Revoke Hail credentials.',
        description='Revoke Hail credentials.')
    list_parser = subparsers.add_parser(
        'list',
        help='List Hail credentials.',
        description='List Hail credentials.')
    user_parser = subparsers.add_parser(
        'user',
        help='Get Hail user information.',
        description='Get Hail user information.'
    )
    create_user_parser = subparsers.add_parser(
        'create-user',
        help='Create a new Hail user.',
        description='Create a new Hail user.'
    )
    delete_user_parser = subparsers.add_parser(
        'delete-user',
        help='Delete a Hail user.',
        description='Delete a Hail user.'
    )

    login_parser.set_defaults(module='login')
    login.init_parser(login_parser)

    copy_paste_login_parser.set_defaults(module='copy-paste-login')
    copy_paste_login.init_parser(copy_paste_login_parser)

    logout_parser.set_defaults(module='logout')
    logout.init_parser(logout_parser)

    list_parser.set_defaults(module='list')
    auth_list.init_parser(list_parser)

    user_parser.set_defaults(module='user')
    user.init_parser(user_parser)

    create_user_parser.set_defaults(module='create-user')
    create_user.init_parser(create_user_parser)

    delete_user_parser.set_defaults(module='delete-user')
    delete_user.init_parser(delete_user_parser)

    return main_parser


def main(args):
    if not args:
        parser().print_help()
        sys.exit(0)
    jmp = {
        'login': login,
        'copy-paste-login': copy_paste_login,
        'logout': logout,
        'list': auth_list,
        'user': user,
        'create-user': create_user,
        'delete-user': delete_user,
    }

    args, pass_through_args = parser().parse_known_args(args=args)
    jmp[args.module].main(args, pass_through_args)
