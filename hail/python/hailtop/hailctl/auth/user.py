import json
import sys

from hailtop.auth import get_userinfo


def init_parser(parent_subparsers):
    user_parser = parent_subparsers.add_parser(
        'user',
        help='Get Hail user information.',
        description='Get Hail user information')
    user_parser.set_defaults(module='hailctl auth user')


def main(args):
    userinfo = get_userinfo()
    if userinfo is None:
        print('not logged in')
        sys.exit(1)
    result = {
        'username': userinfo['username'],
        'email': userinfo['email'],
        'gsa_email': userinfo['gsa_email']
    }
    print(json.dumps(result, indent=4))
