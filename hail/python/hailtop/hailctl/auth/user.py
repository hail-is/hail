import json
import sys

from hailtop.auth import get_userinfo


def init_parser(parser):  # pylint: disable=unused-argument
    pass


def main(args, pass_through_args):  # pylint: disable=unused-argument
    userinfo = get_userinfo()
    if userinfo is None:
        print('not logged in')
        sys.exit(1)
    result = {
        'username': userinfo['username'],
        'email': userinfo['login_id'],  # deprecated - backwards compatibility
        'gsa_email': userinfo['hail_identity'],  # deprecated - backwards compatibility
        'hail_identity': userinfo['hail_identity'],
        'login_id': userinfo['login_id'],
        'display_name': userinfo['display_name'],
    }
    print(json.dumps(result, indent=4))
