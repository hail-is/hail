import json

from hailtop.auth import get_userinfo


def init_parser(parser):  # pylint: disable=unused-argument
    pass


def main(args, pass_through_args):  # pylint: disable=unused-argument
    userinfo = get_userinfo()
    result = {
        'username': userinfo['username'],
        'email': userinfo['email'],
        'gsa_email': userinfo['gsa_email']
    }
    print(json.dumps(result, indent=4))
