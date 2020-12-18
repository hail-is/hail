import sys
import json

from hailtop.auth import get_userinfo

from .auth import auth


@auth.command(
    help="Get Hail user information.")
def user():
    userinfo = get_userinfo()
    if userinfo is None:
        print("Not logged in.", file=sys.stderr)
        sys.exit(1)
    result = {
        'username': userinfo['username'],
        'email': userinfo['email'],
        'gsa_email': userinfo['gsa_email']
    }
    print(json.dumps(result, indent=4))
