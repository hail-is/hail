import os
import requests

def init_parser(parser):
    pass

def main(args, pass_through_args):
    token_file = (os.environ.get('HAIL_TOKEN_FILE')
                  or os.path.expanduser('~/.hail/token'))

    if not os.path.exists(token_file):
        raise ValueError(f'No token file found at {token_file}')
    with open(token_file) as f:
        token = f.read()

    requests.post('http://auth.hail.is/api/v1alpha/logout',
                  headers={'Authorization': f'Bearer {token}'})
    os.remove(token_file)

    print('Logged out.')
