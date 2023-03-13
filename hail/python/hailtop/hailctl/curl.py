import sys
import os

from hailtop.auth import hail_credentials
from hailtop.config import get_deploy_config
from hailtop.utils import async_to_blocking


def main(args):
    if len(args) < 3:
        print('hailctl curl NAMESPACE SERVICE PATH [args] ...', file=sys.stderr)
        sys.exit(1)
    ns = args[0]
    svc = args[1]
    path = args[2]
    headers_dict = async_to_blocking(hail_credentials(namespace=ns).auth_headers())
    headers = [x
               for k, v in headers_dict.items()
               for x in ['-H', f'{k}: {v}']]
    path = get_deploy_config().url(svc, path)
    os.execvp('curl', ['curl', *headers, *args[3:], path])
