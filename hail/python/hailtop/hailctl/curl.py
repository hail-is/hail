import sys
import os

from hailtop.auth import namespace_auth_headers
from hailtop.config import get_deploy_config


def main(args):
    if len(args) < 3:
        print('hailctl curl NAMESPACE SERVICE PATH [args] ...', file=sys.stderr)
        sys.exit(1)
    ns = args[0]
    svc = args[1]
    path = args[2]
    deploy_config = get_deploy_config()
    deploy_config = deploy_config.with_default_namespace(ns)
    headers_dict = namespace_auth_headers(deploy_config, ns)
    headers = [x
               for k, v in headers_dict.items()
               for x in ['-H', f'{k}: {v}']]
    path = deploy_config.url(svc, path)
    os.execvp('curl', ['curl', *headers, *args[3:], path])
