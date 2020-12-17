import sys
import os
import argparse

from hailtop.auth import namespace_auth_headers
from hailtop.config import get_deploy_config


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'curl',
        help="Issue authenticated curl requests to Hail infrastructure.",
        description="Issue authenticated curl requests to Hail infrastructure.")
    parser.set_defaults(module='hailctl curl')
    parser.add_argument('namespace',
                        help="Target namespace for request")
    parser.add_argument('service',
                        help="Target service for request")
    parser.add_argument('path',
                        help="Path to request")
    parser.add_argument('curl_args',
                        nargs=argparse.REMAINDER,
                        help="Additional arguments to pass to curl")


def main(args):
    deploy_config = get_deploy_config()
    deploy_config = deploy_config.with_default_namespace(args.namespace)
    headers = namespace_auth_headers(deploy_config, args.namespace)
    headers = [x
               for k, v in headers.items()
               for x in ['-H', f'{k}: {v}']]
    path = deploy_config.url(args.service, args.path)
    os.execvp('curl', ['curl', *headers, *args.curl_args, path])
