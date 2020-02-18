import os
import json
from hailtop.config import get_deploy_config


def init_parser(parser):
    parser.add_argument("service", type=str,
                        help="The service for which to look up the url.")


def main(args):
    deploy_config = get_deploy_config()
    print(deploy_config.base_url(args.service), end='')
