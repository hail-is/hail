import argparse

from . import config
from . import deploy
from . import query


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'dev',
        help='Developer tools.',
        description='Developer tools.')
    subparsers = parser.add_subparsers(
        title='hailctl dev subcommand',
        dest='hailctl dev subcommand',
        required=True)

    config.init_parser(subparsers)
    deploy.init_parser(subparsers)
    query.init_parser(subparsers)


def main(args):
    if args.module.startswith('hailctl dev deploy'):
        deploy.main(args)
    elif args.module.startswith('hailctl dev config'):
        config.main(args)
    else:
        assert args.module.startswith('hailctl dev query')
        query.main(args)
