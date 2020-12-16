import argparse

from . import config
from . import deploy
from . import query


def parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl',
        description='Manage Hail development utilities.')
    # we have to set dest becuase of a rendering bug in argparse
    # https://bugs.python.org/issue29298
    main_subparsers = main_parser.add_subparsers(title='hailctl subcommand', dest='hailctl subcommand', required=True)

    dev_parser = main_subparsers.add_parser(
        'dev',
        help='Developer tools.',
        description='Developer tools.')
    subparsers = dev_parser.add_subparsers(title='hailctl dev subcommand', dest='hailctl dev subcommand', required=True)

    config_parser = subparsers.add_parser(
        'config',
        help='Configure deployment',
        description='Configure deployment')

    config.cli.init_parser(config_parser)

    deploy_parser = subparsers.add_parser(
        'deploy',
        help='Deploy a branch',
        description='Deploy a branch')
    deploy_parser.set_defaults(module='deploy')
    deploy.cli.init_parser(deploy_parser)

    query_parser = subparsers.add_parser(
        'query',
        help='Set dev settings on query service',
        description='Set dev settings on query service')
    query_parser.set_defaults(module='query')
    query.cli.init_parser(query_parser)

    return main_parser


def main(args):
    p = parser()
    args = p.parse_args()
    if args.module == 'deploy':
        from .deploy import cli  # pylint: disable=import-outside-toplevel
        cli.main(args)
    elif args.module.startswith('hailctl dev config'):
        from .config import cli  # pylint: disable=import-outside-toplevel
        cli.main(args)
    else:
        assert args.module == 'query'
        from .query import cli  # pylint: disable=import-outside-toplevel
        cli.main(args)
