import argparse

from . import config
from . import deploy


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

    return main_parser


def main(args):
    p = parser()
    args = p.parse_args()
    if args.module == 'deploy':
        from .deploy import cli as deploy_cli  # pylint: disable=import-outside-toplevel
        deploy_cli.main(args)
    else:
        prefix = 'hailctl dev config'
        assert args.module[:len(prefix)] == prefix
        from .config import cli as config_cli  # pylint: disable=import-outside-toplevel
        config_cli.main(args)
