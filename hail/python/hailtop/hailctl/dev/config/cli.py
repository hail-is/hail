from . import set_property
from . import list_properties

import warnings


def init_parser(config_parser):
    subparsers = config_parser.add_subparsers(title='hailctl dev config subcommand', dest='hailctl dev config subcommand', required=True)

    set_parser = subparsers.add_parser(
        'set',
        help='Set deploy configuration property.',
        description='Set deploy configuration property.')

    set_parser.set_defaults(module='hailctl dev config set')
    set_property.init_parser(set_parser)

    show_parser = subparsers.add_parser(
        'show',
        help='List all dev configuration properties. Note: This subcommand is deprecated. Use `list` instead',
        description='Set deploy configuration property.')

    show_parser.set_defaults(module='hailctl dev config show')
    list_properties.init_parser(show_parser)

    list_parser = subparsers.add_parser(
        'list',
        help='List all dev configuration properties.',
        description='List all dev configuration properties.')

    list_parser.set_defaults(module='hailctl dev config list')
    list_properties.init_parser(list_parser)


def main(args):
    if args.module == 'hailctl dev config set':
        set_property.main(args)
        return

    if args.module == 'hailctl dev config show':
        warnings.warn('The `show` subcommand is deprecated. Use `list` instead.', stacklevel=2)
    else:
        assert args.module == 'hailctl dev config list'

    list_properties.main(args)
