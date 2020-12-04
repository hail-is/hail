from . import set_property
from . import show


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
        help='Set deploy configuration property.',
        description='Set deploy configuration property.')

    show_parser.set_defaults(module='hailctl dev config show')
    show.init_parser(show_parser)


def main(args):
    if args.module == 'hailctl dev config set':
        set_property.main(args)
        return

    assert args.module == 'hailctl dev config show'
    show.main(args)
