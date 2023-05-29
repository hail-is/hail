from . import init_logging
from .run.resources import all_resources
from .run.utils import ensure_resources, ensure_single_resource


def main(args):
    init_logging()
    if args.group:
        ensure_single_resource(args.data_dir, args.group)
    else:
        ensure_resources(args.data_dir, all_resources)


def register_main(subparser) -> 'None':
    parser = subparser.add_parser(
        'create-resources',
        help='Create benchmark input resources.',
        description='Create benchmark input resources.'
    )

    parser.add_argument("--data-dir", "-d",
                        type=str,
                        required=True,
                        help="Data directory.")
    parser.add_argument("--group",
                        type=str,
                        required=False,
                        help="Resource group to download.")

    parser.set_defaults(main=main)
