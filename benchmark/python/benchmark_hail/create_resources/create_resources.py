import argparse

from .. import init_logging
from ..run.resources import all_resources
from ..run.utils import ensure_resources, ensure_single_resource


def main(args_):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", "-d",
                        type=str,
                        required=True,
                        help="Data directory.")
    parser.add_argument("--group",
                        type=str,
                        required=False,
                        help="Resource group to download.")

    args = parser.parse_args(args_)

    init_logging()
    if args.group:
        ensure_single_resource(args.data_dir, args.group)
    else:
        ensure_resources(args.data_dir, all_resources)
