import argparse

from ..run.utils import download_data
from .. import init_logging


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
    download_data(args.data_dir, args.group)
