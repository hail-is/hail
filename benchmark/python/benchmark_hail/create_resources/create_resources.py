import argparse

from ..run.utils import download_data
from .. import init_logging


def main(args_):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", "-d",
                        type=str,
                        help="Data directory.")

    args = parser.parse_args(args_)

    init_logging()
    download_data(args.data_dir)
