import collections
import json
import logging

import argparse

from .. import init_logging


def main(args_):
    parser = argparse.ArgumentParser()

    parser.add_argument("files",
                        type=str,
                        nargs='*',
                        help="JSON files to summarize.")

    args = parser.parse_args(args_)
    summarize(args.files)


def summarize(files):
    init_logging()
    n_files = len(files)
    if n_files < 1:
        raise ValueError(f"'summarize' requires at least 1 file to summarize")
    logging.info(f'{len(files)} files to summarize')

    for file in files:
        logging.info(f'Summary for {file}:')

        with open(file, 'r') as f:
            data = json.load(f)
        logging.info(f"config: {data['config']}")
        for bm in data['benchmarks']:
            if bm['failed']:
                logging.info(f"benchmark failed: {bm['name']}")
            elif not bm.get('times'):
                logging.info(f"benchmark has no times but is not marked failed: {bm['name']}")
            if bm.get('timed_out'):
                logging.info(f"benchmark timed out: {bm['name']}")
