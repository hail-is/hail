import json
import logging
from argparse import ArgumentParser

from . import init_logging


def summarize(files):
    init_logging()
    n_files = len(files)
    if n_files < 1:
        raise ValueError("'summarize' requires at least 1 file to summarize")
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


if __name__ == '__main__':
    parser = ArgumentParser(
        'summarize',
        description='Summarize a benchmark json results file',
    )
    parser.add_argument("files", type=str, nargs='*', help="JSON files to summarize.")
    argv = parser.parse_args()
    summarize(argv.files)
