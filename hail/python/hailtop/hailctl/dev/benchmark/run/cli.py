import json
import sys

import argparse
import datetime

import hail as hl
from .utils import initialize, run_all, run_pattern, run_list, RunConfig


def main(args_):
    parser = argparse.ArgumentParser()

    parser.add_argument('--tests', '-t',
                        type=str,
                        required=False,
                        help='Run specific comma-delimited tests instead of running all tests.')
    parser.add_argument('--cores', '-c',
                        type=int,
                        default=1,
                        help='Number of cores to use.')
    parser.add_argument('--pattern', '-k', type=str, required=False,
                        help='Run all tests that substring match the pattern')
    parser.add_argument("--n-iter", "-n",
                        type=int,
                        default=3,
                        help='Number of iterations for each test.')
    parser.add_argument("--log", "-l",
                        type=str,
                        help='Log file path')
    parser.add_argument("--verbose", "-v",
                        action="store_true",
                        help="Print testing information to stderr in real time.")
    parser.add_argument("--output", "-o",
                        type=str,
                        help="Output file path.")
    parser.add_argument("--data-dir", "-d",
                        type=str,
                        help="Data directory.")

    args = parser.parse_args(args_)

    initialize(args)

    run_data = {'cores': args.cores,
                'version': hl.__version__,
                'timestamp': str(datetime.datetime.now()),
                'system': sys.platform}

    records = []

    def handler(stats):
        records.append(stats)

    config = RunConfig(args.n_iter, handler, args.verbose)
    if args.tests:
        run_list(args.tests.split(','), config)
    if args.pattern:
        run_pattern(args.pattern, config)
    if not args.pattern and not args.tests:
        run_all(config)

    data = {'config': run_data,
            'benchmarks': records}
    if args.output:
        with open(args.output, 'w') as out:
            json.dump(data, out)
    else:
        print(json.dumps(data))
