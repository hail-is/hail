import argparse
import sys
import json
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
    parser.add_argument("--format", "-f",
                        type=str,
                        help="Output format.",
                        default='table',
                        choices=['table', 'json', 'text'])

    args = parser.parse_args(args_)

    initialize(args)

    finalizers = []
    if args.output:
        out_file = open(args.output, 'w')
        finalizers.append(lambda: out_file.close())
    else:
        out_file = None
    writer = lambda s: print(s, end='', file=out_file)

    run_data = {'cores': args.cores}
    if args.format == 'table':
        writer(f'#{json.dumps(run_data, separators=(",", ":"))}\n')
        writer('Name\tMean\tMedian\tStDev\n')

        def handler(stats):
            writer(f'{stats["name"]}\t{stats["mean"]:.3f}\t{stats["median"]:.3f}\t{stats["stdev"]:.3f}\n')
    elif args.format == 'json':
        records = []

        def handler(stats):
            records.append(stats)

        def write_json():
            with open(args.output, 'w') as out:
                json.dump(records, out)

        finalizers.append(write_json)
    else:
        assert args.format == 'text'
        writer(f'Run data:\n')
        for k, v in run_data.items():
            writer(f'    {k}: {v}\n')

        def handler(stats):
            writer(f'{stats["name"]}:\n')
            for t in stats["times"]:
                writer(f'    {t:.2f}s\n')
            writer(f'    Mean {stats["mean"]:.2f}, Median {stats["median"]:.2f}\n')

    config = RunConfig(args.n_iter, handler, args.verbose)
    if args.tests:
        run_list(args.tests, config)
    if args.pattern:
        run_pattern(args, config)
    if not args.pattern and not args.tests:
        run_all(config)

    for f in finalizers:
        f()
