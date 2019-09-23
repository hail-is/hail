import argparse

from .compare import compare


def main(args_):
    parser = argparse.ArgumentParser()

    parser.add_argument('run1',
                        type=str,
                        help='First benchmarking run.')
    parser.add_argument('run2',
                        type=str,
                        help='Second benchmarking run.')

    args = parser.parse_args(args_)

    compare(args.run1, args.run2)
