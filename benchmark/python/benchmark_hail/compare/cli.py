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
    parser.add_argument('--min-time',
                        type=float,
                        default=1.0,
                        help='Minimum runtime in either run for inclusion.')
    parser.add_argument('--metric',
                        type=str,
                        default='best',
                        choices=['best', 'median'],
                        help='Comparison metric.')


    args = parser.parse_args(args_)

    compare(args)
