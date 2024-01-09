import sys

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from .run import cli as run
from . import compare, create_resources, combine, summarize, visualize

def main(argv=sys.argv[1:]):
    programs = [
        run.register_main,
        compare.register_main,
        create_resources.register_main,
        combine.register_main,
        summarize.register_main,
        visualize.register_main
    ]

    parser = ArgumentParser(
        prog='hail-bench',
        description='Run and analyze Hail benchmarks.',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers()
    for p in programs:
        p(subparsers)

    args = parser.parse_args(argv)
    args.main(args)

if __name__ == '__main__':
    main()
