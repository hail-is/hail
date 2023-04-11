import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict
from .compare import load_file
from typing import List, Optional


def collect_results(files: 'List[str]', metric: 'str') -> 'pd.DataFrame':
    results = defaultdict(lambda: [None] * len(files))
    for k, file in enumerate(files):
        for name, time in load_file(file).items():
            results[name][k] = time[metric] if not time['failed'] else None

    return pd.DataFrame(data=results, index=files)


def plot(results: 'pd.DataFrame', normalize: 'bool', head: 'Optional[int]') -> None:
    if len(results.index) > 1:
        r_ = results.iloc[1:] - results.iloc[0]
        results = r_ / results.iloc[0] if normalize else r_

    if head is not None:
        results = results[
            results.abs().max() \
                .sort_values(ascending=False) \
                .head(head) \
                .keys()
        ]

    results.T.sort_index().plot.bar()
    plt.show()


def main(args) -> 'None':
    files = [args.baseline] + args.runs
    results = collect_results(files, args.metric)
    plot(results, not args.raw, args.head)


def register_main(subparser) -> 'None':
    parser = subparser.add_parser('visualize',
        description='Visualize benchmark results',
        help='Graphically compare zero or more benchmark results against a datum'
    )
    parser.add_argument('baseline', help='baseline benchmark results')
    parser.add_argument('runs', nargs='*', help='benchmarks to compare against baseline')
    parser.add_argument('--metric', choices=['mean','median','stdev','max_memory'], default='mean')
    parser.add_argument('--head', type=int, help="number of most significant results to take")
    parser.add_argument('--raw', action='store_true', help="do not normalize results")
    parser.set_defaults(main=main)
