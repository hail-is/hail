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
    if normalize:
        results = (results - results.mean()) / results.mean()

    if head is not None:
        results = results[
            (results.max() - results.min()) \
                .sort_values(ascending=False) \
                .head(head) \
                .keys()
        ]

    results.T.sort_index().plot.bar()
    plt.show()


def main(args) -> 'None':
    results = collect_results(args.files, args.metric)
    plot(results, not (len(args.files) == 1 and args.raw), args.head)


def register_main(subparser) -> 'None':
    parser = subparser.add_parser('visualize',
        description='Visualize benchmark results',
        help='Graphically compare one or more benchmark results'
    )
    parser.add_argument('files', nargs='+', help='benchmark results to visualize')
    parser.add_argument('--metric', choices=['mean','median','stdev','max_memory'], default='mean')
    parser.add_argument('--head', type=int, help="number of most significant results to take")
    parser.add_argument('--raw', action='store_true', help="do not compute difference from mean")
    parser.set_defaults(main=main)
