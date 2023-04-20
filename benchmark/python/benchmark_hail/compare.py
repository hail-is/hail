import json
import os
import sys
import math

from scipy.stats.mstats import gmean, hmean
import numpy as np


def load_file(path):
    if path.endswith('.json'):
        with open(path, 'r') as f:
            js_data = json.load(f)
    elif path.endswith('.tsv'):
        import pandas as pd
        js_data = pd.read_table(path).to_json(orient='records')
    else:
        raise ValueError(f'unknown format: {os.path.basename(path)}')

    return {x['name']: x for x in js_data['benchmarks']}


def fmt_mem_ratio(m1, m2):
    if m1 is None or m2 is None or m1 == 0 or m2 == 0:
        return 'NA'
    return fmt_diff(m1 / m2)


def fmt_diff(ratio):
    return f'{ratio * 100:.1f}%'


def fmt_time(x, size):
    return f'{x:.3f}'.rjust(size)


def fmt_mem(b):
    if b is None:
        return 'NA'
    return f'{int(math.ceil(b / (1024 * 1024)))}'


def compare(args):
    run1 = args.run1
    run2 = args.run2

    min_time_for_inclusion = args.min_time

    data1 = load_file(run1)
    data2 = load_file(run2)

    names1 = set(data1.keys())
    names2 = set(data2.keys())
    all_names = names1.union(names2)
    overlap = names1.intersection(names2)
    diff = all_names - overlap

    if diff:
        sys.stderr.write(f"Found non-overlapping benchmarks:" + ''.join(f'\n    {t}' for t in diff) + '\n')

    if args.metric == 'best':
        metric_f = min
    elif args.metric == 'median':
        metric_f = np.median

    def get_time_metric(data):
        return metric_f(data['times'])

    def get_memory_metric(data):
        if 'peak_task_memory' in data:
            return max(data['peak_task_memory'])
        return None

    failed_1 = []
    failed_2 = []
    comparison = []
    for name in overlap:
        d1 = data1[name]
        d2 = data2[name]
        d1_failed = d1.get('failed') or d1.get('times') == []  # rescue bugs in previous versions
        d2_failed = d2.get('failed') or d2.get('times') == []  # rescue bugs in previous versions
        if d1_failed:
            failed_1.append(name)
        if d2_failed:
            failed_2.append(name)
        if d1_failed or d2_failed:
            continue
        try:
            run1_time_metric = get_time_metric(d1)
            run2_time_metric = get_time_metric(d2)
            run1_memory_metric = get_memory_metric(d1)
            run2_memory_metric = get_memory_metric(d2)
        except Exception as e:
            raise ValueError(f"error while computing metric for {name}:\n  d1={d1}\n  d2={d2}") from e
        if run1_time_metric < min_time_for_inclusion and run2_time_metric < min_time_for_inclusion:
            continue

        comparison.append((name, run1_time_metric, run2_time_metric, run1_memory_metric, run2_memory_metric))

    if failed_1:
        sys.stderr.write(f"Failed benchmarks in run 1:" + ''.join(f'\n    {t}' for t in failed_1) + '\n')
    if failed_2:
        sys.stderr.write(f"Failed benchmarks in run 2:" + ''.join(f'\n    {t}' for t in failed_2) + '\n')
    comparison = sorted(comparison, key=lambda x: x[2] / x[1], reverse=True)

    longest_name = max(max(len(t[0]) for t in comparison), len('Benchmark Name'))

    comps = []

    def format(name, ratio, t1, t2, memory_ratio, mem1, mem2):
        return f'{name:>{longest_name}}   {ratio:>8}   {t1:>8}   {t2:>8}   {memory_ratio:>9}   {mem1:>10}   {mem2:>10}'

    print(format('Benchmark Name', 'Ratio', 'Time 1', 'Time 2', 'Mem Ratio', 'Mem 1 (MB)', 'Mem 2 (MB)'))
    print(format('--------------', '-----', '------', '------', '---------', '----------', '----------'))
    for name, r1, r2, m1, m2 in comparison:
        print(format(name,
                     fmt_diff(r2 / r1), fmt_time(r1, 8), fmt_time(r2, 8),
                     fmt_mem_ratio(m2, m1), fmt_mem(m1), fmt_mem(m2)))
        if name.startswith('sentinel'):
            continue
        comps.append(r2 / r1)

    print('----------------------')
    print(f'Harmonic mean: {fmt_diff(hmean(comps))}')
    print(f'Geometric mean: {fmt_diff(gmean(comps))}')
    print(f'Arithmetic mean: {fmt_diff(np.mean(comps))}')
    print(f'Median:  {fmt_diff(np.median(comps))}')


def register_main(subparser) -> 'None':
    parser = subparser.add_parser(
        'compare',
        help='Compare Hail benchmarks.',
        description='Run Hail benchmarks.'
    )
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
                        default='median',
                        choices=['best', 'median'],
                        help='Comparison metric.')
    parser.set_defaults(main=compare)
