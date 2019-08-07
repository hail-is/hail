import json
import os
import sys

from scipy.stats.mstats import gmean
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


def fmt_diff(ratio):
    return f'{ratio * 100:.1f}%'


def fmt_time(x, size):
    return f'{x:.3f}'.rjust(size)


def compare(run1, run2):
    data1 = load_file(run1)
    data2 = load_file(run2)

    names1 = set(data1.keys())
    names2 = set(data2.keys())
    all_names = names1.union(names2)
    overlap = names1.intersection(names2)
    diff = all_names - overlap

    if diff:
        sys.stderr.write(f"Found non-overlapping benchmarks:" + ''.join(f'\n    {t}' for t in diff) + '\n')

    failed_1 = []
    failed_2 = []
    comparison = []
    for name in overlap:
        d1 = data1[name]
        d2 = data2[name]
        d1_failed = d1.get('failed')
        d2_failed = d2.get('failed')
        if d1_failed:
            failed_1.append(name)
        if d2_failed:
            failed_2.append(name)
        if d1_failed or d2_failed:
            continue
        run1_med = d1['median']
        run2_med = d2['median']
        comparison.append((name, run1_med, run2_med))

    if failed_1:
        sys.stderr.write(f"Failed benchmarks in run 1:" + ''.join(f'\n    {t}' for t in failed_1) + '\n')
    if failed_2:
        sys.stderr.write(f"Failed benchmarks in run 2:" + ''.join(f'\n    {t}' for t in failed_2) + '\n')
    comparison = sorted(comparison, key=lambda x: x[2] / x[1], reverse=True)

    longest_name = max(len(name) for name, _, _ in comparison)

    comps = []

    def format(name, ratio, t1, t2):
        return f'{name:>{longest_name}}   {ratio:>8}   {t1:>7}   {t2:>7}'

    print(format('Name', 'Ratio', 'Time 1', 'Time 2'))
    print(format('----', '-----', '------', '------'))
    for name, r1, r2 in comparison:
        comps.append(r2 / r1)
        print(format(name, fmt_diff(r2 / r1), fmt_time(r1, 7), fmt_time(r2, 7)))

    print('----------------------')
    print(f'Geometric mean: {fmt_diff(gmean(comps))}')
    print(f'Simple mean: {fmt_diff(np.mean(comps))}')
    print(f'Median:  {fmt_diff(np.median(comps))}')
