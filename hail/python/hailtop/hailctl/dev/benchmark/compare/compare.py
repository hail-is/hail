import json
import os
import sys

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

    return {x['name']: x for x in js_data}


def magnitude(time1, time2):
    return max(time1 / time2, time2 / time1)


def fmt_diff(ratio):
    if ratio < 1:
        return f'+{1 / ratio:.3f}'
    else:
        return f'-{ratio:.3f}'

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
        sys.stderr.write(f"Found non-overlapping benchmarks:\n  {list(diff)}")

    comparison = []
    for name in overlap:
        d1 = data1[name]
        d2 = data2[name]
        if d1.get('failed') or d2.get('failed'):
            continue
        run1_med = d1['median']
        run2_med = d2['median']
        comparison.append((name, run1_med, run2_med))

    comparison = sorted(comparison, key=lambda x: x[2] / x[1], reverse=True)

    longest_name = max(len(name) for name, _, _ in comparison)

    comps = []
    for name, r1, r2 in comparison:
        comps.append(r2 / r1)
        print(f'{name:>{longest_name}}   {fmt_diff(r2 / r1):>8}   {fmt_time(r1, 7):>7}   {fmt_time(r2, 7):>7}')

    print('----------------')
    print(f'Average: {fmt_diff(np.mean(comps))}')
    print(f'Median:  {fmt_diff(np.median(comps))}')