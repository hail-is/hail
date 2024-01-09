import collections
import json
import logging

from . import init_logging


def combine(output, files):
    init_logging()
    logging.info(f'Writing combine output to {output}')
    n_files = len(files)
    if n_files < 1:
        raise ValueError(f"'combine' requires at least 1 file to merge")
    logging.info(f'{len(files)} files to merge')

    config = None
    benchmark_data = collections.defaultdict(lambda: {'failed': False, 'trials': [], 'peak_task_memory': []})

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        config = config or data['config']  # take first config; should be similar
        for bm in data['benchmarks']:
            bm_data = benchmark_data[bm['name']]
            if bm['failed']:
                bm_data['failed'] = True
            else:
                bm_data['trials'].append(bm['times'])
                if 'peak_task_memory' in bm:
                    bm_data['peak_task_memory'].append(bm['peak_task_memory'])

    import numpy as np
    import scipy.stats as stats

    benchmark_json = []
    for name, data in benchmark_data.items():
        data['name'] = name
        if not data['failed']:
            flat_times = [t for trial in data['trials'] for t in trial]
            data['times'] = flat_times
            data['median'] = np.median(flat_times)
            data['mean'] = np.mean(flat_times)
            data['stdev'] = np.std(flat_times)
            flat_peak_memory = [m for memory_list in data['peak_task_memory'] for m in memory_list]
            data['peak_task_memory'] = flat_peak_memory
            if len(flat_peak_memory) > 0:
                data['max_memory'] = max(flat_peak_memory)
            if len(data['trials']) > 1:
                f_stat, p_value = stats.f_oneway(*data['trials'])
                data['f-stat'] = f_stat
                data['p-value'] = p_value
                if p_value < 0.001:
                    logging.warning(
                        f'benchmark {name} had significantly different trial distributions (p={p_value}, F={f_stat}):'
                        + ''.join('\n  ' + ', '.join([f'{x:.2f}s' for x in trial]) for trial in data['trials'])
                    )
            else:
                data['f-stat'] = float('nan')
                data['p-value'] = float('nan')

        benchmark_json.append(data)

    with open(output, 'w') as out:
        json.dump({'config': config, 'benchmarks': benchmark_json}, out)


def register_main(subparser) -> 'None':
    parser = subparser.add_parser(
        'combine', help='Combine parallelized benchmark metrics.', description='Combine parallelized benchmark metrics.'
    )
    parser.add_argument("--output", "-o", type=str, required=True, help="Output file.")
    parser.add_argument("files", type=str, nargs='*', help="JSON files to çombine.")
    parser.set_defaults(main=lambda args: combine(args.output, args.files))
