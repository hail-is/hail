import logging
import re

from hailtop.aiocloud import aiogoogle

from .config import BENCHMARK_RESULTS_PATH

log = logging.getLogger('benchmark')

BENCHMARK_BUCKETS = ['hail-benchmarks', 'hail-benchmarks-2']

FILE_PATH_REGEX = re.compile(r'gs://((?P<bucket>[^/]+)/)(?P<path>.*)')


def get_geometric_mean(prod_of_means, num_of_means):
    return prod_of_means ** (1.0 / num_of_means)


def round_if_defined(x):
    if x is not None:
        return round(x, 6)
    return None


def parse_file_path(regex, name):
    match = regex.fullmatch(name)
    return match.groupdict()


def enumerate_list_of_trials(list_of_trials):
    trial_indices = []
    wall_times = []
    within_group_idx = []
    for count, trial in enumerate(list_of_trials):
        wall_times.extend(trial)
        within_group_idx.extend([f'{j+1}' for j in range(len(trial))])
        temp = [count] * len(trial)
        trial_indices.extend(temp)
    res_dict = {'trial_indices': trial_indices, 'wall_times': wall_times, 'within_group_index': within_group_idx}
    return res_dict


async def list_benchmark_files(fs: aiogoogle.GoogleStorageAsyncFS):
    list_of_files = []
    for bucket in BENCHMARK_BUCKETS:
        files = await fs.listfiles(f'gs://{bucket}/', recursive=True)
        list_of_files.extend(files)
    return list_of_files


async def submit_test_batch(batch_client, sha):
    batch = batch_client.create_batch(attributes={'sha': sha})
    known_file_path = 'gs://hail-benchmarks-2/tpoterba/0.2.21-f6f337d1e9bb.json'
    dest_file_path = f'{BENCHMARK_RESULTS_PATH}/0-{sha}.json'
    job = batch.create_job(
        image='ubuntu:20.04',
        command=['/bin/bash', '-c', 'touch /io/test; sleep 5'],
        resources={'cpu': '0.25'},
        input_files=[(known_file_path, '/io/test')],
        output_files=[('/io/test', dest_file_path)],
    )
    await batch.submit(disable_progress_bar=True)
    log.info(f'submitting batch for commit {sha}')
    return job.batch_id
