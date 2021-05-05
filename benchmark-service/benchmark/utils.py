from google.cloud import storage
import re
import logging
from .config import BENCHMARK_RESULTS_PATH
import google


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


def list_benchmark_files(read_gs):
    list_of_files = []
    for bucket in BENCHMARK_BUCKETS:
        list_of_files.extend(read_gs.list_files(bucket_name=bucket))
    return list_of_files


async def submit_test_batch(batch_client, sha):
    batch = batch_client.create_batch(attributes={'sha': sha})
    known_file_path = 'gs://hail-benchmarks-2/tpoterba/0.2.21-f6f337d1e9bb.json'
    dest_file_path = f'{BENCHMARK_RESULTS_PATH}/0-{sha}.json'
    job = batch.create_job(
        image='ubuntu:18.04',
        command=['/bin/bash', '-c', 'touch /io/test; sleep 5'],
        resources={'cpu': '0.25'},
        input_files=[(known_file_path, '/io/test')],
        output_files=[('/io/test', dest_file_path)],
    )
    await batch.submit(disable_progress_bar=True)
    log.info(f'submitting batch for commit {sha}')
    return job.batch_id


class ReadGoogleStorage:
    def __init__(self, service_account_key_file=None):
        self.storage_client = storage.Client.from_service_account_json(service_account_key_file)

    def get_data_as_string(self, file_path):
        file_info = parse_file_path(FILE_PATH_REGEX, file_path)
        bucket = self.storage_client.get_bucket(file_info['bucket'])
        path = file_info['path']
        try:
            blob = bucket.blob(path)
            data = blob.download_as_string()
        except google.api_core.exceptions.NotFound as e:
            log.exception(f'error while reading file {file_path}: {e}')
            data = None
        return data

    def list_files(self, bucket_name):
        list_of_files = []
        bucket = self.storage_client.get_bucket(bucket_name)
        for blob in bucket.list_blobs():
            list_of_files.append('gs://' + bucket_name + '/' + blob.name)
        return list_of_files

    def file_exists(self, file_path):
        file_info = parse_file_path(FILE_PATH_REGEX, file_path)
        bucket_name = file_info['bucket']
        bucket = self.storage_client.bucket(bucket_name)
        path = file_info['path']
        exists = storage.Blob(bucket=bucket, name=path).exists()
        log.info(f'file {bucket_name}/{path} in bucket {bucket_name} exists? {exists}')
        return exists

    def delete_file(self, file_path):
        file_info = parse_file_path(FILE_PATH_REGEX, file_path)
        bucket_name = file_info['bucket']
        bucket = self.storage_client.bucket(bucket_name)
        path = file_info['path']
        storage.Blob(bucket=bucket, name=path).delete()
