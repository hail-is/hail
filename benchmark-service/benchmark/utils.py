from google.cloud import storage
import re
import logging
from .config import HAIL_BENCHMARK_BUCKET_NAME

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
    res_dict = {
        'trial_indices': trial_indices,
        'wall_times': wall_times,
        'within_group_index': within_group_idx
    }
    return res_dict


def list_benchmark_files(read_gs):
    list_of_files = []
    for bucket in BENCHMARK_BUCKETS:
        list_of_files.extend(read_gs.list_files(bucket_name=bucket))
    return list_of_files


async def submit_test_batch(batch_client, sha):
    batch = batch_client.create_batch()
    job = batch.create_job(image='ubuntu:18.04',
                           command=['/bin/bash', '-c', 'touch /io/test; sleep 5'],
                           resources={'cpu': '0.25'},
                           output_files=[('/io/test', f'gs://{HAIL_BENCHMARK_BUCKET_NAME}/benchmark-test/{sha}.json')])
    await batch.submit(disable_progress_bar=True)
    log.info(f'submitted batch for commit {sha}')
    return job.batch_id


class ReadGoogleStorage:
    def __init__(self, service_account_key_file=None):
        self.storage_client = storage.Client.from_service_account_json(service_account_key_file)

    def get_data_as_string(self, file_path):
        file_info = parse_file_path(FILE_PATH_REGEX, file_path)
        bucket = self.storage_client.get_bucket(file_info['bucket'])
        shorter_file_path = file_info['path']
        try:
            # get bucket data as blob
            blob = bucket.blob(shorter_file_path)
            # convert to string
            data = blob.download_as_string()
        except Exception as e:
            raise NameError() from e
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
        shorter_file_path = file_info['path']
        exists = storage.Blob(bucket=bucket, name=shorter_file_path).exists()
        log.info(f'file {shorter_file_path} in bucket {bucket_name} exists? {exists}')
        return exists
