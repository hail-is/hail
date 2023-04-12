import os

from hailtop import pip_version

DOCKER_ROOT_IMAGE = os.environ.get('DOCKER_ROOT_IMAGE', 'ubuntu:20.04')
HAIL_GENETICS_HAIL_IMAGE = os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{pip_version()}')


def batch_status_job_counter(batch_status, job_state):
    return len([j for j in batch_status['jobs'] if j['state'] == job_state])


def legacy_batch_status(batch):
    status = batch.status()
    status['jobs'] = list(batch.jobs())
    return status


def smallest_machine_type():
    cloud = os.environ['HAIL_CLOUD']

    if cloud == 'gcp':
        return 'n1-standard-1'
    assert cloud == 'azure'
    return 'Standard_D2ds_v4'
