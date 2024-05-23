import inspect
import os
from typing import Union, overload

import hailtop.batch_client.aioclient as aiobc
import hailtop.batch_client.client as bc
from hailtop import pip_version

DOCKER_ROOT_IMAGE = os.environ.get('DOCKER_ROOT_IMAGE', 'ubuntu:22.04')
HAIL_GENETICS_HAIL_IMAGE = os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{pip_version()}')


@overload
def create_batch(client: bc.BatchClient, **kwargs) -> bc.Batch: ...


@overload
def create_batch(client: aiobc.BatchClient, **kwargs) -> aiobc.Batch: ...


def create_batch(client: Union[bc.BatchClient, aiobc.BatchClient], **kwargs) -> Union[bc.Batch, aiobc.Batch]:
    name_of_test_method = inspect.stack()[1][3]
    attrs = kwargs.pop('attributes', {})  # Tests should be able to override the name
    attributes = {'name': name_of_test_method, **attrs}
    this_batch_id = os.environ.get('HAIL_BATCH_ID')
    this_job_id = os.environ.get('HAIL_JOB_ID')
    if this_batch_id and this_job_id:
        attributes['client_job'] = f'{this_batch_id}-{this_job_id}'
    return client.create_batch(attributes, **kwargs)


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
