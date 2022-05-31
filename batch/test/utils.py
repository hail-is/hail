import os

import pytest


def batch_status_job_counter(batch_status, job_state):
    return len([j for j in batch_status['jobs'] if j['state'] == job_state])


def legacy_batch_status(batch):
    status = batch.status()
    status['jobs'] = list(batch.jobs())
    return status


def smallest_machine_type(cloud):
    if cloud == 'gcp':
        return 'n1-standard-1'
    assert cloud == 'azure'
    return 'Standard_D2ds_v4'


fails_in_azure = pytest.mark.xfail(
    os.environ.get('HAIL_CLOUD') == 'azure', reason="doesn't yet work on azure", strict=True
)

skip_in_azure = pytest.mark.skipif(
    os.environ.get('HAIL_CLOUD') == 'azure', reason="not applicable to azure", strict=True
)

fails_in_gcp_storage = pytest.mark.xfail(
    os.environ.get('HAIL_CLOUD') == 'gcp', reason="Batch is temporarily rejecting jobs with extra storage requests", strict=True
)
