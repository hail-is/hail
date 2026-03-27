"""
Tests that require rare GCP instance types (GPU, very high CPU) which may not
be immediately available in a given zone. These are separated from the main
test_batch suite so they run in a dedicated CI shard with a generous timeout.
"""

import pytest

from hailtop.batch_client.client import BatchClient
from hailtop.test_utils import skip_in_azure

from .utils import DOCKER_ROOT_IMAGE, create_batch, xfail_if_infra_failures_only


@skip_in_azure
@pytest.mark.timeout(10 * 60)
async def test_nvidia_driver_accesibility_usage(client: BatchClient):
    b = create_batch(client)._async_batch
    resources = {'machine_type': "g2-standard-4", 'storage': '100Gi'}
    j = b.create_job(
        'pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be',
        [
            '/bin/sh',
            '-c',
            'nvidia-smi && python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"',
        ],
        resources=resources,
        n_max_attempts=4,
    )
    await b.submit()
    status = await j.wait()
    if status['state'] != 'Success':
        xfail_if_infra_failures_only(await j.attempts(), "G2 instances unavailable (ZONE_RESOURCE_POOL_EXHAUSTED)")
    assert status['state'] == 'Success', str((status, b.debug_info()))


@skip_in_azure
@pytest.mark.timeout(10 * 60)
async def test_over_64_cpus(client: BatchClient):
    # This test is being added to validate high CPU counts in custom machines.
    # The relevant part of this machine type ('highmem-96') is the CPU count, which is 96.
    b = create_batch(client)
    resources = {'machine_type': 'n1-highmem-96', 'preemptible': False}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources, n_max_attempts=4)
    b.submit()
    status = j.wait()
    if status['state'] != 'Success':
        xfail_if_infra_failures_only(j.attempts(), "n1-highmem-96 instances unavailable")
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'job-private' in status['status']['worker'], str((status, b.debug_info()))
