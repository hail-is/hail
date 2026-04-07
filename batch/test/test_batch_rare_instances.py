"""
Tests that require rare GCP instance types (GPU, very high CPU) which may not
be immediately available in a given zone. These are separated from the main
test_batch suite so they run in a dedicated CI shard with a generous timeout.
"""

import pytest

from hailtop.batch_client.client import BatchClient
from hailtop.test_utils import skip_in_azure

from .utils import DOCKER_ROOT_IMAGE, create_batch


def _run_on_rare_instance(client: BatchClient, image: str, command, resources: dict, xfail_message: str) -> dict:
    b = create_batch(client)
    j = b.create_job(image, command, resources=resources, n_max_attempts=4)
    b.submit()
    status = j.wait()
    if status['state'] != 'Success':
        infra_only_reasons = {'does_not_exist', 'terminated', 'preempted'}
        real_reasons = {a.get('reason', 'unknown') for a in j.attempts() if a.get('instance_name')}
        if real_reasons and real_reasons.issubset(infra_only_reasons):
            pytest.xfail(f"{xfail_message}: all attempts ended with {sorted(real_reasons)}")
    assert status['state'] == 'Success', str((status, b.debug_info()))
    return status


@skip_in_azure
@pytest.mark.timeout(10 * 60)
def test_nvidia_driver_accesibility_usage(client: BatchClient):
    _run_on_rare_instance(
        client,
        'pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be',
        [
            '/bin/sh',
            '-c',
            'nvidia-smi && python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"',
        ],
        {'machine_type': 'g2-standard-4', 'storage': '100Gi'},
        "G2 instances unavailable (ZONE_RESOURCE_POOL_EXHAUSTED)",
    )


@skip_in_azure
@pytest.mark.timeout(10 * 60)
def test_over_64_cpus(client: BatchClient):
    # The relevant part of this machine type ('highmem-96') is the CPU count, which is 96.
    status = _run_on_rare_instance(
        client,
        DOCKER_ROOT_IMAGE,
        ['true'],
        {'machine_type': 'n1-highmem-96', 'preemptible': False},
        "n1-highmem-96 instances unavailable",
    )
    assert 'job-private' in status['status']['worker'], status
