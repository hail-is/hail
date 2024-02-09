import collections
import os
import secrets
import time
from typing import Set

import orjson
import pytest

from hailtop import httpx
from hailtop.auth import get_userinfo, hail_credentials
from hailtop.batch.backend import HAIL_GENETICS_HAILTOP_IMAGE
from hailtop.batch_client import BatchNotCreatedError, JobNotSubmittedError
from hailtop.batch_client.aioclient import Batch as AioBatch
from hailtop.batch_client.aioclient import BatchClient as AioBatchClient
from hailtop.batch_client.aioclient import SpecBytes, SpecType
from hailtop.batch_client.client import Batch, BatchClient
from hailtop.config import get_deploy_config
from hailtop.test_utils import skip_in_azure
from hailtop.utils import delay_ms_for_try, external_requests_client_session, retry_response_returning_functions
from hailtop.utils.rich_progress_bar import BatchProgressBar

from .failure_injecting_client_session import FailureInjectingClientSession
from .utils import DOCKER_ROOT_IMAGE, HAIL_GENETICS_HAIL_IMAGE, create_batch, legacy_batch_status, smallest_machine_type

deploy_config = get_deploy_config()


@pytest.fixture
def client():
    client = BatchClient('test')
    yield client
    client.close()


def test_job(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    b.submit()

    status = j.wait()
    assert 'attributes' not in status, str((status, b.debug_info()))
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert status['exit_code'] == 0, str((status, b.debug_info()))
    assert j._get_exit_code(status, 'main') == 0, str((status, b.debug_info()))
    job_log = j.log()
    assert job_log['main'] == 'test\n', str((job_log, b.debug_info()))


def test_job_running_logs(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['bash', '-c', 'echo test && sleep 300'])
    b.submit()

    wait_status = j._wait_for_states('Running')
    if wait_status['state'] != 'Running':
        assert False, str((j.log(), b.debug_info()))

    log = j.log()
    if log is not None and log['main'] != '':
        assert log['main'] == 'test\n', str((log, b.debug_info()))

    b.cancel()
    b.wait()


def test_exit_code_duration(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['bash', '-c', 'exit 7'])
    b.submit()
    status = j.wait()
    assert status['exit_code'] == 7, str((status, b.debug_info()))
    assert isinstance(status['duration'], int), str((status, b.debug_info()))
    assert j._get_exit_code(status, 'main') == 7, str((status, b.debug_info()))


def test_attributes(client: BatchClient):
    a = {'name': 'test_attributes', 'foo': 'bar'}
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], attributes=a)
    b.submit()
    assert j.attributes() == a, str(b.debug_info())


def test_garbage_image(client: BatchClient):
    b = create_batch(client)
    j = b.create_job('dsafaaadsf', ['echo', 'test'])
    b.submit()
    status = j.wait()
    assert j._get_exit_codes(status) == {'main': None}, str((status, b.debug_info()))
    assert j._get_error(status, 'main') is not None, str((status, b.debug_info()))
    assert status['state'] == 'Error', str((status, b.debug_info()))


def test_bad_command(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['sleep 5'])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))


def test_invalid_resource_requests(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '1', 'memory': '250Gi', 'storage': '1Gi'}
    b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(httpx.ClientResponseError, match='resource requests.*unsatisfiable'):
        b.submit()

    b = create_batch(client)
    resources = {'cpu': '0', 'memory': '1Gi', 'storage': '1Gi'}
    b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(
        httpx.ClientResponseError,
        match='bad resource request for job.*cpu must be a power of two with a min of 0.25; found.*',
    ):
        b.submit()

    b = create_batch(client)
    resources = {'cpu': '0.1', 'memory': '1Gi', 'storage': '1Gi'}
    b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(
        httpx.ClientResponseError,
        match='bad resource request for job.*cpu must be a power of two with a min of 0.25; found.*',
    ):
        b.submit()

    b = create_batch(client)
    resources = {'cpu': '0.25', 'memory': 'foo', 'storage': '1Gi'}
    b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(
        httpx.ClientResponseError,
        match=".*.resources.memory must match regex:.*.resources.memory must be one of:.*",
    ):
        b.submit()

    b = create_batch(client)
    resources = {'cpu': '0.25', 'memory': '500Mi', 'storage': '10000000Gi'}
    b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(httpx.ClientResponseError, match='resource requests.*unsatisfiable'):
        b.submit()

    b = create_batch(client)
    resources = {'storage': '10000000Gi', 'machine_type': smallest_machine_type()}
    b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(httpx.ClientResponseError, match='resource requests.*unsatisfiable'):
        b.submit()


def test_out_of_memory(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '0.25'}
    j = b.create_job('python:3.6-slim-stretch', ['python', '-c', 'x = "a" * (2 * 1024**3)'], resources=resources)
    b.submit()
    status = j.wait()
    assert j._get_out_of_memory(status, 'main'), str((status, b.debug_info()))


def test_out_of_storage(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '0.25'}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'fallocate -l 100GiB /foo'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    job_log = j.log()
    assert "fallocate failed: No space left on device" in job_log['main']


def test_quota_applies_to_volume(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '0.25'}
    j = b.create_job(
        os.environ['HAIL_VOLUME_IMAGE'], ['/bin/sh', '-c', 'fallocate -l 100GiB /data/foo'], resources=resources
    )
    b.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    job_log = j.log()
    assert "fallocate failed: No space left on device" in job_log['main']


def test_relative_volume_path_is_actually_absolute(client: BatchClient):
    # https://github.com/hail-is/hail/pull/12990#issuecomment-1540332989
    b = create_batch(client)
    resources = {'cpu': '0.25'}
    j = b.create_job(
        os.environ['HAIL_VOLUME_IMAGE'],
        ['/bin/sh', '-c', 'ls / && ls . && ls /relative_volume && ! ls relative_volume'],
        resources=resources,
    )
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_quota_shared_by_io_and_rootfs(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '0.25', 'storage': '10Gi'}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'fallocate -l 7GiB /foo'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))

    b = create_batch(client)
    resources = {'cpu': '0.25', 'storage': '10Gi'}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'fallocate -l 7GiB /io/foo'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))

    b = create_batch(client)
    resources = {'cpu': '0.25', 'storage': '10Gi'}
    j = b.create_job(
        DOCKER_ROOT_IMAGE,
        ['/bin/sh', '-c', 'fallocate -l 7GiB /foo; fallocate -l 7GiB /io/foo'],
        resources=resources,
    )
    b.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    job_log = j.log()
    assert "fallocate failed: No space left on device" in job_log['main'], str((job_log, b.debug_info()))


def test_nonzero_storage(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '0.25', 'storage': '20Gi'}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'true'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


@skip_in_azure
def test_attached_disk(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '0.25', 'storage': '400Gi'}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'df -h; fallocate -l 390GiB /io/foo'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_cwd_from_image_workdir(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(os.environ['HAIL_WORKDIR_IMAGE'], ['/bin/sh', '-c', 'pwd'])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    job_log = j.log()
    assert "/work" in job_log['main'], str((job_log, b.debug_info()))


def test_unsubmitted_state(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])

    with pytest.raises(BatchNotCreatedError):
        j.batch_id  # pylint: disable=pointless-statement
    with pytest.raises(JobNotSubmittedError):
        j.id  # pylint: disable=pointless-statement
    with pytest.raises(JobNotSubmittedError):
        j.status()
    with pytest.raises(JobNotSubmittedError):
        j.is_complete()
    with pytest.raises(JobNotSubmittedError):
        j.log()
    with pytest.raises(JobNotSubmittedError):
        j.wait()


def test_list_batches_v1(client: BatchClient):
    tag = secrets.token_urlsafe(64)
    b1 = create_batch(client, attributes={'tag': tag, 'name': 'b1'})
    b1.create_job(DOCKER_ROOT_IMAGE, ['sleep', '3600'])
    b1.submit()

    b2 = create_batch(client, attributes={'tag': tag, 'name': 'b2'})
    b2.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    b2.submit()

    batch_id_test_universe = {b1.id, b2.id}

    def assert_batch_ids(expected: Set[int], q=None):
        assert expected.issubset(batch_id_test_universe)
        max_id = max(batch_id_test_universe)
        min_id = min(batch_id_test_universe)
        span = max_id - min_id + 1
        # list_batches returns all batches for all prev run tests so we set a limit
        batches = client.list_batches(q, last_batch_id=max_id + 1, limit=span)
        full_actual = {b.id for b in batches}
        actual = full_actual.intersection(batch_id_test_universe)
        assert actual == expected, str((full_actual, max_id, span, b1.debug_info(), b2.debug_info()))

    assert_batch_ids({b1.id, b2.id})

    assert_batch_ids({b1.id, b2.id}, f'tag={tag}')

    b2.wait()

    assert_batch_ids({b1.id}, f'!complete tag={tag}')
    assert_batch_ids({b2.id}, f'complete tag={tag}')

    assert_batch_ids({b1.id}, f'!success tag={tag}')
    assert_batch_ids({b2.id}, f'success tag={tag}')

    b1.cancel()
    b1.wait()

    assert_batch_ids({b1.id}, f'!success tag={tag}')
    assert_batch_ids({b2.id}, f'success tag={tag}')

    assert_batch_ids(set(), f'!complete tag={tag}')
    assert_batch_ids({b1.id, b2.id}, f'complete tag={tag}')

    assert_batch_ids({b2.id}, f'tag={tag} name=b2')


def test_list_batches_v2(client: BatchClient):
    tag = secrets.token_urlsafe(64)
    partial_match_prefix = secrets.token_urlsafe(10)
    b1 = create_batch(client, attributes={'tag': tag, 'name': 'b1', 'partial_match_name': f'{partial_match_prefix}-b1'})
    b1.create_job(DOCKER_ROOT_IMAGE, ['sleep', '3600'])
    b1.submit()

    b2 = create_batch(client, attributes={'tag': tag, 'name': 'b2', 'partial_match_name': f'{partial_match_prefix}-b2'})
    b2.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    b2.submit()

    batch_id_test_universe = {b1.id, b2.id}

    def assert_batch_ids(expected: Set[int], q=None):
        assert expected.issubset(batch_id_test_universe)
        max_id = max(batch_id_test_universe)
        min_id = min(batch_id_test_universe)
        span = max_id - min_id + 1
        # list_batches returns all batches for all prev run tests so we set a limit
        batches = client.list_batches(q, last_batch_id=max_id + 1, limit=span, version=2)
        full_actual = {b.id for b in batches}
        actual = full_actual.intersection(batch_id_test_universe)
        assert actual == expected, str((full_actual, max_id, span, b1.debug_info(), b2.debug_info()))

    try:
        assert_batch_ids({b1.id, b2.id})

        assert_batch_ids({b1.id, b2.id}, f'tag={tag}')
        assert_batch_ids({b1.id, b2.id}, f'tag=~{tag}')
        assert_batch_ids(
            {b1.id, b2.id},
            f"""
name=~b
tag={tag}
""",
        )
        assert_batch_ids(
            {b1.id},
            f"""
name!~b2
tag={tag}
""",
        )
        assert_batch_ids(
            {b1.id},
            f"""
name!=b2
tag={tag}
""",
        )
        assert_batch_ids(
            {b2.id},
            f"""
{partial_match_prefix[3:]}-b2
tag={tag}
""",
        )

        b2.wait()

        assert_batch_ids(
            {b1.id},
            f"""
state != complete
tag = {tag}
""",
        )
        assert_batch_ids(
            {b2.id},
            f"""
state=complete
tag={tag}
""",
        )

        assert_batch_ids(
            {b1.id},
            f"""
state != success
tag={tag}
""",
        )
        assert_batch_ids(
            {b2.id},
            f"""
state == success
tag={tag}
""",
        )

        b1.cancel()
        b1.wait()

        assert_batch_ids(
            {b1.id},
            f"""
state!=success
tag={tag}
""",
        )
        assert_batch_ids(
            {b2.id},
            f"""
state = success
tag={tag}
""",
        )

        assert_batch_ids(
            set(),
            f"""
state != complete
tag={tag}
""",
        )
        assert_batch_ids(
            {b1.id, b2.id},
            f"""
state = complete
tag={tag}
""",
        )

        assert_batch_ids(
            {b2.id},
            f"""
tag={tag}
name=b2
""",
        )

        assert_batch_ids(
            {b2.id},
            f"""
tag={tag}
"b2"
""",
        )

        assert_batch_ids(
            {b2.id},
            f"""
tag=~{tag}
"b2"
""",
        )

        assert_batch_ids(
            batch_id_test_universe,
            f"""
user != foo
tag={tag}
""",
        )

        assert_batch_ids(
            batch_id_test_universe,
            f"""
billing_project = {client.billing_project}
tag={tag}
""",
        )

        assert_batch_ids(
            {b1.id, b2.id},
            f"""
start_time >= 2023-02-24T17:15:25Z
end_time < 3000-02-24T17:15:25Z
tag = {tag}
""",
        )

        assert_batch_ids(
            set(),
            f"""
start_time >= 2023-02-24T17:15:25Z
end_time == 2023-02-24T17:15:25Z
tag = {tag}
""",
        )

        assert_batch_ids(
            set(),
            f"""
duration > 50000
tag = {tag}
""",
        )
        assert_batch_ids(
            set(),
            f"""
cost > 1000
tag = {tag}
""",
        )
        assert_batch_ids(
            {b1.id},
            f"""
batch_id = {b1.id}
tag = {tag}
""",
        )
        assert_batch_ids(
            {b1.id},
            f"""
batch_id == {b1.id}
tag = {tag}
""",
        )

        with pytest.raises(httpx.ClientResponseError, match='could not parse term'):
            assert_batch_ids(batch_id_test_universe, 'batch_id >= 1 abcde')
        with pytest.raises(httpx.ClientResponseError, match='expected float, but found'):
            assert_batch_ids(batch_id_test_universe, 'duration >= abcd')
        with pytest.raises(httpx.ClientResponseError, match='expected int, but found'):
            assert_batch_ids(batch_id_test_universe, 'batch_id >= abcd')
        with pytest.raises(httpx.ClientResponseError, match='expected float, but found'):
            assert_batch_ids(batch_id_test_universe, 'cost >= abcd')
        with pytest.raises(httpx.ClientResponseError, match='unexpected operator'):
            assert_batch_ids(batch_id_test_universe, 'state >= 1')
        with pytest.raises(httpx.ClientResponseError, match='unknown state'):
            assert_batch_ids(batch_id_test_universe, 'state = 1')
        with pytest.raises(httpx.ClientResponseError, match='unexpected operator'):
            assert_batch_ids(batch_id_test_universe, 'user >= 1')
        with pytest.raises(httpx.ClientResponseError, match='unexpected operator'):
            assert_batch_ids(batch_id_test_universe, 'billing_project >= 1')
        with pytest.raises(httpx.ClientResponseError, match='expected date, but found'):
            assert_batch_ids(batch_id_test_universe, 'start_time >= 1')
        with pytest.raises(httpx.ClientResponseError, match='expected date, but found'):
            assert_batch_ids(batch_id_test_universe, 'end_time >= 1')

    finally:
        try:
            b1.cancel()
        finally:
            b2.cancel()


def test_list_jobs_v1(client: BatchClient):
    b = create_batch(client)
    j_success = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    j_failure = b.create_job(DOCKER_ROOT_IMAGE, ['false'])
    j_error = b.create_job(DOCKER_ROOT_IMAGE, ['sleep 5'], attributes={'tag': 'bar'})
    j_running = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '1800'], attributes={'tag': 'foo'})

    b.submit()

    def assert_job_ids(expected, q=None):
        jobs = b.jobs(q=q)
        actual = set(j['job_id'] for j in jobs)
        assert actual == expected, str((jobs, b.debug_info()))

    try:
        j_success.wait()
        j_failure.wait()
        j_error.wait()

        assert_job_ids({j_success.job_id}, 'success')
        assert_job_ids({j_success.job_id, j_failure.job_id, j_error.job_id}, 'done')
        assert_job_ids({j_running.job_id}, '!done')
        assert_job_ids({j_running.job_id}, 'tag=foo')
        assert_job_ids({j_error.job_id, j_running.job_id}, 'has:tag')
        assert_job_ids({j_success.job_id, j_failure.job_id, j_error.job_id, j_running.job_id}, None)
    finally:
        b.cancel()


def test_list_jobs_v2(client: BatchClient):
    b = create_batch(client)
    j_success = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    j_failure = b.create_job(DOCKER_ROOT_IMAGE, ['false'])
    j_error = b.create_job(DOCKER_ROOT_IMAGE, ['sleep 5'], attributes={'tag': 'bar'})
    j_running = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '1800'], attributes={'tag': 'foo'})

    b.submit()

    def assert_job_ids(expected, q=None):
        jobs = b.jobs(q=q, version=2)
        actual = set(j['job_id'] for j in jobs)
        assert actual == expected, str((jobs, b.debug_info()))

    try:
        j_success.wait()
        j_failure.wait()
        j_error.wait()
        wait_status = j_running._wait_for_states('Running')
        if wait_status['state'] != 'Running':
            assert False, str((b.debug_info(), wait_status))

        assert_job_ids({j_success.job_id}, 'state = success')
        assert_job_ids({j_success.job_id}, 'state == success')
        assert_job_ids({j_success.job_id}, 'state=success')
        assert_job_ids({j_success.job_id}, 'state==success')

        assert_job_ids({j_success.job_id, j_failure.job_id, j_error.job_id}, 'state=done')
        assert_job_ids({j_running.job_id}, 'state != done')

        assert_job_ids({j_running.job_id}, 'tag=foo')
        assert_job_ids({j_running.job_id}, 'tag=~fo')
        assert_job_ids({j_running.job_id}, 'tag = foo')
        assert_job_ids({j_running.job_id}, 'tag =~ fo')

        assert_job_ids({j_error.job_id}, 'tag!=foo')
        assert_job_ids({j_error.job_id}, 'tag != foo')
        assert_job_ids({j_error.job_id, j_running.job_id}, '"tag"')
        assert_job_ids({j_running.job_id}, 'foo')

        no_jobs: Set[int] = set()
        all_jobs = {j_error.job_id, j_running.job_id, j_failure.job_id, j_success.job_id}
        assert_job_ids(no_jobs, 'duration > 50000')
        assert_job_ids(all_jobs, 'instance_collection = standard')
        assert_job_ids(no_jobs, 'cost > 1000')

        assert_job_ids(no_jobs, 'start_time == 2023-02-24T17:15:25Z')
        assert_job_ids(no_jobs, 'end_time == 2023-02-24T17:15:25Z')

        assert_job_ids(no_jobs, 'start_time<2023-02-24T17:15:25Z')
        assert_job_ids(no_jobs, 'start_time<=2023-02-24T17:15:25Z')
        assert_job_ids(all_jobs, 'start_time != 2023-02-24T17:15:25Z')
        assert_job_ids(all_jobs, 'start_time>2023-02-24T17:15:25Z')
        assert_job_ids(all_jobs, 'start_time>=2023-02-24T17:15:25Z')

        assert_job_ids(no_jobs, 'start_time < 2023-02-24T17:15:25Z')
        assert_job_ids(no_jobs, 'start_time <= 2023-02-24T17:15:25Z')
        assert_job_ids(all_jobs, 'start_time > 2023-02-24T17:15:25Z')
        assert_job_ids(all_jobs, 'start_time >= 2023-02-24T17:15:25Z')

        assert_job_ids(no_jobs, 'instance = batch-worker')
        assert_job_ids(all_jobs, 'instance != batch-worker')
        assert_job_ids(all_jobs, 'instance =~ batch-worker')
        assert_job_ids(no_jobs, 'instance !~ batch-worker')

        assert_job_ids(no_jobs, 'instance=batch-worker')
        assert_job_ids(all_jobs, 'instance!=batch-worker')
        assert_job_ids(all_jobs, 'instance=~batch-worker')
        assert_job_ids(no_jobs, 'instance!~batch-worker')

        assert_job_ids({j_success.job_id}, 'job_id = 1')
        assert_job_ids(all_jobs, 'job_id >= 1')

        assert_job_ids(all_jobs, None)

        assert_job_ids(
            no_jobs,
            """
job_id >=1
instance == foo
foo = bar
start_time >= 2023-02-24T17:15:25Z
end_time <= 2023-02-24T17:18:25Z
""",
        )

        with pytest.raises(httpx.ClientResponseError, match='could not parse term'):
            assert_job_ids(no_jobs, 'job_id >= 1 abcde')
        with pytest.raises(httpx.ClientResponseError, match='expected float, but found'):
            assert_job_ids(no_jobs, 'duration >= abcd')
        with pytest.raises(httpx.ClientResponseError, match='expected int, but found'):
            assert_job_ids(no_jobs, 'job_id >= abcd')
        with pytest.raises(httpx.ClientResponseError, match='expected float, but found'):
            assert_job_ids(no_jobs, 'cost >= abcd')
        with pytest.raises(httpx.ClientResponseError, match='unexpected operator'):
            assert_job_ids(no_jobs, 'state >= 1')
        with pytest.raises(httpx.ClientResponseError, match='unknown state'):
            assert_job_ids(no_jobs, 'state = 1')
        with pytest.raises(httpx.ClientResponseError, match='unexpected operator'):
            assert_job_ids(no_jobs, 'instance_collection >= 1')
        with pytest.raises(httpx.ClientResponseError, match='unexpected operator'):
            assert_job_ids(no_jobs, 'instance >= 1')
        with pytest.raises(httpx.ClientResponseError, match='expected date, but found'):
            assert_job_ids(no_jobs, 'start_time >= 1')
        with pytest.raises(httpx.ClientResponseError, match='expected date, but found'):
            assert_job_ids(no_jobs, 'end_time >= 1')
    finally:
        b.cancel()


def test_include_jobs(client: BatchClient):
    b1 = create_batch(client)
    for _ in range(2):
        b1.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b1.submit()
    s = b1.status()
    assert 'jobs' not in s, str((s, b1.debug_info()))


def test_fail(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['false'])
    b.submit()
    status = j.wait()
    assert j._get_exit_code(status, 'main') == 1, str((status, b.debug_info()))


def test_unknown_image(client: BatchClient):
    DOCKER_PREFIX = os.environ['DOCKER_PREFIX']
    b = create_batch(client)
    j = b.create_job(f'{DOCKER_PREFIX}/does-not-exist', ['echo', 'test'])
    b.submit()
    status = j.wait()
    try:
        assert j._get_exit_code(status, 'main') is None
        assert status['status']['container_statuses']['main']['short_error'] == 'image not found', str((
            status,
            b.debug_info(),
        ))
    except Exception as e:
        raise AssertionError(str((status, b.debug_info()))) from e


@skip_in_azure
def test_invalid_gar(client: BatchClient):
    b = create_batch(client)
    # GCP projects can't be strictly numeric
    j = b.create_job('us-docker.pkg.dev/1/does-not-exist', ['echo', 'test'])
    b.submit()
    status = j.wait()
    try:
        assert j._get_exit_code(status, 'main') is None
        assert status['status']['container_statuses']['main']['short_error'] == 'image cannot be pulled', str((
            status,
            b.debug_info(),
        ))
    except Exception as e:
        raise AssertionError(str((status, b.debug_info()))) from e


def test_running_job_log_and_status(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '300'])
    b.submit()

    while True:
        if j.status()['state'] == 'Running' or j.is_complete():
            break

    j.log()
    # FIXME after batch1 goes away, check running status
    b.cancel()


def test_deleted_job_log(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    b.submit()
    j.wait()
    b.delete()

    try:
        j.log()
    except httpx.ClientResponseError as e:
        if e.status == 404:
            pass
        else:
            assert False, str((e, b.debug_info()))


def test_delete_batch(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b.submit()
    b.delete()

    # verify doesn't exist
    try:
        client.get_job(*j.id)
    except httpx.ClientResponseError as e:
        if e.status == 404:
            pass
        else:
            raise


def test_cancel_batch(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b.submit()

    status = j.status()
    assert status['state'] in ('Ready', 'Running'), str((status, b.debug_info()))

    b.cancel()

    status = j.wait()
    assert status['state'] == 'Cancelled', str((status, b.debug_info()))
    assert 'log' not in status, str((status, b.debug_info()))

    # cancelled job has no log
    try:
        j.log()
    except httpx.ClientResponseError as e:
        if e.status == 404:
            pass
        else:
            raise


def test_get_nonexistent_job(client: BatchClient):
    try:
        client.get_job(1, 666)
    except httpx.ClientResponseError as e:
        if e.status == 404:
            pass
        else:
            raise


def test_get_job(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()

    j2 = client.get_job(*j.id)
    status2 = j2.status()
    assert (status2['batch_id'], status2['job_id']) == j.id, str((status2, b.debug_info()))


def test_batch(client: BatchClient):
    b = create_batch(client)
    j1 = b.create_job(DOCKER_ROOT_IMAGE, ['false'])
    j2 = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '1'])
    b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b.submit()

    j1.wait()
    j2.wait()
    b.cancel()
    b.wait()
    bstatus = legacy_batch_status(b)

    assert len(bstatus['jobs']) == 3, str((bstatus, b.debug_info()))
    state_count = collections.Counter([j['state'] for j in bstatus['jobs']])
    n_cancelled = state_count['Cancelled']
    n_complete = state_count['Error'] + state_count['Failed'] + state_count['Success']
    assert n_cancelled <= 1, str((bstatus, b.debug_info()))
    assert n_cancelled + n_complete == 3, str((bstatus, b.debug_info()))

    n_failed = sum(j['exit_code'] > 0 for j in bstatus['jobs'] if j['state'] in ('Failed', 'Error'))
    assert n_failed == 1, str((bstatus, b.debug_info()))


def test_batch_status(client: BatchClient):
    b1 = create_batch(client)
    b1.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b1.submit()
    b1.wait()
    b1s = b1.status()
    assert b1s['complete'] and b1s['state'] == 'success', str((b1s, b1.debug_info()))

    b2 = create_batch(client)
    b2.create_job(DOCKER_ROOT_IMAGE, ['false'])
    b2.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b2.submit()
    b2.wait()
    b2s = b2.status()
    assert b2s['complete'] and b2s['state'] == 'failure', str((b2s, b2.debug_info()))

    b3 = create_batch(client)
    b3.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b3.submit()
    b3s = b3.status()
    assert not b3s['complete'] and b3s['state'] == 'running', str((b3s, b3.debug_info()))
    b3.cancel()

    b4 = create_batch(client)
    b4.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b4.submit()
    b4.cancel()
    b4.wait()
    b4s = b4.status()
    assert b4s['complete'] and b4s['state'] == 'cancelled', str((b4s, b4.debug_info()))


def test_log_after_failing_job(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'echo test; exit 127'])
    b.submit()
    status = j.wait()
    assert 'attributes' not in status, str((status, b.debug_info()))
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    assert j._get_exit_code(status, 'main') == 127, str((status, b.debug_info()))

    job_log = j.log()
    assert job_log['main'] == 'test\n', str((job_log, b.debug_info()))

    assert j.is_complete(), str(b.debug_info())


def test_non_utf_8_log(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', "echo -n 'hello \\x80'"])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))

    job_main_log = j.container_log('main')
    assert job_main_log == b'hello \\x80'


def test_long_log_line(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'for _ in {0..70000}; do echo -n a; done'])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_authorized_users_only():
    session = external_requests_client_session()
    endpoints = [
        (session.get, '/api/v1alpha/billing_projects', 401),
        (session.get, '/api/v1alpha/billing_projects/foo', 401),
        (session.post, '/api/v1alpha/billing_projects/foo/users/foo/add', 401),
        (session.post, '/api/v1alpha/billing_projects/foo/users/foo/remove', 401),
        (session.post, '/api/v1alpha/billing_projects/foo/create', 401),
        (session.post, '/api/v1alpha/billing_projects/foo/close', 401),
        (session.post, '/api/v1alpha/billing_projects/foo/reopen', 401),
        (session.post, '/api/v1alpha/billing_projects/foo/delete', 401),
        (session.post, '/api/v1alpha/billing_limits/foo/edit', 401),
        (session.get, '/api/v1alpha/batches/0/jobs/0', 401),
        (session.get, '/api/v1alpha/batches/0/jobs/0/log', 401),
        (session.get, '/api/v1alpha/batches', 401),
        (session.post, '/api/v1alpha/batches/create', 401),
        (session.post, '/api/v1alpha/batches/0/jobs/create', 401),
        (session.get, '/api/v1alpha/batches/0', 401),
        (session.delete, '/api/v1alpha/batches/0', 401),
        (session.patch, '/api/v1alpha/batches/0/close', 401),
        (session.get, '/api/v1alpha/batches/0/job-groups', 401),
        (session.get, '/api/v1alpha/batches/0/job-groups/0/job-groups', 401),
        (session.post, '/api/v1alpha/batches/0/updates/0/job-groups/create', 401),
        (session.post, '/api/v1alpha/batches/0/updates/0/jobs/create', 401),
        # redirect to auth/login
        (session.get, '/batches', 302),
        (session.get, '/batches/0', 302),
        (session.post, '/batches/0/cancel', 401),
        (session.get, '/batches/0/jobs/0', 302),
    ]
    for method, url, expected in endpoints:
        full_url = deploy_config.url('batch', url)
        r = retry_response_returning_functions(method, full_url, allow_redirects=False)
        assert r.status_code == expected, (full_url, r, expected)


def test_cloud_image(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(os.environ['HAIL_CURL_IMAGE'], ['echo', 'test'])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_service_account(client: BatchClient):
    NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
    b = create_batch(client)
    j = b.create_job(
        os.environ['CI_UTILS_IMAGE'],
        ['/bin/sh', '-c', 'kubectl version'],
        service_account={'namespace': NAMESPACE, 'name': 'test-batch-sa'},
    )
    b.submit()
    status = j.wait()
    assert j._get_exit_code(status, 'main') == 0, str((status, b.debug_info()))


def test_port(client: BatchClient):
    b = create_batch(client)
    b.create_job(
        DOCKER_ROOT_IMAGE,
        [
            'bash',
            '-c',
            """
echo $HAIL_BATCH_WORKER_PORT
echo $HAIL_BATCH_WORKER_IP
""",
        ],
        port=5000,
    )
    b.submit()
    batch = b.wait()
    assert batch['state'] == 'success', str((batch, b.debug_info()))


def test_timeout(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'], timeout=5)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Error', str((status, b.debug_info()))
    error_msg = j._get_error(status, 'main')
    assert error_msg and 'ContainerTimeoutError' in error_msg, str((error_msg, b.debug_info()))
    assert j.exit_code(status) is None, str((status, b.debug_info()))


def test_client_max_size(client: BatchClient):
    b = create_batch(client)
    for _ in range(4):
        b.create_job(DOCKER_ROOT_IMAGE, ['echo', 'a' * (900 * 1024)])
    b.submit()


async def test_restartable_insert():
    i = 0

    def every_third_time():
        nonlocal i
        i += 1
        if i % 3 == 0:
            return True
        return False

    async with FailureInjectingClientSession(every_third_time) as session:
        client = await AioBatchClient.create('test', session=session)
        b = create_batch(client)

        for _ in range(9):
            b.create_job(DOCKER_ROOT_IMAGE, ['echo', 'a'])

        await b.submit(max_bunch_size=1)
        b = await client.get_batch(b.id)  # get a batch untainted by the FailureInjectingClientSession
        status = await b.wait()
        assert status['state'] == 'success', str((status, await b.debug_info()))
        jobs = [x async for x in b.jobs()]
        assert len(jobs) == 9, str((jobs, await b.debug_info()))


def test_create_idempotence(client: BatchClient):
    token = secrets.token_urlsafe(32)
    b1 = Batch._open_batch(client, token=token)
    b2 = Batch._open_batch(client, token=token)
    assert b1.id == b2.id


async def test_batch_create_validation():
    bad_configs = [
        # unexpected field fleep
        {'billing_project': 'foo', 'n_jobs': 5, 'token': 'baz', 'fleep': 'quam'},
        # billing project None/missing
        {'billing_project': None, 'n_jobs': 5, 'token': 'baz'},
        {'n_jobs': 5, 'token': 'baz'},
        # n_jobs None/missing
        {'billing_project': 'foo', 'n_jobs': None, 'token': 'baz'},
        {'billing_project': 'foo', 'token': 'baz'},
        # n_jobs wrong type
        {'billing_project': 'foo', 'n_jobs': '5', 'token': 'baz'},
        # token None/missing
        {'billing_project': 'foo', 'n_jobs': 5, 'token': None},
        {'billing_project': 'foo', 'n_jobs': 5},
        # empty gcsfuse bucket name
        {
            'billing_project': 'foo',
            'n_jobs': 5,
            'token': 'baz',
            'gcsfuse': [{'bucket': '', 'mount_path': '/bucket', 'read_only': False}],
        },
        # empty gcsfuse mount_path name
        {
            'billing_project': 'foo',
            'n_jobs': 5,
            'token': 'baz',
            'gcsfuse': [{'bucket': 'foo', 'mount_path': '', 'read_only': False}],
        },
        # attribute key/value None
        {'attributes': {'k': None}, 'billing_project': 'foo', 'n_jobs': 5, 'token': 'baz'},
    ]
    url = deploy_config.url('batch', '/api/v1alpha/batches/create')
    async with hail_credentials() as creds:
        headers = await creds.auth_headers()
    session = external_requests_client_session()
    for config in bad_configs:
        r = retry_response_returning_functions(session.post, url, json=config, allow_redirects=True, headers=headers)
        assert r.status_code == 400, (config, r)


def test_duplicate_parents(client: BatchClient):
    b = create_batch(client)
    head = b.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    b.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[head, head])
    try:
        b.submit()
    except httpx.ClientResponseError as e:
        assert e.status == 400
    else:
        assert False, f'should receive a 400 Bad Request {b.id}'


@skip_in_azure
def test_hail_metadata_server_uses_correct_user_credentials(client: BatchClient):
    b = create_batch(client)
    userinfo = get_userinfo()
    assert userinfo
    hail_identity = userinfo['hail_identity']
    j = b.create_job(
        os.environ['HAIL_CURL_IMAGE'],
        ['curl', '-fsSL', 'metadata.google.internal/computeMetadata/v1/instance/service-accounts/', '--max-time', '10'],
    )
    b.submit()
    status = j.wait()
    job_log = j.log()
    service_accounts = set(sa.strip() for sa in job_log['main'].split())
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert service_accounts == set(('default', hail_identity))


@skip_in_azure
def test_gcloud_works_with_hail_metadata_server(client: BatchClient):
    b = create_batch(client)
    token = secrets.token_urlsafe(16)
    tmpdir = os.environ['HAIL_BATCH_REMOTE_TMPDIR']
    random_dir = f'{tmpdir}/{token}'
    script = f"""
set -ex
unset GOOGLE_APPLICATION_CREDENTIALS
gcloud config list account
echo "hello" >hello.txt
gcloud storage cp hello.txt {random_dir}/hello.txt
gcloud storage ls {random_dir}
gcloud storage rm -r {random_dir}/
"""
    j = b.create_job(os.environ['CI_UTILS_IMAGE'], ['/bin/bash', '-c', script])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_hail_metadata_server_available_only_in_gcp(client: BatchClient):
    cloud = os.environ['HAIL_CLOUD']
    b = create_batch(client)
    j = b.create_job(os.environ['HAIL_CURL_IMAGE'], ['curl', '-fsSL', '169.254.169.254', '--max-time', '10'])
    b.submit()
    status = j.wait()
    if cloud == 'gcp':
        assert status['state'] == 'Success', str((status, b.debug_info()))
    else:
        assert cloud == 'azure'
        assert status['state'] == 'Failed', str((status, b.debug_info()))
        job_log = j.log()
        assert "Connection timeout" in job_log['main'], str((job_log, b.debug_info()))


def test_submit_batch_in_job(client: BatchClient, remote_tmpdir: str):
    b = create_batch(client)
    script = f"""import hailtop.batch as hb
backend = hb.ServiceBackend("test", remote_tmpdir="{remote_tmpdir}")
b = hb.Batch(backend=backend)
j = b.new_bash_job()
j.command("echo hi")
b.run()
backend.close()
"""
    j = b.create_job(
        HAIL_GENETICS_HAILTOP_IMAGE,
        ['/bin/bash', '-c', f"""python3 -c \'{script}\'"""],
    )
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_cant_submit_to_default_with_other_ns_creds(client: BatchClient, remote_tmpdir: str):
    DOMAIN = os.environ['HAIL_PRODUCTION_DOMAIN']
    NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']

    script = f"""import hailtop.batch as hb
backend = hb.ServiceBackend("test", remote_tmpdir="{remote_tmpdir}")
b = hb.Batch(backend=backend)
j = b.new_bash_job()
j.command("echo hi")
b.run()
backend.close()
"""

    b = create_batch(client)
    j = b.create_job(
        HAIL_GENETICS_HAILTOP_IMAGE,
        [
            '/bin/bash',
            '-c',
            f"""
python3 -c \'{script}\'""",
        ],
        env={
            'HAIL_DOMAIN': DOMAIN,
            'HAIL_DEFAULT_NAMESPACE': 'default',
            'HAIL_LOCATION': 'external',
            'HAIL_BASE_PATH': '',
        },
    )
    b.submit()
    status = j.wait()
    if NAMESPACE == 'default':
        assert status['state'] == 'Success', str((status, b.debug_info()))
    else:
        assert status['state'] == 'Failed', str((status, b.debug_info()))
        assert 'Unauthorized' in j.log()['main'], (str(j.log()['main']), status)


def test_deploy_config_is_mounted_as_readonly(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(
        HAIL_GENETICS_HAILTOP_IMAGE,
        [
            '/bin/bash',
            '-c',
            """
set -ex
jq '.default_namespace = "default"' /deploy-config/deploy-config.json > tmp.json
mv tmp.json /deploy-config/deploy-config.json""",
        ],
        mount_tokens=True,
    )
    b.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    job_log = j.log()
    assert "mv: cannot move" in job_log['main'], str((job_log, b.debug_info()))


def test_cannot_contact_other_internal_ips(client: BatchClient):
    internal_ips = [f'10.128.0.{i}' for i in (10, 11, 12)]
    b = create_batch(client)
    script = f"""
if [ "$HAIL_BATCH_WORKER_IP" != "{internal_ips[0]}" ] && ! grep -Fq {internal_ips[0]} /etc/hosts; then
    OTHER_IP={internal_ips[0]}
elif [ "$HAIL_BATCH_WORKER_IP" != "{internal_ips[1]}" ] && ! grep -Fq {internal_ips[1]} /etc/hosts; then
    OTHER_IP={internal_ips[1]}
else
    OTHER_IP={internal_ips[2]}
fi

curl -fsSL -m 5 $OTHER_IP
"""
    j = b.create_job(os.environ['HAIL_CURL_IMAGE'], ['/bin/bash', '-c', script], port=5000)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    job_log = j.log()
    assert "Connection timed out" in job_log['main'], str((job_log, b.debug_info()))


@skip_in_azure
def test_hadoop_can_use_cloud_credentials(client: BatchClient, remote_tmpdir: str):
    token = os.environ["HAIL_TOKEN"]
    b = create_batch(client)
    script = f"""import hail as hl
import secrets
attempt_token = secrets.token_urlsafe(5)
location = f"{remote_tmpdir}/{ token }/{{ attempt_token }}/test_can_use_hailctl_auth.t"
hl.utils.range_table(10).write(location)
hl.read_table(location).show()
"""
    j = b.create_job(HAIL_GENETICS_HAIL_IMAGE, ['/bin/bash', '-c', f'python3 -c >out 2>err \'{script}\'; cat out err'])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', f'{j.log(), status}'
    expected_log = """+-------+
|   idx |
+-------+
| int32 |
+-------+
|     0 |
|     1 |
|     2 |
|     3 |
|     4 |
|     5 |
|     6 |
|     7 |
|     8 |
|     9 |
+-------+
"""
    log = j.log()
    assert expected_log in log['main'], str((log, b.debug_info()))


def test_user_authentication_within_job(client: BatchClient):
    b = create_batch(client)
    cmd = ['bash', '-c', 'hailctl auth user']
    no_token = b.create_job(HAIL_GENETICS_HAILTOP_IMAGE, cmd)
    b.submit()

    status = no_token.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_verify_access_to_public_internet(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(os.environ['HAIL_CURL_IMAGE'], ['curl', '-fsSL', 'example.com'])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_verify_can_tcp_to_localhost(client: BatchClient):
    b = create_batch(client)
    script = """
set -e
nc -l -p 5000 &
sleep 5
echo "hello" | nc -q 1 localhost 5000
""".lstrip('\n')
    j = b.create_job(os.environ['HAIL_NETCAT_UBUNTU_IMAGE'], command=['/bin/bash', '-c', script])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    job_log = j.log()
    assert 'hello\n' == job_log['main'], str((job_log, b.debug_info()))


def test_verify_can_tcp_to_127_0_0_1(client: BatchClient):
    b = create_batch(client)
    script = """
set -e
nc -l -p 5000 &
sleep 5
echo "hello" | nc -q 1 127.0.0.1 5000
""".lstrip('\n')
    j = b.create_job(os.environ['HAIL_NETCAT_UBUNTU_IMAGE'], command=['/bin/bash', '-c', script])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    job_log = j.log()
    assert 'hello\n' == job_log['main'], str((job_log, b.debug_info()))


def test_verify_can_tcp_to_self_ip(client: BatchClient):
    b = create_batch(client)
    script = """
set -e
nc -l -p 5000 &
sleep 5
echo "hello" | nc -q 1 $(hostname -i) 5000
""".lstrip('\n')
    j = b.create_job(os.environ['HAIL_NETCAT_UBUNTU_IMAGE'], command=['/bin/sh', '-c', script])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    job_log = j.log()
    assert 'hello\n' == job_log['main'], str((job_log, b.debug_info()))


def test_verify_private_network_is_restricted(client: BatchClient):
    b = create_batch(client)
    b.create_job(
        os.environ['HAIL_CURL_IMAGE'], command=['curl', 'internal.hail', '--connect-timeout', '60'], network='private'
    )
    try:
        b.submit()
    except httpx.ClientResponseError as err:
        assert err.status == 400
        assert 'unauthorized network private' in err.body
    else:
        assert False


async def test_old_clients_that_submit_mount_docker_socket_false_is_ok(client: BatchClient):
    b = create_batch(client)._async_batch
    await b._open_batch()
    b.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'])
    update_id = await b._create_update()
    with BatchProgressBar() as pbar:
        process = {
            'type': 'docker',
            'command': ['sleep', '30'],
            'image': DOCKER_ROOT_IMAGE,
            'mount_docker_socket': False,
        }
        spec = {'always_run': False, 'job_id': 1, 'parent_ids': [], 'process': process}
        with pbar.with_task('submitting jobs', total=1) as pbar_task:
            spec_bytes = SpecBytes(orjson.dumps(spec), SpecType.JOB)
            await b._submit_jobs(update_id, [spec_bytes], pbar_task)


async def test_old_clients_that_submit_mount_docker_socket_true_is_rejected(client: BatchClient):
    b = create_batch(client)._async_batch
    await b._open_batch()
    b.create_job(DOCKER_ROOT_IMAGE, command=['sleep', '30'])
    update_id = await b._create_update()
    with BatchProgressBar() as pbar:
        process = {
            'type': 'docker',
            'command': ['sleep', '30'],
            'image': DOCKER_ROOT_IMAGE,
            'mount_docker_socket': True,
        }
        spec = {'always_run': False, 'job_id': 1, 'parent_ids': [], 'process': process}
        with pbar.with_task('submitting jobs', total=1) as pbar_task:
            with pytest.raises(
                httpx.ClientResponseError,
                match='mount_docker_socket is no longer supported but was set to True in request. Please upgrade.',
            ):
                spec_bytes = SpecBytes(orjson.dumps(spec), SpecType.JOB)
                await b._submit_jobs(update_id, [spec_bytes], pbar_task)


def test_pool_highmem_instance(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '0.25', 'memory': 'highmem'}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'highmem' in status['status']['worker'], str((status, b.debug_info()))


def test_pool_highmem_instance_cheapest(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '1', 'memory': '5Gi'}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'highmem' in status['status']['worker'], str((status, b.debug_info()))


def test_pool_highcpu_instance(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '0.25', 'memory': 'lowmem'}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'highcpu' in status['status']['worker'], str((status, b.debug_info()))


@pytest.mark.xfail(os.environ.get('HAIL_CLOUD') == 'azure', strict=True, reason='prices changed in Azure 2023-06-01')
def test_pool_highcpu_instance_cheapest(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '0.25', 'memory': '50Mi'}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'highcpu' in status['status']['worker'], str((status, b.debug_info()))


def test_pool_standard_instance(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '0.25', 'memory': 'standard'}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'standard' in status['status']['worker'], str((status, b.debug_info()))


def test_pool_standard_instance_cheapest(client: BatchClient):
    b = create_batch(client)
    resources = {'cpu': '1', 'memory': '2.5Gi'}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'standard' in status['status']['worker'], str((status, b.debug_info()))


@skip_in_azure
@pytest.mark.timeout(10 * 60)
def test_gpu_accesibility_g2(client: BatchClient):
    b = create_batch(client)
    resources = {'machine_type': "g2-standard-4", 'storage': '100Gi'}
    j = b.create_job(
        os.environ['HAIL_GPU_IMAGE'],
        ['python3', '-c', 'import torch; assert torch.cuda.is_available()'],
        resources=resources,
    )
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_job_private_instance_preemptible(client: BatchClient):
    b = create_batch(client)
    resources = {'machine_type': smallest_machine_type()}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'job-private' in status['status']['worker'], str((status, b.debug_info()))


def test_job_private_instance_nonpreemptible(client: BatchClient):
    b = create_batch(client)
    resources = {'machine_type': smallest_machine_type(), 'preemptible': False}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'job-private' in status['status']['worker'], str((status, b.debug_info()))


def test_job_private_instance_cancel(client: BatchClient):
    b = create_batch(client)
    resources = {'machine_type': smallest_machine_type()}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b.submit()

    tries = 0
    start = time.time()
    while True:
        status = j.status()
        if status['state'] == 'Creating':
            break
        now = time.time()

        tries += 1
        cumulative_delay = now - start
        if cumulative_delay > 60:
            assert False, str((status, b.debug_info()))
        delay = min(delay_ms_for_try(tries), 60 - cumulative_delay)
        time.sleep(delay)
    b.cancel()
    status = j.wait()
    assert status['state'] == 'Cancelled', str((status, b.debug_info()))


def test_always_run_job_private_instance_cancel(client: BatchClient):
    b = create_batch(client)
    resources = {'machine_type': smallest_machine_type()}
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources, always_run=True)
    b.submit()
    b.cancel()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_create_fast_path_more_than_one_job(client: BatchClient):
    b = create_batch(client)
    b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()
    assert b._submission_info.used_fast_path, b._submission_info


def test_update_batch_no_deps(client: BatchClient):
    b = create_batch(client)
    j1 = b.create_job(DOCKER_ROOT_IMAGE, ['false'])
    b.submit()
    j1.wait()

    j2 = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()
    j2_status = j2.wait()

    assert j2_status['state'] == 'Success', str((j2_status, b.debug_info()))
    assert b.status()['state'] == 'failure', str((b.status(), b.debug_info()))


def test_update_batch_w_submitted_job_deps(client: BatchClient):
    b = create_batch(client)
    j1 = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()
    j1.wait()
    j2 = b.create_job(DOCKER_ROOT_IMAGE, ['true'], parents=[j1])
    b.submit()
    status = j2.wait()

    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert b.status()['state'] == 'success', str((b.status(), b.debug_info()))


def test_update_batch_w_failing_submitted_job_deps(client: BatchClient):
    b = create_batch(client)
    j1 = b.create_job(DOCKER_ROOT_IMAGE, ['false'])
    b.submit()

    j2 = b.create_job(DOCKER_ROOT_IMAGE, ['true'], parents=[j1])
    b.submit()
    status = j2.wait()

    assert status['state'] == 'Cancelled', str((status, b.debug_info()))
    assert b.status()['state'] == 'failure', str((b.status(), b.debug_info()))


def test_update_batch_w_deps_in_update(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()

    j1 = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    j2 = b.create_job(DOCKER_ROOT_IMAGE, ['true'], parents=[j, j1])
    b.submit()
    status = j2.wait()

    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert b.status()['state'] == 'success', str((b.status(), b.debug_info()))
    assert b.status()['n_jobs'] == 3, str(b.debug_info())


def test_update_batch_w_deps_in_update_always_run(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['false'])
    b.submit()

    j.wait()

    j1 = b.create_job(DOCKER_ROOT_IMAGE, ['true'], always_run=True)
    j2 = b.create_job(DOCKER_ROOT_IMAGE, ['true'], parents=[j, j1])
    b.submit()
    j2.wait()

    assert j1.status()['state'] == 'Success', str((j1.status(), b.debug_info()))
    assert j2.status()['state'] == 'Cancelled', str((j2.status(), b.debug_info()))
    assert b.status()['state'] == 'failure', str((b.status(), b.debug_info()))


def test_update_batch_w_failing_deps_in_same_update_and_deps_across_updates(client: BatchClient):
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()

    j1 = b.create_job(DOCKER_ROOT_IMAGE, ['false'], parents=[j])
    j2 = b.create_job(DOCKER_ROOT_IMAGE, ['true'], parents=[j1])
    b.submit()
    j2_status = j2.wait()

    assert j.status()['state'] == 'Success', str((j.status(), b.debug_info()))
    assert j2_status['state'] == 'Cancelled', str((j2_status, b.debug_info()))
    assert b.status()['state'] == 'failure', str((b.status(), b.debug_info()))


def test_update_with_always_run(client: BatchClient):
    b = create_batch(client)
    j1 = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '3600'])
    b.submit()

    j2 = b.create_job(DOCKER_ROOT_IMAGE, ['true'], always_run=True, parents=[j1])
    b.submit()

    wait_status = j1._wait_for_states('Running')
    if wait_status['state'] != 'Running':
        assert False, str((j1.status(), b.debug_info()))

    assert j2.is_pending(), str((j2.status(), b.debug_info()))

    b.cancel()
    j2.wait()

    assert j2.status()['state'] == 'Success', str((j2.status(), b.debug_info()))
    assert b.status()['state'] == 'cancelled', str((b.status(), b.debug_info()))


def test_update_jobs_are_not_serialized(client: BatchClient):
    b = create_batch(client)
    j1 = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '3600'])
    b.submit()

    j2 = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()

    j2.wait()

    wait_status = j1._wait_for_states('Running')
    if wait_status['state'] != 'Running':
        assert False, str((j1.status(), b.debug_info()))

    b.cancel()

    assert j2.status()['state'] == 'Success', str((j2.status(), b.debug_info()))
    assert b.status()['state'] == 'cancelled', str((b.status(), b.debug_info()))


def test_update_batch_w_empty_initial_batch(client: BatchClient):
    b = create_batch(client)
    b.submit()

    j1 = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()
    status = j1.wait()

    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_update_batch_w_multiple_empty_updates(client: BatchClient):
    b = create_batch(client)
    b.submit()
    b.submit()
    b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()
    status = b.wait()

    assert status['state'] == 'success', str((status, b.debug_info()))


def test_update_batch_w_new_batch_builder(client: BatchClient):
    b = create_batch(client)
    b.submit()
    b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()
    status = b.wait()

    assert status['state'] == 'success', str((status, b.debug_info()))


def test_update_batch_wout_fast_path(client: BatchClient):
    b = create_batch(client)
    b.submit()
    for _ in range(4):
        b.create_job(DOCKER_ROOT_IMAGE, ['echo', 'a' * (900 * 1024)])
    b.submit()
    assert not b._submission_info.used_fast_path


def test_update_cancelled_batch_wout_fast_path(client: BatchClient):
    b = create_batch(client)
    b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '3600'])
    b.submit()
    b.cancel()

    try:
        for _ in range(4):
            b.create_job(DOCKER_ROOT_IMAGE, ['echo', 'a' * (900 * 1024)])
        b.submit()
    except httpx.ClientResponseError as err:
        assert err.status == 400
        assert 'bunch contains job where the job group has already been cancelled' in err.body
    else:
        assert False


def test_submit_update_to_cancelled_batch(client: BatchClient):
    b = create_batch(client)
    b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '3600'])
    b.submit()
    b.cancel()

    try:
        b.create_job(DOCKER_ROOT_IMAGE, ['true'])
        b.submit()
    except httpx.ClientResponseError as err:
        assert err.status == 400
        assert 'bunch contains job where the job group has already been cancelled' in err.body
    else:
        assert False


def test_submit_update_to_deleted_batch(client: BatchClient):
    b = create_batch(client)
    b.submit()
    b.cancel()
    b.delete()

    try:
        b.create_job(DOCKER_ROOT_IMAGE, ['true'])
        b.submit()
    except httpx.ClientResponseError as err:
        assert err.status == 404
    else:
        assert False


@pytest.mark.timeout(24 * 60)
def test_region(client: BatchClient):
    CLOUD = os.environ['HAIL_CLOUD']

    b = create_batch(client)
    if CLOUD == 'gcp':
        region = 'us-east1'
    else:
        assert CLOUD == 'azure'
        region = 'eastus'
    j = b.create_job(DOCKER_ROOT_IMAGE, ['printenv', 'HAIL_REGION'], regions=[region])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert status['status']['region'] == region, str((status, b.debug_info()))
    assert region in j.log()['main'], str((status, b.debug_info()))


def test_get_job_group_status(client: BatchClient):
    b = create_batch(client)
    jg = b.create_job_group(attributes={'name': 'foo'})
    jg.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()

    status = jg.wait()
    last_known_status = jg.last_known_status()
    debug_info = jg.debug_info()

    jg_from_client = b.get_job_group(jg.job_group_id)
    jg_from_client_status = jg_from_client.status()

    assert status['batch_id'] == b.id, str(status)
    assert last_known_status['batch_id'] == b.id, str(last_known_status)
    assert debug_info['status']['batch_id'] == b.id, str(debug_info)
    assert jg_from_client_status['batch_id'] == b.id, str(jg_from_client_status)

    assert len(debug_info['jobs']) == 1, str(debug_info)
    assert len(list(jg.jobs())) == 1, str(debug_info)
    assert jg.attributes()['name'] == 'foo', str(debug_info)


def test_job_group_creation_with_no_jobs(client: BatchClient):
    b = create_batch(client)
    b.create_job_group(attributes={'name': 'foo'})
    b.submit()
    job_groups = list(b.job_groups())
    assert len(job_groups) == 1, str(job_groups)
    assert len(list(b.jobs())) == 0, str(b.debug_info())


def test_job_group_creation_on_update_with_no_jobs(client: BatchClient):
    b = create_batch(client)
    b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()
    b.create_job_group(attributes={'name': 'foo'})
    b.submit()

    jobs = list(b.jobs())
    job_groups = list(b.job_groups())
    assert len(job_groups) == 1, str(job_groups)
    assert job_groups[0].attributes()['name'] == 'foo', str(job_groups)
    assert len(jobs) == 1, str(jobs)
    b.cancel()


def test_job_group_attributes(client: BatchClient):
    b = create_batch(client)
    b.create_job_group(attributes={'name': 'foo', 'test': '1'})
    b.submit()
    job_groups = list(b.job_groups())
    assert len(job_groups) == 1, str(job_groups)
    jg = job_groups[0]
    assert jg.attributes() == {'name': 'foo', 'test': '1'}, str(jg.debug_info())


def test_job_groups_with_slow_create(client: BatchClient):
    b = create_batch(client)
    b.create_job_group(attributes={'name': 'foo'})
    for _ in range(4):
        b.create_job(DOCKER_ROOT_IMAGE, ['echo', 'a' * (900 * 1024)])
    b.submit()
    job_groups = list(b.job_groups())
    assert len(job_groups) == 1, str(job_groups)
    jobs = list(b.jobs())
    assert len(jobs) == 4, str(jobs)


def test_job_groups_with_slow_update(client: BatchClient):
    b = create_batch(client)
    jg = b.create_job_group(attributes={'name': 'foo'})
    b.submit()

    for _ in range(4):
        jg.create_job(DOCKER_ROOT_IMAGE, ['echo', 'a' * (900 * 1024)])
    b.submit()

    status = b.status()
    assert status['n_jobs'] == 4, str(b.debug_info())
    assert len(list(b.job_groups())) == 1, str(b.debug_info())


def test_more_than_one_bunch_of_job_groups_created(client: BatchClient):
    max_bunch_size = AioBatch.MAX_BUNCH_SIZE
    b = create_batch(client)
    for i in range(max_bunch_size + 1):
        b.create_job_group(attributes={'name': f'foo{i}'})
    b.submit()
    job_groups = list(b.job_groups())
    assert len(job_groups) == max_bunch_size + 1, str(job_groups)


def test_more_than_one_bunch_of_job_groups_updated(client: BatchClient):
    max_bunch_size = AioBatch.MAX_BUNCH_SIZE
    b = create_batch(client)
    b.create_job_group(attributes={'name': 'foo'})
    b.submit()
    for i in range(max_bunch_size + 1):
        b.create_job_group(attributes={'name': f'foo{i}'})
    b.submit()
    job_groups = list(b.job_groups())
    # need to include the initial job group created
    assert len(job_groups) == max_bunch_size + 2, str(job_groups)


def test_job_group_cancel_after_n_failures(client: BatchClient):
    b = create_batch(client)
    jg = b.create_job_group(cancel_after_n_failures=1)
    jg.create_job(DOCKER_ROOT_IMAGE, ['false'])
    j2 = jg.create_job(DOCKER_ROOT_IMAGE, ['sleep', '300'])
    b.submit()
    j2_status = j2.wait()
    jg_status = jg.wait()
    assert j2_status['state'] == 'Cancelled', str((j2_status, jg.debug_info()))
    assert jg_status['state'] == 'failure', str((jg_status, jg.debug_info()))


def test_cancel_job_group(client: BatchClient):
    b = create_batch(client)
    jg = b.create_job_group()
    head = jg.create_job(DOCKER_ROOT_IMAGE, ['sleep', '300'])
    tail = jg.create_job(DOCKER_ROOT_IMAGE, ['true'], parents=[head])
    b.submit()

    head._wait_for_states('Running')

    jg.cancel()
    b_status = b.wait()
    jg_status = jg.status()

    assert b_status['state'] == 'cancelled', str(b_status)
    assert jg_status['state'] == 'cancelled', str(jg_status)

    assert head.status()['state'] == 'Cancelled', str(head.status())
    assert tail.status()['state'] == 'Cancelled', str(tail.status())

    jg.create_job(DOCKER_ROOT_IMAGE, ['true'])
    with pytest.raises(
        httpx.ClientResponseError, match='bunch contains job where the job group has already been cancelled'
    ):
        b.submit()


def test_get_job_group_from_client_batch(client: BatchClient):
    b = create_batch(client)
    jg = b.create_job_group(attributes={'name': 'foo'})
    b.submit()

    b_copy = client.get_batch(b.id)
    jg_copy = b_copy.get_job_group(jg.job_group_id)
    jg_copy.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b_copy.submit()
    status = jg_copy.wait()
    assert status['n_jobs'] == 1, str(status)


def test_cancellation_doesnt_cancel_other_job_groups(client: BatchClient):
    b = create_batch(client)
    jg1 = b.create_job_group()
    j1 = jg1.create_job(DOCKER_ROOT_IMAGE, ['sleep', '300'])
    jg2 = b.create_job_group()
    j2 = jg2.create_job(DOCKER_ROOT_IMAGE, ['sleep', '300'])
    b.submit()

    j1._wait_for_states('Running')

    jg1.cancel()
    jg1_status = jg1.wait()
    jg2_status = jg2.status()

    # assert b.status()['state'] == 'cancelled', str(b.debug_info())  # FIXME???: n_cancelled jobs propogates upwards which might be confusing
    assert jg1_status['state'] == 'cancelled', str(jg1.debug_info())
    assert jg2_status['state'] != 'cancelled', str(jg2.debug_info())

    assert j1.status()['state'] == 'Cancelled', str(j1.status())
    assert j2.status()['state'] != 'Cancelled', str(j2.status())

    b.cancel()


def test_dependencies_across_job_groups(client: BatchClient):
    b = create_batch(client)
    jg1 = b.create_job_group()
    j1 = jg1.create_job(DOCKER_ROOT_IMAGE, ['true'])
    jg2 = b.create_job_group()
    jg2.create_job(DOCKER_ROOT_IMAGE, ['true'], parents=[j1])
    b.submit()
    status = b.wait()
    assert status['state'] == 'success', str(b.debug_info())


def test_job_group_cancel_after_n_failures_does_not_cancel_higher_up_jobs(client: BatchClient):
    b = create_batch(client)
    b_j = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '300'])
    jg = b.create_job_group(cancel_after_n_failures=1)
    jg.create_job(DOCKER_ROOT_IMAGE, ['false'])
    j2 = jg.create_job(DOCKER_ROOT_IMAGE, ['sleep', '300'])
    b.submit()
    j2_status = j2.wait()
    jg_status = jg.wait()
    b_j_status = b_j.status()
    try:
        assert b_j_status['state'] != 'Cancelled', str((b_j_status, b.debug_info()))
        assert j2_status['state'] == 'Cancelled', str((j2_status, jg.debug_info()))
        assert jg_status['state'] == 'failure', str((jg_status, jg.debug_info()))
    finally:
        b.cancel()
