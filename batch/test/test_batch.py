import collections
import os
import secrets
import time
from typing import Set

import aiohttp
import pytest

from hailtop.auth import service_auth_headers
from hailtop.batch_client.client import BatchClient
from hailtop.config import get_deploy_config, get_user_config
from hailtop.utils import external_requests_client_session, retry_response_returning_functions, sync_sleep_and_backoff

from .failure_injecting_client_session import FailureInjectingClientSession
from .utils import fails_in_azure, legacy_batch_status, skip_in_azure, smallest_machine_type

deploy_config = get_deploy_config()

DOCKER_PREFIX = os.environ['DOCKER_PREFIX']
DOCKER_ROOT_IMAGE = os.environ['DOCKER_ROOT_IMAGE']
UBUNTU_IMAGE = 'ubuntu:20.04'
DOMAIN = os.environ.get('HAIL_DOMAIN')
NAMESPACE = os.environ.get('HAIL_DEFAULT_NAMESPACE')
SCOPE = os.environ.get('HAIL_SCOPE', 'test')
CLOUD = os.environ.get('HAIL_CLOUD')


@pytest.fixture
def client():
    client = BatchClient('test')
    yield client
    client.close()


def test_job(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    b = builder.submit()

    status = j.wait()
    assert 'attributes' not in status, str((status, b.debug_info()))
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert status['exit_code'] == 0, str((status, b.debug_info()))
    assert j._get_exit_code(status, 'main') == 0, str((status, b.debug_info()))
    job_log = j.log()
    assert job_log['main'] == 'test\n', str((job_log, b.debug_info()))


def test_job_running_logs(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['bash', '-c', 'echo test && sleep 300'])
    b = builder.submit()

    delay = 1
    while True:
        status = j.status()
        if status['state'] == 'Running':
            log = j.log()
            if log is not None:
                assert log['main'] == 'test\n', str((log, b.debug_info()))
                break
        delay = sync_sleep_and_backoff(delay)

    b.cancel()
    b.wait()


def test_exit_code_duration(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['bash', '-c', 'exit 7'])
    b = builder.submit()
    status = j.wait()
    assert status['exit_code'] == 7, str((status, b.debug_info()))
    assert isinstance(status['duration'], int), str((status, b.debug_info()))
    assert j._get_exit_code(status, 'main') == 7, str((status, b.debug_info()))


def test_attributes(client: BatchClient):
    a = {'name': 'test_attributes', 'foo': 'bar'}
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], attributes=a)
    b = builder.submit()
    assert j.attributes() == a, str(b.debug_info())


def test_garbage_image(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job('dsafaaadsf', ['echo', 'test'])
    b = builder.submit()
    status = j.wait()
    assert j._get_exit_codes(status) == {'main': None}, str((status, b.debug_info()))
    assert j._get_error(status, 'main') is not None, str((status, b.debug_info()))
    assert status['state'] == 'Error', str((status, b.debug_info()))


def test_bad_command(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['sleep 5'])
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))


def test_invalid_resource_requests(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '1', 'memory': '250Gi', 'storage': '1Gi'}
    builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(aiohttp.client.ClientResponseError, match='resource requests.*unsatisfiable'):
        builder.submit()

    builder = client.create_batch()
    resources = {'cpu': '0', 'memory': '1Gi', 'storage': '1Gi'}
    builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(
        aiohttp.client.ClientResponseError,
        match='bad resource request for job.*cpu must be a power of two with a min of 0.25; found.*',
    ):
        builder.submit()

    builder = client.create_batch()
    resources = {'cpu': '0.1', 'memory': '1Gi', 'storage': '1Gi'}
    builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(
        aiohttp.client.ClientResponseError,
        match='bad resource request for job.*cpu must be a power of two with a min of 0.25; found.*',
    ):
        builder.submit()

    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': 'foo', 'storage': '1Gi'}
    builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(
        aiohttp.client.ClientResponseError,
        match=".*.resources.memory must match regex:.*.resources.memory must be one of:.*",
    ):
        builder.submit()

    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '500Mi', 'storage': '10000000Gi'}
    builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(aiohttp.client.ClientResponseError, match='resource requests.*unsatisfiable'):
        builder.submit()

    builder = client.create_batch()
    resources = {'storage': '10000000Gi', 'machine_type': smallest_machine_type(CLOUD)}
    builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(aiohttp.client.ClientResponseError, match='resource requests.*unsatisfiable'):
        builder.submit()


def test_out_of_memory(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '10Gi'}
    j = builder.create_job('python:3.6-slim-stretch', ['python', '-c', 'x = "a" * 1000**3'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert j._get_out_of_memory(status, 'main'), str((status, b.debug_info()))


def test_out_of_storage(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '5Gi'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'fallocate -l 100GiB /foo'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    job_log = j.log()
    assert "fallocate failed: No space left on device" in job_log['main']


def test_quota_applies_to_volume(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '5Gi'}
    j = builder.create_job(
        os.environ['HAIL_VOLUME_IMAGE'], ['/bin/sh', '-c', 'fallocate -l 100GiB /data/foo'], resources=resources
    )
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    job_log = j.log()
    assert "fallocate failed: No space left on device" in job_log['main']


def test_quota_shared_by_io_and_rootfs(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '10Gi'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'fallocate -l 7GiB /foo'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))

    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '10Gi'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'fallocate -l 7GiB /io/foo'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))

    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '10Gi'}
    j = builder.create_job(
        DOCKER_ROOT_IMAGE,
        ['/bin/sh', '-c', 'fallocate -l 7GiB /foo; fallocate -l 7GiB /io/foo'],
        resources=resources,
    )
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    job_log = j.log()
    assert "fallocate failed: No space left on device" in job_log['main'], str((job_log, b.debug_info()))


def test_nonzero_storage(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '20Gi'}
    j = builder.create_job(UBUNTU_IMAGE, ['/bin/sh', '-c', 'true'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


@skip_in_azure()
def test_attached_disk(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '400Gi'}
    j = builder.create_job(UBUNTU_IMAGE, ['/bin/sh', '-c', 'df -h; fallocate -l 390GiB /io/foo'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_cwd_from_image_workdir(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job(os.environ['HAIL_WORKDIR_IMAGE'], ['/bin/sh', '-c', 'pwd'])
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    job_log = j.log()
    assert "/work" in job_log['main'], str((job_log, b.debug_info()))


def test_unsubmitted_state(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])

    with pytest.raises(ValueError):
        j.batch_id
    with pytest.raises(ValueError):
        j.id
    with pytest.raises(ValueError):
        j.status()
    with pytest.raises(ValueError):
        j.is_complete()
    with pytest.raises(ValueError):
        j.log()
    with pytest.raises(ValueError):
        j.wait()

    builder.submit()
    with pytest.raises(ValueError):
        builder.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])


def test_list_batches(client: BatchClient):
    tag = secrets.token_urlsafe(64)
    b1 = client.create_batch(attributes={'tag': tag, 'name': 'b1'})
    b1.create_job(DOCKER_ROOT_IMAGE, ['sleep', '3600'])
    b1 = b1.submit()

    b2 = client.create_batch(attributes={'tag': tag, 'name': 'b2'})
    b2.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    b2 = b2.submit()

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


def test_list_jobs(client: BatchClient):
    b = client.create_batch()
    j_success = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    j_failure = b.create_job(DOCKER_ROOT_IMAGE, ['false'])
    j_error = b.create_job(DOCKER_ROOT_IMAGE, ['sleep 5'], attributes={'tag': 'bar'})
    j_running = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '1800'], attributes={'tag': 'foo'})

    b = b.submit()
    j_success.wait()
    j_failure.wait()
    j_error.wait()

    def assert_job_ids(expected, q=None):
        jobs = b.jobs(q=q)
        actual = set([j['job_id'] for j in jobs])
        assert actual == expected, str((jobs, b.debug_info()))

    assert_job_ids({j_success.job_id}, 'success')
    assert_job_ids({j_success.job_id, j_failure.job_id, j_error.job_id}, 'done')
    assert_job_ids({j_running.job_id}, '!done')
    assert_job_ids({j_running.job_id}, 'tag=foo')
    assert_job_ids({j_error.job_id, j_running.job_id}, 'has:tag')
    assert_job_ids({j_success.job_id, j_failure.job_id, j_error.job_id, j_running.job_id}, None)

    b.cancel()


def test_include_jobs(client: BatchClient):
    b1 = client.create_batch()
    for i in range(2):
        b1.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b1 = b1.submit()
    s = b1.status()
    assert 'jobs' not in s, str((s, b1.debug_info()))


def test_fail(client: BatchClient):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['false'])
    b = b.submit()
    status = j.wait()
    assert j._get_exit_code(status, 'main') == 1, str((status, b.debug_info()))


def test_unknown_image(client: BatchClient):
    b = client.create_batch()
    j = b.create_job(f'{DOCKER_PREFIX}/does-not-exist', ['echo', 'test'])
    b = b.submit()
    status = j.wait()
    assert j._get_exit_code(status, 'main') is None
    assert status['status']['container_statuses']['main']['short_error'] == 'image not found', str(
        (status, b.debug_info())
    )


def test_running_job_log_and_status(client: BatchClient):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '300'])
    b = b.submit()

    while True:
        if j.status()['state'] == 'Running' or j.is_complete():
            break

    j.log()
    # FIXME after batch1 goes away, check running status
    b.cancel()


def test_deleted_job_log(client: BatchClient):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    b = b.submit()
    j.wait()
    b.delete()

    try:
        j.log()
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            pass
        else:
            assert False, str((e, b.debug_info()))


def test_delete_batch(client: BatchClient):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b = b.submit()
    b.delete()

    # verify doesn't exist
    try:
        client.get_job(*j.id)
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            pass
        else:
            raise


def test_cancel_batch(client: BatchClient):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b = b.submit()

    status = j.status()
    assert status['state'] in ('Ready', 'Running'), str((status, b.debug_info()))

    b.cancel()

    status = j.wait()
    assert status['state'] == 'Cancelled', str((status, b.debug_info()))
    assert 'log' not in status, str((status, b.debug_info()))

    # cancelled job has no log
    try:
        j.log()
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            pass
        else:
            raise


def test_get_nonexistent_job(client: BatchClient):
    try:
        client.get_job(1, 666)
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            pass
        else:
            raise


def test_get_job(client: BatchClient):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b = b.submit()

    j2 = client.get_job(*j.id)
    status2 = j2.status()
    assert (status2['batch_id'], status2['job_id']) == j.id, str((status2, b.debug_info()))


def test_batch(client: BatchClient):
    b = client.create_batch()
    j1 = b.create_job(DOCKER_ROOT_IMAGE, ['false'])
    j2 = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '1'])
    j3 = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b = b.submit()

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

    n_failed = sum([j['exit_code'] > 0 for j in bstatus['jobs'] if j['state'] in ('Failed', 'Error')])
    assert n_failed == 1, str((bstatus, b.debug_info()))


def test_batch_status(client: BatchClient):
    b1 = client.create_batch()
    b1.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b1 = b1.submit()
    b1.wait()
    b1s = b1.status()
    assert b1s['complete'] and b1s['state'] == 'success', str((b1s, b1.debug_info()))

    b2 = client.create_batch()
    b2.create_job(DOCKER_ROOT_IMAGE, ['false'])
    b2.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b2 = b2.submit()
    b2.wait()
    b2s = b2.status()
    assert b2s['complete'] and b2s['state'] == 'failure', str((b2s, b2.debug_info()))

    b3 = client.create_batch()
    b3.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b3 = b3.submit()
    b3s = b3.status()
    assert not b3s['complete'] and b3s['state'] == 'running', str((b3s, b3.debug_info()))
    b3.cancel()

    b4 = client.create_batch()
    b4.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b4 = b4.submit()
    b4.cancel()
    b4.wait()
    b4s = b4.status()
    assert b4s['complete'] and b4s['state'] == 'cancelled', str((b4s, b4.debug_info()))


def test_log_after_failing_job(client: BatchClient):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'echo test; exit 127'])
    b = b.submit()
    status = j.wait()
    assert 'attributes' not in status, str((status, b.debug_info()))
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    assert j._get_exit_code(status, 'main') == 127, str((status, b.debug_info()))

    job_log = j.log()
    assert job_log['main'] == 'test\n', str((job_log, b.debug_info()))

    assert j.is_complete(), str(b.debug_info())


def test_long_log_line(client: BatchClient):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'for _ in {0..70000}; do echo -n a; done'])
    b = b.submit()
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
    builder = client.create_batch()
    j = builder.create_job(os.environ['HAIL_CURL_IMAGE'], ['echo', 'test'])
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_service_account(client: BatchClient):
    b = client.create_batch()
    j = b.create_job(
        os.environ['CI_UTILS_IMAGE'],
        ['/bin/sh', '-c', 'kubectl version'],
        service_account={'namespace': NAMESPACE, 'name': 'test-batch-sa'},
    )
    b = b.submit()
    status = j.wait()
    assert j._get_exit_code(status, 'main') == 0, str((status, b.debug_info()))


def test_port(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job(
        DOCKER_ROOT_IMAGE,
        [
            'bash',
            '-c',
            '''
echo $HAIL_BATCH_WORKER_PORT
echo $HAIL_BATCH_WORKER_IP
''',
        ],
        port=5000,
    )
    b = builder.submit()
    batch = b.wait()
    assert batch['state'] == 'success', str((batch, b.debug_info()))


def test_timeout(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'], timeout=5)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Error', str((status, b.debug_info()))
    error_msg = j._get_error(status, 'main')
    assert error_msg and 'JobTimeoutError' in error_msg, str((error_msg, b.debug_info()))
    assert j.exit_code(status) is None, str((status, b.debug_info()))


def test_client_max_size(client: BatchClient):
    builder = client.create_batch()
    for i in range(4):
        builder.create_job(DOCKER_ROOT_IMAGE, ['echo', 'a' * (900 * 1024)])
    builder.submit()


def test_restartable_insert(client: BatchClient):
    i = 0

    def every_third_time():
        nonlocal i
        i += 1
        if i % 3 == 0:
            return True
        return False

    with FailureInjectingClientSession(every_third_time) as session:
        client = BatchClient('test', session=session)
        builder = client.create_batch()

        for _ in range(9):
            builder.create_job(DOCKER_ROOT_IMAGE, ['echo', 'a'])

        b = builder.submit(max_bunch_size=1)
        b = client.get_batch(b.id)  # get a batch untainted by the FailureInjectingClientSession
        status = b.wait()
        assert status['state'] == 'success', str((status, b.debug_info()))
        jobs = list(b.jobs())
        assert len(jobs) == 9, str((jobs, b.debug_info()))


def test_create_idempotence(client: BatchClient):
    token = secrets.token_urlsafe(32)
    builder1 = client.create_batch(token=token)
    builder2 = client.create_batch(token=token)
    b1 = builder1._open_batch()
    b2 = builder2._open_batch()
    assert b1.id == b2.id


def test_batch_create_validation():
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
    headers = service_auth_headers(deploy_config, 'batch')
    session = external_requests_client_session()
    for config in bad_configs:
        r = retry_response_returning_functions(session.post, url, json=config, allow_redirects=True, headers=headers)
        assert r.status_code == 400, (config, r)


def test_duplicate_parents(client: BatchClient):
    batch = client.create_batch()
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[head, head])
    try:
        batch = batch.submit()
    except aiohttp.ClientResponseError as e:
        assert e.status == 400
    else:
        assert False, f'should receive a 400 Bad Request {batch.id}'


@skip_in_azure()
def test_verify_no_access_to_google_metadata_server(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job(
        os.environ['HAIL_CURL_IMAGE'], ['curl', '-fsSL', 'metadata.google.internal', '--max-time', '10']
    )
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    job_log = j.log()
    assert "Could not resolve host" in job_log['main'], str((job_log, b.debug_info()))


def test_verify_no_access_to_metadata_server(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job(os.environ['HAIL_CURL_IMAGE'], ['curl', '-fsSL', '169.254.169.254', '--max-time', '10'])
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    job_log = j.log()
    assert "Connection timed out" in job_log['main'], str((job_log, b.debug_info()))


def test_submit_batch_in_job(client: BatchClient):
    builder = client.create_batch()
    remote_tmpdir = get_user_config().get('batch', 'remote_tmpdir')
    script = f'''import hailtop.batch as hb
backend = hb.ServiceBackend("test", remote_tmpdir="{remote_tmpdir}")
b = hb.Batch(backend=backend)
j = b.new_bash_job()
j.command("echo hi")
b.run()
backend.close()
'''
    j = builder.create_job(
        os.environ['HAIL_HAIL_BASE_IMAGE'],
        ['/bin/bash', '-c', f'''python3 -c \'{script}\''''],
        mount_tokens=True,
    )
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_cant_submit_to_default_with_other_ns_creds(client: BatchClient):
    remote_tmpdir = get_user_config().get('batch', 'remote_tmpdir')
    script = f'''import hailtop.batch as hb
backend = hb.ServiceBackend("test", remote_tmpdir="{remote_tmpdir}")
b = hb.Batch(backend=backend)
j = b.new_bash_job()
j.command("echo hi")
b.run()
backend.close()
'''

    builder = client.create_batch()
    j = builder.create_job(
        os.environ['HAIL_HAIL_BASE_IMAGE'],
        [
            '/bin/bash',
            '-c',
            f'''
hailctl config set domain {DOMAIN}
rm /deploy-config/deploy-config.json
python3 -c \'{script}\'''',
        ],
        mount_tokens=True,
    )
    b = builder.submit()
    status = j.wait()
    if NAMESPACE == 'default':
        assert status['state'] == 'Success', str((status, b.debug_info()))
    else:
        assert status['state'] == 'Failed', str((status, b.debug_info()))
        assert "Please log in" in j.log()['main'], (str(j.log()['main']), status)

    builder = client.create_batch()
    j = builder.create_job(
        os.environ['HAIL_HAIL_BASE_IMAGE'],
        [
            '/bin/bash',
            '-c',
            f'''
jq '.default_namespace = "default"' /deploy-config/deploy-config.json > tmp.json
mv tmp.json /deploy-config/deploy-config.json
python3 -c \'{script}\'''',
        ],
        mount_tokens=True,
    )
    b = builder.submit()
    status = j.wait()
    if NAMESPACE == 'default':
        assert status['state'] == 'Success', str((status, b.debug_info()))
    else:
        assert status['state'] == 'Failed', str((status, b.debug_info()))
        job_log = j.log()
        assert "Please log in" in job_log['main'], str((job_log, b.debug_info()))


def test_cannot_contact_other_internal_ips(client: BatchClient):
    internal_ips = [f'10.128.0.{i}' for i in (10, 11, 12)]
    builder = client.create_batch()
    script = f'''
if [ "$HAIL_BATCH_WORKER_IP" != "{internal_ips[0]}" ] && ! grep -Fq {internal_ips[0]} /etc/hosts; then
    OTHER_IP={internal_ips[0]}
elif [ "$HAIL_BATCH_WORKER_IP" != "{internal_ips[1]}" ] && ! grep -Fq {internal_ips[1]} /etc/hosts; then
    OTHER_IP={internal_ips[1]}
else
    OTHER_IP={internal_ips[2]}
fi

curl -fsSL -m 5 $OTHER_IP
'''
    j = builder.create_job(os.environ['HAIL_CURL_IMAGE'], ['/bin/bash', '-c', script], port=5000)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str((status, b.debug_info()))
    job_log = j.log()
    assert "Connection timed out" in job_log['main'], str((job_log, b.debug_info()))


@skip_in_azure()
def test_can_use_google_credentials(client: BatchClient):
    token = os.environ["HAIL_TOKEN"]
    remote_tmpdir = get_user_config().get('batch', 'remote_tmpdir')
    builder = client.create_batch()
    script = f'''import hail as hl
import secrets
attempt_token = secrets.token_urlsafe(5)
location = f"{remote_tmpdir}/{ token }/{{ attempt_token }}/test_can_use_hailctl_auth.t"
hl.utils.range_table(10).write(location)
hl.read_table(location).show()
'''
    j = builder.create_job(
        os.environ['HAIL_HAIL_BASE_IMAGE'], ['/bin/bash', '-c', f'python3 -c >out 2>err \'{script}\'; cat out err']
    )
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', f'{j.log(), status}'
    expected_log = '''+-------+
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
'''
    log = j.log()
    assert expected_log in log['main'], str((log, b.debug_info()))


def test_user_authentication_within_job(client: BatchClient):
    batch = client.create_batch()
    cmd = ['bash', '-c', 'hailctl auth user']
    no_token = batch.create_job(os.environ['CI_UTILS_IMAGE'], cmd, mount_tokens=False)
    b = batch.submit()

    no_token_status = no_token.wait()
    assert no_token_status['state'] == 'Failed', str((no_token_status, b.debug_info()))


def test_verify_access_to_public_internet(client: BatchClient):
    builder = client.create_batch()
    j = builder.create_job(os.environ['HAIL_CURL_IMAGE'], ['curl', '-fsSL', 'example.com'])
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


def test_verify_can_tcp_to_localhost(client: BatchClient):
    builder = client.create_batch()
    script = '''
set -e
nc -l -p 5000 &
sleep 5
echo "hello" | nc -q 1 localhost 5000
'''.lstrip(
        '\n'
    )
    j = builder.create_job(os.environ['HAIL_NETCAT_UBUNTU_IMAGE'], command=['/bin/bash', '-c', script])
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    job_log = j.log()
    assert 'hello\n' == job_log['main'], str((job_log, b.debug_info()))


def test_verify_can_tcp_to_127_0_0_1(client: BatchClient):
    builder = client.create_batch()
    script = '''
set -e
nc -l -p 5000 &
sleep 5
echo "hello" | nc -q 1 127.0.0.1 5000
'''.lstrip(
        '\n'
    )
    j = builder.create_job(os.environ['HAIL_NETCAT_UBUNTU_IMAGE'], command=['/bin/bash', '-c', script])
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    job_log = j.log()
    assert 'hello\n' == job_log['main'], str((job_log, b.debug_info()))


def test_verify_can_tcp_to_self_ip(client: BatchClient):
    builder = client.create_batch()
    script = '''
set -e
nc -l -p 5000 &
sleep 5
echo "hello" | nc -q 1 $(hostname -i) 5000
'''.lstrip(
        '\n'
    )
    j = builder.create_job(os.environ['HAIL_NETCAT_UBUNTU_IMAGE'], command=['/bin/sh', '-c', script])
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    job_log = j.log()
    assert 'hello\n' == job_log['main'], str((job_log, b.debug_info()))


def test_verify_private_network_is_restricted(client: BatchClient):
    builder = client.create_batch()
    builder.create_job(
        os.environ['HAIL_CURL_IMAGE'], command=['curl', 'internal.hail', '--connect-timeout', '60'], network='private'
    )
    try:
        builder.submit()
    except aiohttp.ClientResponseError as err:
        assert err.status == 400
        assert 'unauthorized network private' in err.message
    else:
        assert False


def test_pool_highmem_instance(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': 'highmem'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'highmem' in status['status']['worker'], str((status, b.debug_info()))


def test_pool_highmem_instance_cheapest(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '1', 'memory': '5Gi'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'highmem' in status['status']['worker'], str((status, b.debug_info()))


def test_pool_highcpu_instance(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': 'lowmem'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'highcpu' in status['status']['worker'], str((status, b.debug_info()))


def test_pool_highcpu_instance_cheapest(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '50Mi'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'highcpu' in status['status']['worker'], str((status, b.debug_info()))


def test_pool_standard_instance(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': 'standard'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'standard' in status['status']['worker'], str((status, b.debug_info()))


def test_pool_standard_instance_cheapest(client: BatchClient):
    builder = client.create_batch()
    resources = {'cpu': '1', 'memory': '2.5Gi'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'standard' in status['status']['worker'], str((status, b.debug_info()))


def test_job_private_instance_preemptible(client: BatchClient):
    builder = client.create_batch()
    resources = {'machine_type': smallest_machine_type(CLOUD)}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'job-private' in status['status']['worker'], str((status, b.debug_info()))


def test_job_private_instance_nonpreemptible(client: BatchClient):
    builder = client.create_batch()
    resources = {'machine_type': smallest_machine_type(CLOUD), 'preemptible': False}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
    assert 'job-private' in status['status']['worker'], str((status, b.debug_info()))


def test_job_private_instance_cancel(client: BatchClient):
    builder = client.create_batch()
    resources = {'machine_type': smallest_machine_type(CLOUD)}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    b = builder.submit()

    delay = 0.1
    start = time.time()
    while True:
        status = j.status()
        if status['state'] == 'Creating':
            break
        now = time.time()
        if now + delay - start > 60:
            assert False, str((status, b.debug_info()))
        delay = sync_sleep_and_backoff(delay)
    b.cancel()
    status = j.wait()
    assert status['state'] == 'Cancelled', str((status, b.debug_info()))
