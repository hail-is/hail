import random
import datetime
import math
import collections
import os
import secrets
import time
import aiohttp
import pytest

from hailtop.config import get_deploy_config, get_user_config
from hailtop.auth import service_auth_headers
from hailtop.utils import retry_response_returning_functions, external_requests_client_session, sync_sleep_and_backoff
from hailtop.batch_client.client import BatchClient

from .utils import legacy_batch_status
from .failure_injecting_client_session import FailureInjectingClientSession

deploy_config = get_deploy_config()

DOCKER_ROOT_IMAGE = os.environ.get('DOCKER_ROOT_IMAGE', 'gcr.io/hail-vdc/ubuntu:18.04')
SCOPE = os.environ.get('HAIL_SCOPE', 'test')


def poll_until(p, max_polls=None):
    i = 0
    while True and (max_polls is None or i < max_polls):
        x = p()
        if x:
            return x
        # max 4.5s
        j = random.randrange(math.floor(1.1 ** min(i, 40)))
        time.sleep(0.100 * j)
        i = i + 1
    raise ValueError(f'poll_until: exceeded max polls: {i} {max_polls}')


@pytest.fixture
def client():
    client = BatchClient('test')
    yield client
    client.close()


def test_job(client):
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    b = builder.submit()
    status = j.wait()
    assert 'attributes' not in status, (status, j.log())
    assert status['state'] == 'Success', (status, j.log())
    assert status['exit_code'] == 0, status
    assert j._get_exit_code(status, 'main') == 0, (status, j.log())
    assert j.log()['main'] == 'test\n', status


def test_exit_code_duration(client):
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['bash', '-c', 'exit 7'])
    b = builder.submit()
    status = j.wait()
    assert status['exit_code'] == 7, status
    assert isinstance(status['duration'], int)
    assert j._get_exit_code(status, 'main') == 7, status


def test_attributes(client):
    a = {'name': 'test_attributes', 'foo': 'bar'}
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], attributes=a)
    builder.submit()
    assert j.attributes() == a


def test_garbage_image(client):
    builder = client.create_batch()
    j = builder.create_job('dsafaaadsf', ['echo', 'test'])
    builder.submit()
    status = j.wait()
    assert j._get_exit_codes(status) == {'main': None}, status
    assert j._get_error(status, 'main') is not None
    assert status['state'] == 'Error', str(status)


def test_bad_command(client):
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['sleep 5'])
    builder.submit()
    status = j.wait()
    assert j._get_exit_codes(status) == {'main': None}, status
    assert j._get_error(status, 'main') is not None
    assert status['state'] == 'Error', str(status)


def test_invalid_resource_requests(client):
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
    resources = {'cpu': '0.1', 'memory': 'foo', 'storage': '1Gi'}
    builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    with pytest.raises(
        aiohttp.client.ClientResponseError,
        match=".*.resources.memory must match regex:.*.resources.memory must be one of:.*",
    ):
        builder.submit()


def test_out_of_memory(client):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '10Gi'}
    j = builder.create_job('python:3.6-slim-stretch', ['python', '-c', 'x = "a" * 1000**3'], resources=resources)
    builder.submit()
    status = j.wait()
    assert j._get_out_of_memory(status, 'main'), str(status)


def test_out_of_storage(client):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '5Gi'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'fallocate -l 100GiB /foo'], resources=resources)
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str(status)
    assert "fallocate failed: No space left on device" in j.log()['main']


def test_nonzero_storage(client):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '20Gi'}
    j = builder.create_job('ubuntu:18.04', ['/bin/sh', '-c', 'true'], resources=resources)
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(status)


def test_attached_disk(client):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '10M', 'storage': '400Gi'}
    j = builder.create_job('ubuntu:18.04', ['/bin/sh', '-c', 'df -h; fallocate -l 390GiB /io/foo'], resources=resources)
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, j.log()))


def test_unsubmitted_state(client):
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


def test_list_batches(client):
    tag = secrets.token_urlsafe(64)
    b1 = client.create_batch(attributes={'tag': tag, 'name': 'b1'})
    b1.create_job(DOCKER_ROOT_IMAGE, ['sleep', '3600'])
    b1 = b1.submit()

    b2 = client.create_batch(attributes={'tag': tag, 'name': 'b2'})
    b2.create_job(DOCKER_ROOT_IMAGE, ['echo', 'test'])
    b2 = b2.submit()

    def assert_batch_ids(expected, q=None):
        # list_batches returns all batches for all prev run tests so we set a limit
        batches = client.list_batches(q, limit=200)
        actual = set([b.id for b in batches]).intersection({b1.id, b2.id})
        assert actual == expected

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


def test_list_jobs(client):
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
        actual = set([j['job_id'] for j in b.jobs(q=q)])
        assert actual == expected

    assert_job_ids({j_success.job_id}, 'success')
    assert_job_ids({j_success.job_id, j_failure.job_id, j_error.job_id}, 'done')
    assert_job_ids({j_running.job_id}, '!done')
    assert_job_ids({j_running.job_id}, 'tag=foo')
    assert_job_ids({j_error.job_id, j_running.job_id}, 'has:tag')
    assert_job_ids({j_success.job_id, j_failure.job_id, j_error.job_id, j_running.job_id}, None)

    b.cancel()


def test_include_jobs(client):
    b1 = client.create_batch()
    for i in range(2):
        b1.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b1 = b1.submit()
    s = b1.status()
    assert 'jobs' not in s


def test_fail(client):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['false'])
    b.submit()
    status = j.wait()
    assert j._get_exit_code(status, 'main') == 1, str(status)


def test_running_job_log_and_status(client):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '300'])
    b = b.submit()

    while True:
        if j.status()['state'] == 'Running' or j.is_complete():
            break

    j.log()
    # FIXME after batch1 goes away, check running status
    b.cancel()


def test_deleted_job_log(client):
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
            assert False, f"batch should have deleted log {e}"


def test_delete_batch(client):
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


def test_cancel_batch(client):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b = b.submit()

    status = j.status()
    assert status['state'] in ('Ready', 'Running'), str(status)

    b.cancel()

    status = j.wait()
    assert status['state'] == 'Cancelled', str(status)
    assert 'log' not in status, str(status)

    # cancelled job has no log
    try:
        j.log()
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            pass
        else:
            raise


def test_get_nonexistent_job(client):
    try:
        client.get_job(1, 666)
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            pass
        else:
            raise


def test_get_job(client):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b.submit()

    j2 = client.get_job(*j.id)
    status2 = j2.status()
    assert (status2['batch_id'], status2['job_id']) == j.id, str(status2)


def test_batch(client):
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

    assert len(bstatus['jobs']) == 3, str(bstatus)
    state_count = collections.Counter([j['state'] for j in bstatus['jobs']])
    n_cancelled = state_count['Cancelled']
    n_complete = state_count['Error'] + state_count['Failed'] + state_count['Success']
    assert n_cancelled <= 1, str(bstatus)
    assert n_cancelled + n_complete == 3, str(bstatus)

    n_failed = sum([j['exit_code'] > 0 for j in bstatus['jobs'] if j['state'] in ('Failed', 'Error')])
    assert n_failed == 1, str(bstatus)


def test_batch_status(client):
    b1 = client.create_batch()
    b1.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b1 = b1.submit()
    b1.wait()
    b1s = b1.status()
    assert b1s['complete'] and b1s['state'] == 'success', str(b1s)

    b2 = client.create_batch()
    b2.create_job(DOCKER_ROOT_IMAGE, ['false'])
    b2.create_job(DOCKER_ROOT_IMAGE, ['true'])
    b2 = b2.submit()
    b2.wait()
    b2s = b2.status()
    assert b2s['complete'] and b2s['state'] == 'failure', str(b2s)

    b3 = client.create_batch()
    b3.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b3 = b3.submit()
    b3s = b3.status()
    assert not b3s['complete'] and b3s['state'] == 'running', str(b3s)
    b3.cancel()

    b4 = client.create_batch()
    b4.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'])
    b4 = b4.submit()
    b4.cancel()
    b4.wait()
    b4s = b4.status()
    assert b4s['complete'] and b4s['state'] == 'cancelled', str(b4s)


def test_log_after_failing_job(client):
    b = client.create_batch()
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', 'echo test; exit 127'])
    b.submit()
    status = j.wait()
    assert 'attributes' not in status
    assert status['state'] == 'Failed'
    assert j._get_exit_code(status, 'main') == 127

    assert j.log()['main'] == 'test\n'

    assert j.is_complete()


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


def test_gcr_image(client):
    builder = client.create_batch()
    j = builder.create_job(os.environ['HAIL_CURL_IMAGE'], ['echo', 'test'])
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(status, j.log())


def test_service_account(client):
    b = client.create_batch()
    j = b.create_job(
        os.environ['CI_UTILS_IMAGE'],
        ['/bin/sh', '-c', 'kubectl version'],
        service_account={'namespace': os.environ['HAIL_DEFAULT_NAMESPACE'], 'name': 'test-batch-sa'},
    )
    b.submit()
    status = j.wait()
    assert j._get_exit_code(status, 'main') == 0, str(status)


def test_port(client):
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
    assert batch['state'] == 'success', str(batch)


def test_timeout(client):
    builder = client.create_batch()
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['sleep', '30'], timeout=5)
    b = builder.submit()
    status = j.wait()
    assert status['state'] == 'Error', (status, j.log())
    error_msg = j._get_error(status, 'main')
    assert error_msg and 'JobTimeoutError' in error_msg
    assert j.exit_code(status) is None, str(status)


def test_client_max_size(client):
    builder = client.create_batch()
    for i in range(4):
        builder.create_job(DOCKER_ROOT_IMAGE, ['echo', 'a' * (900 * 1024)])
    builder.submit()


def test_restartable_insert(client):
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
        batch = b.wait()
        assert batch['state'] == 'success', str(batch)
        assert len(list(b.jobs())) == 9


def test_create_idempotence(client):
    token = secrets.token_urlsafe(32)
    builder1 = client.create_batch(token=token)
    builder2 = client.create_batch(token=token)
    b1 = builder1._create()
    b2 = builder2._create()
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
        # attribute key/value None
        {'attributes': {'k': None}, 'billing_project': 'foo', 'n_jobs': 5, 'token': 'baz'},
    ]
    url = deploy_config.url('batch', '/api/v1alpha/batches/create')
    headers = service_auth_headers(deploy_config, 'batch')
    session = external_requests_client_session()
    for config in bad_configs:
        r = retry_response_returning_functions(session.post, url, json=config, allow_redirects=True, headers=headers)
        assert r.status_code == 400, (config, r)


def test_duplicate_parents(client):
    batch = client.create_batch()
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[head, head])
    try:
        batch = batch.submit()
    except aiohttp.ClientResponseError as e:
        assert e.status == 400
    else:
        assert False, f'should receive a 400 Bad Request {batch.id}'


def test_verify_no_access_to_metadata_server(client):
    builder = client.create_batch()
    j = builder.create_job(
        os.environ['HAIL_CURL_IMAGE'], ['curl', '-fsSL', 'metadata.google.internal', '--max-time', '10']
    )
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Failed', str(status)
    assert "Connection timed out" in j.log()['main'], str(j.log()['main'], status)


def test_can_use_google_credentials(client):
    token = os.environ["HAIL_TOKEN"]
    bucket_name = get_user_config().get('batch', 'bucket')
    builder = client.create_batch()
    script = f'''import hail as hl
import secrets
attempt_token = secrets.token_urlsafe(5)
location = f"gs://{ bucket_name }/{ token }/{{ attempt_token }}/test_can_use_hailctl_auth.t"
hl.utils.range_table(10).write(location)
hl.read_table(location).show()
'''
    j = builder.create_job(
        os.environ['HAIL_HAIL_BASE_IMAGE'], ['/bin/bash', '-c', f'python3 -c >out 2>err \'{script}\'; cat out err']
    )
    builder.submit()
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
    assert expected_log in log['main'], f'{j.log(), status}'


def test_user_authentication_within_job(client):
    batch = client.create_batch()
    cmd = ['bash', '-c', 'hailctl auth user']
    no_token = batch.create_job(os.environ['CI_UTILS_IMAGE'], cmd, mount_tokens=False)
    batch.submit()

    no_token_status = no_token.wait()
    assert no_token_status['state'] == 'Failed', f'{(no_token.log(), no_token_status)}'


def test_verify_access_to_public_internet(client):
    builder = client.create_batch()
    j = builder.create_job(os.environ['HAIL_CURL_IMAGE'], ['curl', '-fsSL', 'example.com'])
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', status


def test_verify_can_tcp_to_localhost(client):
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
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(j.log()['main'], status)
    assert 'hello\n' == j.log()['main']


def test_verify_can_tcp_to_127_0_0_1(client):
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
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(j.log()['main'], status)
    assert 'hello\n' == j.log()['main']


def test_verify_can_tcp_to_self_ip(client):
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
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(j.log()['main'], status)
    assert 'hello\n' == j.log()['main'], str(j.log())


def test_verify_private_network_is_restricted(client):
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


def test_pool_highmem_instance(client):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': 'highmem'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(j.log()['main'], status)
    assert 'highmem' in status['status']['worker'], str(status)

    builder = client.create_batch()
    resources = {'cpu': '1', 'memory': '5Gi'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(j.log()['main'], status)
    assert 'highmem' in status['status']['worker'], str(status)

    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '500Mi'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(j.log()['main'], status)
    assert 'standard' in status['status']['worker'], str(status)


def test_pool_highcpu_instance(client):
    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': 'lowmem'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(j.log()['main'], status)
    assert 'highcpu' in status['status']['worker'], str(status)

    builder = client.create_batch()
    resources = {'cpu': '0.25', 'memory': '50Mi'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(j.log()['main'], status)
    assert 'highcpu' in status['status']['worker'], str(status)

    builder = client.create_batch()
    resources = {'cpu': '0.5', 'memory': '1Gi'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(j.log()['main'], status)
    assert 'standard' in status['status']['worker'], str(status)


def test_job_private_instance_preemptible(client):
    builder = client.create_batch()
    resources = {'machine_type': 'n1-standard-1'}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(j.log()['main'], status)
    assert 'job-private' in status['status']['worker'], str(status)


def test_job_private_instance_nonpreemptible(client):
    builder = client.create_batch()
    resources = {'machine_type': 'n1-standard-1', 'preemptible': False}
    j = builder.create_job(DOCKER_ROOT_IMAGE, ['true'], resources=resources)
    builder.submit()
    status = j.wait()
    assert status['state'] == 'Success', str(j.log()['main'], status)
    assert 'job-private' in status['status']['worker'], str(status)


def test_job_private_instance_cancel(client):
    builder = client.create_batch()
    resources = {'machine_type': 'n1-standard-1'}
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
            assert False, f'timed out waiting for creating state: {status} {datetime.datetime.fromtimestamp(now)}'
        delay = sync_sleep_and_backoff(delay)
    b.cancel()
    status = j.wait()
    assert status['state'] == 'Cancelled', str(status)
