import time
import re
import os
import secrets
import pytest
from flask import Response
from hailtop.config import get_user_config
from hailtop.batch_client.client import BatchClient, Job
import hailtop.batch_client.aioclient as aioclient

from .utils import batch_status_job_counter, \
    legacy_batch_status
from .serverthread import ServerThread

DOCKER_ROOT_IMAGE = os.environ.get('DOCKER_ROOT_IMAGE', 'gcr.io/hail-vdc/ubuntu:18.04')


@pytest.fixture
def client():
    client = BatchClient('test')
    yield client
    client.close()


def test_simple(client):
    batch = client.create_batch()
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    tail = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[head])
    batch = batch.submit()
    batch.wait()
    status = legacy_batch_status(batch)
    assert batch_status_job_counter(status, 'Success') == 2, str(status)
    assert all([j['exit_code'] == 0 for j in status['jobs']])


def test_missing_parent_is_400(client):
    try:
        batch = client.create_batch()
        fake_job = aioclient.Job.unsubmitted_job(batch._async_builder, 10000)
        fake_job = Job.from_async_job(fake_job)
        batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'], parents=[fake_job])
        batch.submit()
    except ValueError as err:
        assert re.search('parents with invalid job ids', str(err))
        return
    assert False


def test_dag(client):
    batch = client.create_batch()
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    left = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'left'], parents=[head])
    right = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'right'], parents=[head])
    tail = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[left, right])
    batch = batch.submit()
    batch.wait()
    status = legacy_batch_status(batch)

    node_status_logs = []
    for node in [head, left, right, tail]:
        node_status_logs.append((node.status(), node.log()))

    assert batch_status_job_counter(status, 'Success') == 4, str(status, str(node_status_logs))

    for node in [head, left, right, tail]:
        status = node._status
        assert status['state'] == 'Success', str(status, node.log())
        assert node._get_exit_code(status, 'main') == 0


def test_cancel_tail(client):
    batch = client.create_batch()
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    left = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'left'], parents=[head])
    right = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'right'], parents=[head])
    tail = batch.create_job(
        DOCKER_ROOT_IMAGE,
        command=['/bin/sh', '-c', 'while true; do sleep 86000; done'],
        parents=[left, right])
    batch = batch.submit()
    left.wait()
    right.wait()
    batch.cancel()
    batch.wait()
    status = legacy_batch_status(batch)
    assert batch_status_job_counter(status, 'Success') == 3, str(status)
    for node in [head, left, right]:
        status = node.status()
        assert status['state'] == 'Success'
        assert node._get_exit_code(status, 'main') == 0
    assert tail.status()['state'] == 'Cancelled'


def test_cancel_left_after_tail(client):
    batch = client.create_batch()
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    left = batch.create_job(
        DOCKER_ROOT_IMAGE,
        command=['/bin/sh', '-c', 'while true; do sleep 86000; done'],
        parents=[head])
    right = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'right'], parents=[head])
    tail = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[left, right])
    batch = batch.submit()
    head.wait()
    right.wait()
    batch.cancel()
    batch.wait()
    status = legacy_batch_status(batch)
    assert batch_status_job_counter(status, 'Success') == 2, str(status)
    for node in [head, right]:
        status = node.status()
        assert status['state'] == 'Success'
        assert node._get_exit_code(status, 'main') == 0
    for node in [left, tail]:
        assert node.status()['state'] == 'Cancelled'


def test_callback(client):
    from flask import Flask, request
    app = Flask('test-client')
    callback_body = []

    @app.route('/test', methods=['POST'])
    def test():
        body = request.get_json()
        callback_body.append(body)
        return Response(status=200)

    try:
        server = ServerThread(app)
        server.start()
        token = secrets.token_urlsafe(32)
        b = client.create_batch(
            callback=server.url_for('/test'),
            attributes={'foo': 'bar'},
            token=token)
        head = b.create_job('alpine:3.8', command=['echo', 'head'])
        tail = b.create_job('alpine:3.8', command=['echo', 'tail'], parents=[head])
        b = b.submit()
        b.wait()

        i = 0
        while not callback_body:
            time.sleep(0.100 * (3/2) ** i)
            i += 1
            if i > 14:
                break
        callback_body = callback_body[0]

        # verify required fields present
        callback_body.pop('cost')
        callback_body.pop('msec_mcpu')
        callback_body.pop('time_created')
        callback_body.pop('time_closed')
        callback_body.pop('time_completed')
        callback_body.pop('duration')
        assert (callback_body == {
            'id': b.id,
            'user': 'test',
            'billing_project': 'test',
            'token': token,
            'state': 'success',
            'complete': True,
            'closed': True,
            'n_jobs': 2,
            'n_completed': 2,
            'n_succeeded': 2,
            'n_failed': 0,
            'n_cancelled': 0,
            'attributes': {'foo': 'bar'}
        }), callback_body
    finally:
        if server:
            server.shutdown()
            server.join()


def test_no_parents_allowed_in_other_batches(client):
    b1 = client.create_batch()
    b2 = client.create_batch()
    head = b1.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    try:
        b2.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[head])
    except ValueError as err:
        assert re.search('parents from another batch', str(err))
        return
    assert False


def test_input_dependency(client):
    bucket_name = get_user_config().get('batch', 'bucket')
    batch = client.create_batch()
    head = batch.create_job(DOCKER_ROOT_IMAGE,
                            command=['/bin/sh', '-c', 'echo head1 > /io/data1; echo head2 > /io/data2'],
                            output_files=[('/io/data1', f'gs://{bucket_name}/data1'),
                                          ('/io/data2', f'gs://{bucket_name}/data2')])
    tail = batch.create_job(DOCKER_ROOT_IMAGE,
                            command=['/bin/sh', '-c', 'cat /io/data1; cat /io/data2'],
                            input_files=[(f'gs://{bucket_name}/data1', '/io/data1'),
                                         (f'gs://{bucket_name}/data2', '/io/data2')],
                            parents=[head])
    batch.submit()
    tail.wait()
    assert head._get_exit_code(head.status(), 'main') == 0, str(head._status)
    assert tail.log()['main'] == 'head1\nhead2\n', str(tail.log(), tail.status())


def test_input_dependency_wildcard(client):
    bucket_name = get_user_config().get('batch', 'bucket')
    batch = client.create_batch()
    head = batch.create_job(DOCKER_ROOT_IMAGE,
                            command=['/bin/sh', '-c', 'echo head1 > /io/data1 ; echo head2 > /io/data2'],
                            output_files=[('/io/data1', f'gs://{bucket_name}/data1'),
                                          ('/io/data2', f'gs://{bucket_name}/data2')])
    tail = batch.create_job(DOCKER_ROOT_IMAGE,
                            command=['/bin/sh', '-c', 'cat /io/data1 ; cat /io/data2'],
                            input_files=[(f'gs://{bucket_name}/data1', '/io/data1'),
                                         (f'gs://{bucket_name}/data2', '/io/data2')],
                            parents=[head])
    batch.submit()
    tail.wait()
    assert head._get_exit_code(head.status(), 'input') != 0, str(head._status, head.log())
    assert tail.log()['main'] == 'head1\nhead2\n', str(tail.status(), tail.log())


def test_input_dependency_directory(client):
    bucket_name = get_user_config().get('batch', 'bucket')
    batch = client.create_batch()
    head = batch.create_job(DOCKER_ROOT_IMAGE,
                            command=['/bin/sh', '-c', 'mkdir -p /io/test/; echo head1 > /io/test/data1 ; echo head2 > /io/test/data2'],
                            output_files=[('/io/test', f'gs://{bucket_name}/test')])
    tail = batch.create_job(DOCKER_ROOT_IMAGE,
                            command=['/bin/sh', '-c', 'cat /io/test/data1; cat /io/test/data2'],
                            input_files=[(f'gs://{bucket_name}/test', '/io/test')],
                            parents=[head])
    batch.submit()
    tail.wait()
    assert head._get_exit_code(head.status(), 'main') == 0, str(head._status, head.log())
    assert tail.log()['main'] == 'head1\nhead2\n', str(tail.status(), tail.log())


def test_always_run_cancel(client):
    batch = client.create_batch()
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    left = batch.create_job(
        DOCKER_ROOT_IMAGE,
        command=['/bin/sh', '-c', 'while true; do sleep 86000; done'],
        parents=[head])
    right = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'right'], parents=[head])
    tail = batch.create_job(DOCKER_ROOT_IMAGE,
                            command=['echo', 'tail'],
                            parents=[left, right],
                            always_run=True)
    batch = batch.submit()
    right.wait()
    batch.cancel()
    batch.wait()
    status = legacy_batch_status(batch)
    assert batch_status_job_counter(status, 'Success') == 3, str(status)
    assert batch_status_job_counter(status, 'Cancelled') == 1, str(status)

    for node in [head, right, tail]:
        status = node.status()
        assert status['state'] == 'Success', str(status, node.log())
        assert node._get_exit_code(status, 'main') == 0, str(status, node.log())


def test_always_run_error(client):
    batch = client.create_batch()
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['/bin/sh', '-c', 'exit 1'])
    tail = batch.create_job(DOCKER_ROOT_IMAGE,
                            command=['echo', 'tail'],
                            parents=[head],
                            always_run=True)
    batch = batch.submit()
    batch.wait()
    status = legacy_batch_status(batch)
    assert batch_status_job_counter(status, 'Failed') == 1
    assert batch_status_job_counter(status, 'Success') == 1

    for job, ec, state in [(head, 1, 'Failed'), (tail, 0, 'Success')]:
        status = job.status()
        assert status['state'] == state, status
        assert job._get_exit_code(status, 'main') == ec, str(status)
