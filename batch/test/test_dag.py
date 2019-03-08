import os
import time
import pkg_resources
import pytest
import re
import requests
from flask import Response

from batch.client import BatchClient

from .serverthread import ServerThread

@pytest.fixture
def client():
    return BatchClient(url=os.environ.get('BATCH_URL'))


def test_simple(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    assert head.parent_ids == []
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[head.id])
    assert tail.parent_ids == [head.id]
    status = batch.wait()
    assert status['jobs']['Complete'] == 2
    head_status = head.status()
    assert head_status['state'] == 'Complete'
    assert head_status['exit_code'] == 0
    tail_status = tail.status()
    assert tail_status['state'] == 'Complete'
    assert tail_status['exit_code'] == 0


def test_missing_parent_is_400(client):
    try:
        batch = client.create_batch()
        batch.create_job('alpine:3.8', command=['echo', 'head'], parent_ids=[100000])
    except requests.exceptions.HTTPError as err:
        assert err.response.status_code == 400
        assert re.search('.*invalid parent_id: no job with id.*', err.response.text)
        return
    assert False


def test_already_deleted_parent_is_400(client):
    try:
        batch = client.create_batch()
        head = batch.create_job('alpine:3.8', command=['echo', 'head'])
        head_id = head.id
        head.delete()
        batch.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[head_id])
    except requests.exceptions.HTTPError as err:
        assert err.response.status_code == 400
        assert re.search('.*invalid parent_id: no job with id.*', err.response.text)
        return
    assert False


def test_dag(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    assert head.parent_ids == []
    left = batch.create_job('alpine:3.8', command=['echo', 'left'], parent_ids=[head.id])
    assert left.parent_ids == [head.id]
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parent_ids=[head.id])
    assert right.parent_ids == [head.id]
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[left.id, right.id])
    assert tail.parent_ids == [left.id, right.id]
    status = batch.wait()
    assert status['jobs']['Complete'] == 4
    for node in [head, left, right, tail]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0


def test_cancel_tail(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    left = batch.create_job('alpine:3.8', command=['echo', 'left'], parent_ids=[head.id])
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parent_ids=[head.id])
    tail = batch.create_job(
        'alpine:3.8',
        command=['/bin/sh', '-c', 'while true; do sleep 86000; done'],
        parent_ids=[left.id, right.id])
    tail.cancel()
    status = batch.wait()
    assert status['jobs']['Complete'] == 3
    for node in [head, left, right]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0
    assert tail.status()['state'] == 'Cancelled'


def test_cancel_left_before_tail(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    left = batch.create_job(
        'alpine:3.8',
        command=['/bin/sh', '-c', 'while true; do sleep 86000; done'],
        parent_ids=[head.id])
    left.cancel()
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parent_ids=[head.id])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[left.id, right.id])
    status = batch.wait()
    assert status['jobs']['Complete'] == 2
    for node in [head, right]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0
    for node in [left, tail]:
        assert node.status()['state'] == 'Cancelled'


def test_cancel_left_after_tail(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    left = batch.create_job(
        'alpine:3.8',
        command=['/bin/sh', '-c', 'while true; do sleep 86000; done'],
        parent_ids=[head.id])
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parent_ids=[head.id])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[left.id, right.id])
    left.cancel()
    status = batch.wait()
    assert status['jobs']['Complete'] == 2
    for node in [head, right]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0
    for node in [left, tail]:
        assert node.status()['state'] == 'Cancelled'


def test_delete(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    left = batch.create_job('alpine:3.8', command=['echo', 'left'], parent_ids=[head.id])
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parent_ids=[head.id])
    tail = batch.create_job(
        'alpine:3.8',
        command=['/bin/sh', '-c', 'while true; do sleep 86000; done'],
        parent_ids=[left.id, right.id])
    tail.delete()
    status = batch.wait()
    assert status['jobs']['Complete'] >= 3
    for node in [head, left, right]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0


def test_one_of_two_parent_ids_cancelled(client):
    batch = client.create_batch()
    left = batch.create_job(
        'alpine:3.8',
        command=['/bin/sh', '-c', 'while true; do sleep 86000; done'])
    right = batch.create_job('alpine:3.8', command=['echo', 'right'])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[left.id, right.id])
    left.cancel()
    status = batch.wait()
    assert status['jobs']['Complete'] == 1
    assert status['jobs']['Cancelled'] == 2
    right_status = right.status()
    assert right_status['state'] == 'Complete'
    assert right_status['exit_code'] == 0
    for node in [left, tail]:
        assert node.status()['state'] == 'Cancelled'


def test_parent_already_done(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    head.wait()
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[head.id])
    status = batch.wait()
    assert status['jobs']['Complete'] == 2
    for node in [head, tail]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0


def test_one_of_two_parent_ids_already_done(client):
    batch = client.create_batch()
    left = batch.create_job('alpine:3.8', command=['echo', 'left'])
    left.wait()
    right = batch.create_job('alpine:3.8', command=['echo', 'right'])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[left.id, right.id])
    status = batch.wait()
    assert status['jobs']['Complete'] == 3
    for node in [left, right, tail]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0


def test_one_of_two_parent_ids_already_cancelled(client):
    batch = client.create_batch()
    left = batch.create_job(
        'alpine:3.8',
        command=['/bin/sh', '-c', 'while true; do sleep 86000; done'])
    left.cancel()
    right = batch.create_job('alpine:3.8', command=['echo', 'right'])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[left.id, right.id])
    status = batch.wait()
    assert status['jobs']['Complete'] == 1
    assert status['jobs']['Cancelled'] == 2
    right_status = right.status()
    assert right_status['state'] == 'Complete'
    assert right_status['exit_code'] == 0
    for node in [left, tail]:
        assert node.status()['state'] == 'Cancelled'


def test_parent_deleted(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    left = batch.create_job(
        'alpine:3.8',
        command=['/bin/sh', '-c', 'while true; do sleep 86000; done'],
        parent_ids=[head.id])
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parent_ids=[head.id])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[left.id, right.id])
    left.delete()
    status = batch.wait()
    assert status['jobs']['Complete'] == 2
    for node in [head, right]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0
    assert tail.status()['state'] == 'Cancelled'


def test_callback(client):
    from flask import Flask, request
    app = Flask('test-client')
    output = []

    @app.route('/test', methods=['POST'])
    def test():
        output.append(request.get_json())
        return Response(status=200)

    try:
        server = ServerThread(app)
        server.start()
        batch = client.create_batch(callback=server.url_for('/test'))
        head = batch.create_job('alpine:3.8', command=['echo', 'head'])
        left = batch.create_job('alpine:3.8', command=['echo', 'left'], parent_ids=[head.id])
        right = batch.create_job('alpine:3.8', command=['echo', 'right'], parent_ids=[head.id])
        tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[left.id, right.id])
        batch.wait()
        i = 0
        while len(output) != 4:
            time.sleep(0.100 * (3/2) ** i)
            i = i + 1
            if i > 14:
                break
        assert len(output) == 4
        assert all([job_result['state'] == 'Complete' and job_result['exit_code'] == 0
                    for job_result in output])
        assert output[0]['id'] == head.id
        middle_ids = (output[1]['id'], output[2]['id'])
        assert middle_ids in ((left.id, right.id), (right.id, left.id))
        assert output[3]['id'] == tail.id
    finally:
        if server:
            server.shutdown()
            server.join()


def test_from_file(client):
        fname = pkg_resources.resource_filename(
            __name__,
            'diamond_dag.yml')
        with open(fname) as f:
            batch = client.create_batch_from_file(f)

        status = batch.wait()
        assert status['jobs']['Complete'] == 4


def test_no_parents_allowed_in_other_batches(client):
    b1 = client.create_batch()
    b2 = client.create_batch()
    head = b1.create_job('alpine:3.8', command=['echo', 'head'])
    try:
        b2.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[head.id])
    except requests.exceptions.HTTPError as err:
        assert err.response.status_code == 400
        assert re.search('.*invalid parent batch: .*', err.response.text)
        return
    assert False


def test_no_parents_allowed_without_batches(client):
    head = client.create_job('alpine:3.8', command=['echo', 'head'])
    try:
        client.create_job('alpine:3.8', command=['echo', 'tail'], parent_ids=[head.id])
    except requests.exceptions.HTTPError as err:
        assert err.response.status_code == 400
        assert re.search('.*invalid parent batch: .*', err.response.text)
        return
    assert False


def test_output_files_no_service_account_is_error(client):
    batch = client.create_batch()
    try:
        batch.create_job('alpine:3.8',
                         command=['/bin/sh', '-c', 'echo head > /out'],
                         output_files=[('/out', 'gs://hail-ci-0-1-batch-volume-test-bucket')])
    except requests.exceptions.HTTPError as err:
        assert err.response.status_code == 400
        assert re.search('.*invalid request: service account may be specified '
                         'if and only if input_files or output_files is set.*',
                         err.response.text)
        return
    assert False


def test_input_files_no_service_account_is_error(client):
    batch = client.create_batch()
    try:
        batch.create_job('alpine:3.8',
                         command=['/bin/sh', '-c', 'echo head > /out'],
                         output_input=[('gs://hail-ci-0-1-batch-volume-test-bucket', '/in')])
    except requests.exceptions.HTTPError as err:
        assert err.response.status_code == 400
        assert re.search('.*invalid request: service account may be specified '
                         'if and only if input_files or output_files is set.*',
                         err.response.text)
        return
    assert False


def test_service_account_no_files_is_error(client):
    batch = client.create_batch()
    try:
        batch.create_job('alpine:3.8',
                         command=['/bin/sh', '-c', 'echo head > /out'],
                         copy_service_account_name='batch-volume-tester')
    except requests.exceptions.HTTPError as err:
        assert err.response.status_code == 400
        assert re.search('.*invalid request: service account may be specified '
                         'if and only if input_files or output_files is set.*',
                         err.response.text)
        return
    assert False


def test_input_dependency(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8',
                            command=['/bin/sh', '-c', 'echo head > /out'],
                            output_files=[('/out', 'gs://hail-ci-0-1-batch-volume-test-bucket')],
                            copy_service_account_name='batch-volume-tester')
    tail = batch.create_job('alpine:3.8',
                            command=['/bin/sh', '-c', 'cat /in'],
                            input_files=[('gs://hail-ci-0-1-batch-volume-test-bucket', '/in')],
                            copy_service_account_name='batch-volume-tester',
                            parent_ids=[head.id])
    tail.wait()
    assert tail.log() == 'head\n'
