import os
import pytest
import re
import requests

from batch.client import BatchClient


@pytest.fixture
def client():
    return BatchClient(url=os.environ.get('BATCH_URL'))


def test_simple(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[head.id])
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
        batch.create_job('alpine:3.8', command=['echo', 'head'], parents=[100000])
    except requests.exceptions.HTTPError as err:
        assert err.response.status_code == 400
        assert re.search('.*invalid parent: no job with id.*', err.response.text)
        return
    assert False


def test_already_deleted_parent_is_400(client):
    try:
        batch = client.create_batch()
        head = batch.create_job('alpine:3.8', command=['echo', 'head'])
        head_id = head.id
        head.delete()
        tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[head_id])
    except requests.exceptions.HTTPError as err:
        assert err.response.status_code == 400
        assert re.search('.*invalid parent: no job with id.*', err.response.text)
        return
    assert False


def test_dag(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    left = batch.create_job('alpine:3.8', command=['echo', 'left'], parents=[head.id])
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parents=[head.id])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[left.id, right.id])
    status = batch.wait()
    assert status['jobs']['Complete'] == 4
    for node in [head, left, right, tail]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0


def test_cancel_tail(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    left = batch.create_job('alpine:3.8', command=['echo', 'left'], parents=[head.id])
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parents=[head.id])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[left.id, right.id])
    tail.cancel()
    status = batch.wait()
    assert status['jobs']['Complete'] >= 3
    for node in [head, left, right]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0
    assert tail.status()['state'] in ('Cancelled', 'Complete')


def test_cancel_left_before_tail(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    left = batch.create_job('alpine:3.8', command=['echo', 'left'], parents=[head.id])
    left.cancel()
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parents=[head.id])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[left.id, right.id])
    status = batch.wait()
    assert status['jobs']['Complete'] >= 2
    for node in [head, right]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0
    assert left.status()['state'] in ('Cancelled', 'Complete')
    assert tail.status()['state'] == 'Cancelled'


def test_cancel_left_after_tail(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    left = batch.create_job('alpine:3.8', command=['sleep', '60'], parents=[head.id])
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parents=[head.id])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[left.id, right.id])
    left.cancel()
    status = batch.wait()
    assert status['jobs']['Complete'] >= 2
    for node in [head, right]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0
    assert left.status()['state'] in ('Cancelled', 'Complete')
    assert tail.status()['state'] == left.status()['state']


def test_delete(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    left = batch.create_job('alpine:3.8', command=['sleep', '60'], parents=[head.id])
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parents=[head.id])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[left.id, right.id])
    tail.delete()
    status = batch.wait()
    assert status['jobs']['Complete'] >= 3
    for node in [head, left, right]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0


def test_one_of_two_parents_cancelled(client):
    batch = client.create_batch()
    left = batch.create_job('alpine:3.8', command=['sleep', '60'])
    right = batch.create_job('alpine:3.8', command=['echo', 'right'])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[left.id, right.id])
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
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[head.id])
    status = batch.wait()
    assert status['jobs']['Complete'] == 2
    for node in [head, tail]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0


def test_one_of_two_parents_already_done(client):
    batch = client.create_batch()
    left = batch.create_job('alpine:3.8', command=['echo', 'left'])
    left.wait()
    right = batch.create_job('alpine:3.8', command=['echo', 'right'])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[left.id, right.id])
    status = batch.wait()
    assert status['jobs']['Complete'] == 3
    for node in [left, right, tail]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0


def test_one_of_two_parents_already_cancelled(client):
    batch = client.create_batch()
    left = batch.create_job('alpine:3.8', command=['echo', 'left'])
    left.cancel()
    right = batch.create_job('alpine:3.8', command=['echo', 'right'])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[left.id, right.id])
    status = batch.wait()
    assert status['jobs']['Complete'] == 1
    assert status['jobs']['Cancelled'] == 2
    right_status = right.status()
    assert right_status['state'] == 'Complete'
    assert right_status['exit_code'] == 0
    assert left.status()['state'] in ('Cancelled', 'Complete')
    assert tail.status()['state'] == 'Cancelled'


def test_parent_deleted(client):
    batch = client.create_batch()
    head = batch.create_job('alpine:3.8', command=['echo', 'head'])
    left = batch.create_job('alpine:3.8', command=['sleep', '60'], parents=[head.id])
    right = batch.create_job('alpine:3.8', command=['echo', 'right'], parents=[head.id])
    tail = batch.create_job('alpine:3.8', command=['echo', 'tail'], parents=[left.id, right.id])
    left.delete()
    status = batch.wait()
    assert status['jobs']['Complete'] == 2
    for node in [head, right]:
        status = node.status()
        assert status['state'] == 'Complete'
        assert status['exit_code'] == 0
    assert tail.status()['state'] in ('Cancelled', 'Complete')
