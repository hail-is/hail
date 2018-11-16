import os
import pytest
import re
import requests

import batch
from batch.data import DagNodeSpec, JobSpec


@pytest.fixture
def client():
    return batch.client.BatchClient(url=os.environ.get('BATCH_URL'))


def test_duplicate_name_is_400(client):
    try:
        client.create_dag([
            DagNodeSpec(
                'head',
                [],
                JobSpec.from_parameters('alpine:3.8', ['echo', 'head'])),
            DagNodeSpec(
                'head',
                ['head'],
                JobSpec.from_parameters('alpine:3.8', ['echo', 'left']))])
    except requests.exceptions.HTTPError as err:
        assert err.response.status_code == 400
        assert re.match('.*duplicate name: head.*', err.response.text)
        return
    assert False


def test_missing_dependency_is_400(client):
    try:
        client.create_dag([
            DagNodeSpec(
                'head',
                [],
                JobSpec.from_parameters('alpine:3.8', ['echo', 'head'])),
            DagNodeSpec(
                'tail',
                ['missing_dependency'],
                JobSpec.from_parameters('alpine:3.8', ['echo', 'left']))])
    except requests.exceptions.HTTPError as err:
        assert err.response.status_code == 400
        assert re.match('.*parent not found: missing_dependency.*', err.response.text)
        return
    assert False


def test_dag(client):
    head = DagNodeSpec(
        'head',
        [],
        JobSpec.from_parameters('alpine:3.8', ['echo', 'head']))
    left = DagNodeSpec(
        'left',
        ['head'],
        JobSpec.from_parameters('alpine:3.8', ['echo', 'left']))
    right = DagNodeSpec(
        'right',
        ['head'],
        JobSpec.from_parameters('alpine:3.8', ['echo', 'right']))
    tail = DagNodeSpec(
        'tail',
        ['left', 'right'],
        JobSpec.from_parameters('alpine:3.8', ['echo', 'tail']))
    id = client.create_dag([head, left, right, tail])
    client.get_dag(id)
