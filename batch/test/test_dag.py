import asyncio
import os
import re
import secrets

import pytest
from aiohttp import web

from hailtop.batch_client import aioclient
from hailtop.batch_client.client import BatchClient, Job

from .utils import DOCKER_ROOT_IMAGE, batch_status_job_counter, create_batch, legacy_batch_status


@pytest.fixture
def client():
    client = BatchClient('test')
    yield client
    client.close()


def test_simple(client):
    batch = create_batch(client)
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[head])
    batch.submit()
    batch.wait()
    status = legacy_batch_status(batch)
    assert batch_status_job_counter(status, 'Success') == 2, str((status, batch.debug_info()))
    assert all(j['exit_code'] == 0 for j in status['jobs']), str(batch.debug_info())


def test_missing_parent_is_400(client):
    try:
        batch = create_batch(client)
        fake_job = aioclient.Job.unsubmitted_job(batch._async_batch, 10000)
        fake_job = Job(fake_job)
        batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'], parents=[fake_job])
        batch.submit()
    except ValueError as err:
        assert re.search('parents with invalid job ids', str(err))
        return
    assert False, str(batch.debug_info())


def test_dag(client):
    batch = create_batch(client)
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    left = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'left'], parents=[head])
    right = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'right'], parents=[head])
    tail = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[left, right])
    batch.submit()
    batch.wait()
    status = legacy_batch_status(batch)

    node_status_logs = []
    for node in [head, left, right, tail]:
        node_status_logs.append((node.status(), node.log()))

    assert batch_status_job_counter(status, 'Success') == 4, str((status, batch.debug_info()))

    for node in [head, left, right, tail]:
        status = node._status
        assert status['state'] == 'Success', str((status, batch.debug_info()))
        assert node._get_exit_code(status, 'main') == 0, str((status, batch.debug_info()))


def test_cancel_tail(client):
    batch = create_batch(client)
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    left = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'left'], parents=[head])
    right = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'right'], parents=[head])
    tail = batch.create_job(
        DOCKER_ROOT_IMAGE, command=['/bin/sh', '-c', 'while true; do sleep 86000; done'], parents=[left, right]
    )
    try:
        batch.submit()
        left.wait()
        right.wait()
    finally:
        batch.cancel()
    batch.wait()
    status = legacy_batch_status(batch)
    assert batch_status_job_counter(status, 'Success') == 3, str((status, batch.debug_info()))
    for node in [head, left, right]:
        status = node.status()
        assert status['state'] == 'Success', str((status, batch.debug_info()))
        assert node._get_exit_code(status, 'main') == 0, str((status, batch.debug_info()))
    tail_status = tail.status()
    assert tail_status['state'] == 'Cancelled', str((tail_status, batch.debug_info()))


def test_cancel_left_after_tail(client):
    batch = create_batch(client)
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    left = batch.create_job(
        DOCKER_ROOT_IMAGE, command=['/bin/sh', '-c', 'while true; do sleep 86000; done'], parents=[head]
    )
    right = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'right'], parents=[head])
    tail = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[left, right])
    try:
        batch.submit()
        head.wait()
        right.wait()
    finally:
        batch.cancel()
    batch.wait()
    status = legacy_batch_status(batch)
    assert batch_status_job_counter(status, 'Success') == 2, str((status, batch.debug_info()))
    for node in [head, right]:
        status = node.status()
        assert status['state'] == 'Success', str((status, batch.debug_info()))
        assert node._get_exit_code(status, 'main') == 0, str((status, batch.debug_info()))
    for node in [left, tail]:
        node_status = node.status()
        assert node_status['state'] == 'Cancelled', str((node_status, batch.debug_info()))


async def test_callback(client):
    import nest_asyncio  # pylint: disable=import-outside-toplevel

    nest_asyncio.apply()

    app = web.Application()
    callback_bodies = []
    callback_event = asyncio.Event()

    def url_for(uri):
        host = os.environ['HAIL_BATCH_WORKER_IP']
        port = os.environ['HAIL_BATCH_WORKER_PORT']
        return f'http://{host}:{port}{uri}'

    async def callback(request):
        body = await request.json()
        callback_bodies.append(body)
        callback_event.set()
        return web.Response()

    app.add_routes([web.post('/test', callback)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 5000)
    await site.start()

    try:
        token = secrets.token_urlsafe(32)
        b = create_batch(
            client, callback=url_for('/test'), attributes={'foo': 'bar', 'name': 'test_callback'}, token=token
        )
        head = b.create_job('alpine:3.8', command=['echo', 'head'])
        b.create_job('alpine:3.8', command=['echo', 'tail'], parents=[head])
        b.submit()
        await asyncio.wait_for(callback_event.wait(), 5 * 60)
        callback_body = callback_bodies[0]

        # verify required fields present
        callback_body.pop('cost')
        callback_body.pop('msec_mcpu')
        callback_body.pop('time_created')
        callback_body.pop('time_closed')
        callback_body.pop('time_completed')
        callback_body.pop('duration')
        assert callback_body == {
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
            'attributes': {'foo': 'bar', 'name': 'test_callback'},
        }, callback_body
    finally:
        await runner.cleanup()


def test_no_parents_allowed_in_other_batches(client):
    b1 = create_batch(client)
    b2 = create_batch(client)
    head = b1.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    try:
        b2.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[head])
    except ValueError as err:
        assert re.search('parents from another batch', str(err))
        return
    assert False


def test_input_dependency(client, remote_tmpdir):
    batch = create_batch(client)
    head = batch.create_job(
        DOCKER_ROOT_IMAGE,
        command=['/bin/sh', '-c', 'echo head1 > /io/data1; echo head2 > /io/data2'],
        output_files=[('/io/data1', f'{remote_tmpdir}data1'), ('/io/data2', f'{remote_tmpdir}data2')],
    )
    tail = batch.create_job(
        DOCKER_ROOT_IMAGE,
        command=['/bin/sh', '-c', 'cat /io/data1; cat /io/data2'],
        input_files=[(f'{remote_tmpdir}data1', '/io/data1'), (f'{remote_tmpdir}data2', '/io/data2')],
        parents=[head],
    )
    batch.submit()
    tail.wait()
    head_status = head.status()
    assert head._get_exit_code(head_status, 'main') == 0, str((head_status, batch.debug_info()))
    tail_log = tail.log()
    assert tail_log['main'] == 'head1\nhead2\n', str((tail_log, batch.debug_info()))


def test_input_dependency_wildcard(client, remote_tmpdir):
    batch = create_batch(client)
    head = batch.create_job(
        DOCKER_ROOT_IMAGE,
        command=['/bin/sh', '-c', 'echo head1 > /io/data1 ; echo head2 > /io/data2'],
        output_files=[('/io/data1', f'{remote_tmpdir}data1'), ('/io/data2', f'{remote_tmpdir}data2')],
    )
    tail = batch.create_job(
        DOCKER_ROOT_IMAGE,
        command=['/bin/sh', '-c', 'cat /io/data1 ; cat /io/data2'],
        input_files=[(f'{remote_tmpdir}data1', '/io/data1'), (f'{remote_tmpdir}data2', '/io/data2')],
        parents=[head],
    )
    batch.submit()
    tail.wait()
    head_status = head.status()
    assert head._get_exit_code(head_status, 'input') != 0, str((head_status, batch.debug_info()))
    tail_log = tail.log()
    assert tail_log['main'] == 'head1\nhead2\n', str((tail_log, batch.debug_info()))


def test_input_dependency_directory(client, remote_tmpdir):
    batch = create_batch(client)
    head = batch.create_job(
        DOCKER_ROOT_IMAGE,
        command=['/bin/sh', '-c', 'mkdir -p /io/test/; echo head1 > /io/test/data1 ; echo head2 > /io/test/data2'],
        output_files=[('/io/test', f'{remote_tmpdir}test')],
    )
    tail = batch.create_job(
        DOCKER_ROOT_IMAGE,
        command=['/bin/sh', '-c', 'cat /io/test/data1; cat /io/test/data2'],
        input_files=[(f'{remote_tmpdir}test', '/io/test')],
        parents=[head],
    )
    batch.submit()
    tail.wait()
    head_status = head.status()
    assert head._get_exit_code(head_status, 'main') == 0, str((head_status, batch.debug_info()))
    tail_log = tail.log()
    assert tail_log['main'] == 'head1\nhead2\n', str((tail_log, batch.debug_info()))


def test_always_run_cancel(client):
    batch = create_batch(client)
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'head'])
    left = batch.create_job(
        DOCKER_ROOT_IMAGE, command=['/bin/sh', '-c', 'while true; do sleep 86000; done'], parents=[head]
    )
    right = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'right'], parents=[head])
    tail = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[left, right], always_run=True)
    try:
        batch.submit()
        right.wait()
    finally:
        batch.cancel()
    batch.wait()
    status = legacy_batch_status(batch)
    assert batch_status_job_counter(status, 'Success') == 3, str((status, batch.debug_info()))
    assert batch_status_job_counter(status, 'Cancelled') == 1, str((status, batch.debug_info()))

    for node in [head, right, tail]:
        status = node.status()
        assert status['state'] == 'Success', str((status, batch.debug_info()))
        assert node._get_exit_code(status, 'main') == 0, str((status, batch.debug_info()))


def test_always_run_error(client):
    batch = create_batch(client)
    head = batch.create_job(DOCKER_ROOT_IMAGE, command=['/bin/sh', '-c', 'exit 1'])
    tail = batch.create_job(DOCKER_ROOT_IMAGE, command=['echo', 'tail'], parents=[head], always_run=True)
    batch.submit()
    batch.wait()
    status = legacy_batch_status(batch)
    assert batch_status_job_counter(status, 'Failed') == 1, str((status, batch.debug_info()))
    assert batch_status_job_counter(status, 'Success') == 1, str((status, batch.debug_info()))

    for job, ec, state in [(head, 1, 'Failed'), (tail, 0, 'Success')]:
        status = job.status()
        assert status['state'] == state, str((status, batch.debug_info()))
        assert job._get_exit_code(status, 'main') == ec, str((status, batch.debug_info()))
