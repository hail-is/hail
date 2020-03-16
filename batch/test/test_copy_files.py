import os
import re
import pytest
import uuid
from hailtop.batch_client.client import BatchClient, Job
from hailtop.auth import get_userinfo


global_token = uuid.uuid4().hex[:8]


@pytest.fixture
def client():
    client = BatchClient('test')
    yield client
    client.close()


def test_copy_input_to_filename(client):
    user = get_userinfo()
    batch = client.create_batch()
    token = uuid.uuid4().hex[:6]
    print(f'global_token={global_token} token={token}')
    head = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'echo head1 > /io/data1'],
                            output_files=[('/io/data1', f'gs://{user["bucket_name"]}/{global_token}/{token}/data2')])
    tail = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'cat /io/data2'],
                            input_files=[(f'gs://{user["bucket_name"]}/{global_token}/{token}/data2', '/io/')],
                            parents=[head])
    batch.submit()
    tail.wait()
    assert head._get_exit_code(head.status(), 'main') == 0, head._status
    assert tail.log()['main'] == 'head1\n', tail.status()


def test_copy_input_with_spaces_file_name(client):
    user = get_userinfo()
    batch = client.create_batch()
    token = uuid.uuid4().hex[:6]
    print(f'global_token={global_token} token={token}')
    head = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'echo head1 > "/io/data with spaces.txt"'],
                            output_files=[('/io/data with spaces.txt', f'gs://{user["bucket_name"]}/{global_token}/{token}/')])
    tail = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'cat "/io/data with spaces.txt"'],
                            input_files=[(f'gs://{user["bucket_name"]}/{global_token}/{token}/data with spaces.txt', '/io/')],
                            parents=[head])
    batch.submit()
    tail.wait()
    assert head._get_exit_code(head.status(), 'main') == 0, head._status
    assert tail.log()['main'] == 'head1\n', tail.status()


def test_copy_input_to_directory(client):
    user = get_userinfo()
    batch = client.create_batch()
    token = uuid.uuid4().hex[:6]
    print(f'global_token={global_token} token={token}')
    head = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'echo head1 > /io/data1'],
                            output_files=[('/io/data1', f'gs://{user["bucket_name"]}/{global_token}/{token}/')])
    tail = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'cat /io/data1'],
                            input_files=[(f'gs://{user["bucket_name"]}/{global_token}/{token}/data1', '/io/')],
                            parents=[head])
    batch.submit()
    tail.wait()
    assert head._get_exit_code(head.status(), 'main') == 0, head._status
    assert tail.log()['main'] == 'head1\n', tail.status()


def test_copy_input_to_directory_with_wildcard(client):
    user = get_userinfo()
    batch = client.create_batch()
    token = uuid.uuid4().hex[:6]
    print(f'global_token={global_token} token={token}')
    head = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'echo head1 > /io/data1 ; echo head2 > /io/data2'],
                            output_files=[('/io/data*', f'gs://{user["bucket_name"]}/{global_token}/{token}/')])
    tail = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'cat /io/data1 ; cat /io/data2'],
                            input_files=[(f'gs://{user["bucket_name"]}/{global_token}/{token}/data*', '/io/')],
                            parents=[head])
    batch.submit()
    tail.wait()
    assert head._get_exit_code(head.status(), 'main') == 0, head._status
    assert tail.log()['main'] == 'head1\nhead2\n', tail.status()


def test_copy_input_directory_where_subdir_not_exist(client):
    user = get_userinfo()
    batch = client.create_batch()
    token = uuid.uuid4().hex[:6]
    print(f'global_token={global_token} token={token}')
    head = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'mkdir -p /io/test/; echo head1 > /io/test/data1 ; echo head2 > /io/test/data2'],
                            output_files=[('/io/test/', f'gs://{user["bucket_name"]}/{global_token}/{token}/')])
    tail = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'cat /io/test/data1 ; cat /io/test/data2'],
                            input_files=[(f'gs://{user["bucket_name"]}/{global_token}/{token}/test/', f'/io/')],
                            parents=[head])
    batch.submit()
    tail.wait()
    assert head._get_exit_code(head.status(), 'main') == 0, head._status
    assert tail.log()['main'] == 'head1\nhead2\n', tail.status()


def test_copy_input_directory_where_subdir_exists(client):
    user = get_userinfo()
    batch = client.create_batch()
    token = uuid.uuid4().hex[:6]
    print(f'global_token={global_token} token={token}')
    head1 = batch.create_job('ubuntu:18.04',
                             command=['/bin/sh', '-c', 'mkdir -p /io/test/; echo head1 > /io/test/data1 ; echo head2 > /io/test/data2'],
                             output_files=[('/io/test/', f'gs://{user["bucket_name"]}/{global_token}/{token}/')])
    head2 = batch.create_job('ubuntu:18.04',
                             command=['/bin/sh', '-c', 'mkdir -p /io/test2/; echo head3 > /io/test2/data3 ; echo head4 > /io/test2/data4'],
                             output_files=[('/io/test2/', f'gs://{user["bucket_name"]}/{global_token}/{token}/')],
                             parents=[head1])
    tail = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'cat /io/test2/data3 ; cat /io/test2/data4'],
                            input_files=[(f'gs://{user["bucket_name"]}/{global_token}/{token}/test2', '/io/')],
                            parents=[head2])
    batch.submit()
    tail.wait()
    assert head1._get_exit_code(head1.status(), 'main') == 0, head1._status
    assert head2._get_exit_code(head1.status(), 'main') == 0, head2._status
    assert tail.log()['main'] == 'head3\nhead4\n', tail.status()


def test_input_dependency_directory_with_file_same_name(client):
    user = get_userinfo()
    batch = client.create_batch()
    token = uuid.uuid4().hex[:6]
    print(f'global_token={global_token} token={token}')
    j1 = batch.create_job('ubuntu:18.04',
                          command=['/bin/sh', '-c', 'mkdir -p /io/test/; echo head1 > /io/test/data1 ; echo head2 > /io/test/data2'],
                          output_files=[('/io/test/', f'gs://{user["bucket_name"]}/{global_token}/{token}/')])
    j2 = batch.create_job('ubuntu:18.04',
                          command=['/bin/sh', '-c', 'mkdir -p /io/test2/; touch /io/test2/test'],
                          output_files=[('/io/test2/test', f'gs://{user["bucket_name"]}/{global_token}/{token}/test')])
    j3 = batch.create_job('ubuntu:18.04',
                          command=['/bin/sh', '-c', 'cat /io/test/data1 ; cat /io/test/data2'],
                          input_files=[(f'gs://{user["bucket_name"]}/{global_token}/{token}/*', '/io/')],
                          parents=[j1, j2])
    batch.submit()
    j3.wait()
    input_log = j3.log()['input']
    assert 'IsADirectoryError' in input_log or 'FileExistsError' in input_log, input_log


def test_copy_input_within_gcs(client):
    user = get_userinfo()
    batch = client.create_batch()
    token = uuid.uuid4().hex[:6]
    print(f'global_token={global_token} token={token}')
    head = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'echo head1 > /io/data1'],
                            output_files=[('/io/data1', f'gs://{user["bucket_name"]}/{global_token}/{token}/')])
    tail = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'echo hello'],
                            input_files=[(f'gs://{user["bucket_name"]}/{global_token}/{token}/data1',
                                          f'gs://{user["bucket_name"]}/{global_token}/{token}/data2')],
                            parents=[head])
    batch.submit()
    tail.wait()
    assert head._get_exit_code(head.status(), 'main') == 0, head._status
    assert tail.log()['main'] == 'hello\n', tail.status()


def test_copy_directory_within_gcs(client):
    user = get_userinfo()
    batch = client.create_batch()
    token1 = uuid.uuid4().hex[:6]
    token2 = uuid.uuid4().hex[:6]
    print(f'global_token={global_token} token1={token1} token2={token2}')
    head1 = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'echo head1 > /io/data1 ; echo head2 > /io/data2'],
                            output_files=[('/io/data*', f'gs://{user["bucket_name"]}/{global_token}/{token1}/')])
    head2 = batch.create_job('ubuntu:18.04',
                             command=['/bin/sh', '-c', 'echo hello'],
                             input_files=[(f'gs://{user["bucket_name"]}/{global_token}/{token1}/',
                                           f'gs://{user["bucket_name"]}/{global_token}/{token2}/')],
                             parents=[head1])
    tail = batch.create_job('ubuntu:18.04',
                            command=['/bin/sh', '-c', 'cat /io/data1 ; cat /io/data2'],
                            input_files=[(f'gs://{user["bucket_name"]}/{global_token}/{token2}/{token1}/data*', '/io/')],
                            parents=[head2])
    batch.submit()
    tail.wait()
    assert head1._get_exit_code(head1.status(), 'main') == 0, head1._status
    assert head2._get_exit_code(head2.status(), 'main') == 0, head2._status
    assert tail.log()['main'] == 'head1\nhead2\n', tail.status()


def test_copy_locally(client):
    batch = client.create_batch()

    head = batch.create_job(os.environ['BATCH_WORKER_IMAGE'],
                            command=['/bin/sh', '-c', f'''
mkdir -p /a/b/c/d/
touch /a/b/foo1
touch /a/b/foo2
touch /a/b/c/d/e.txt
mkdir /test1/
mkdir /test2/
mkdir /test3/
mkdir -p /test4/a/

cat | python3 /batch/copy_files.py --key-file /gsa-key/key.json --project hail-vdc <<EOF
/a/ /test1/
/a/b/foo* /test2/
/a/b/c/d/e.txt /test3/e2.txt
/a/ /test4/
EOF

test -f /test1/a/b/c/d/e.txt
test -f /test1/a/b/foo1
test -f /test1/a/b/foo2
test -f /test2/foo1
test -f /test2/foo2
test -f /test3/e2.txt
test -f /test4/a/b/c/d/e.txt
'''])

    batch.submit()
    head.wait()
    assert head._get_exit_code(head.status(), 'main') == 0, (head.log()['main'], head.status())
