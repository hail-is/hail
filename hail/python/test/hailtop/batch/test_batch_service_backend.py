import asyncio
import inspect
import os
import secrets
from configparser import ConfigParser
from shlex import quote as shq
from typing import AsyncIterator, Tuple

import orjson
import pytest

import hailtop.batch_client.client as bc
from hailtop import pip_version
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.batch import Batch, ResourceGroup, ServiceBackend
from hailtop.batch.exceptions import BatchException
from hailtop.batch.globals import arg_max
from hailtop.config import configuration_of, get_remote_tmpdir, get_user_config, user_config
from hailtop.config.variables import ConfigVariable
from hailtop.httpx import ClientResponseError
from hailtop.test_utils import skip_in_azure
from hailtop.utils import grouped, secret_alnum_string

DOCKER_ROOT_IMAGE = os.environ.get('DOCKER_ROOT_IMAGE', 'ubuntu:22.04')
PYTHON_DILL_IMAGE = 'hailgenetics/python-dill:3.9-slim'
HAIL_GENETICS_HAIL_IMAGE = os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{pip_version()}')
REQUESTER_PAYS_PROJECT = os.environ.get('GCS_REQUESTER_PAYS_PROJECT')


@pytest.fixture(scope="session")
async def backend() -> AsyncIterator[ServiceBackend]:
    sb = ServiceBackend()
    try:
        yield sb
    finally:
        await sb.async_close()


@pytest.fixture(scope="session")
async def fs() -> AsyncIterator[RouterAsyncFS]:
    fs = RouterAsyncFS()
    try:
        yield fs
    finally:
        await fs.close()


@pytest.fixture(scope="session")
def tmpdir() -> str:
    return os.path.join(
        get_remote_tmpdir('test_batch_service_backend.py::tmpdir'),
        secret_alnum_string(5),  # create a unique URL for each split of the tests
    )


@pytest.fixture
def output_tmpdir(tmpdir: str) -> str:
    return os.path.join(tmpdir, 'output', secret_alnum_string(5))


@pytest.fixture
def output_bucket_path(fs: RouterAsyncFS, output_tmpdir: str) -> Tuple[str, str, str]:
    url = fs.parse_url(output_tmpdir)
    bucket = '/'.join(url.bucket_parts)
    path = url.path
    path = '/' + os.path.join(bucket, path)
    return bucket, path, output_tmpdir


@pytest.fixture(scope="session")
async def upload_test_files(
    fs: RouterAsyncFS, tmpdir: str
) -> Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]]:
    test_files = (
        (os.path.join(tmpdir, 'inputs/hello.txt'), b'hello world'),
        (os.path.join(tmpdir, 'inputs/hello spaces.txt'), b'hello'),
        (os.path.join(tmpdir, 'inputs/hello (foo) spaces.txt'), b'hello'),
    )
    await asyncio.gather(*(fs.write(url, data) for url, data in test_files))
    return test_files


def batch(backend, **kwargs):
    name_of_test_method = inspect.stack()[1][3]
    return Batch(
        name=name_of_test_method,
        backend=backend,
        default_image=DOCKER_ROOT_IMAGE,
        attributes={'foo': 'a', 'bar': 'b'},
        **kwargs,
    )


def test_single_task_no_io(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job()
    j.command('echo hello')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_input(
    backend: ServiceBackend, upload_test_files: Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]]
):
    (url1, data1), _, _ = upload_test_files
    b = batch(backend)
    input = b.read_input(url1)
    j = b.new_job()
    j.command(f'cat {input}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_input_resource_group(
    backend: ServiceBackend, upload_test_files: Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]]
):
    (url1, data1), _, _ = upload_test_files
    b = batch(backend)
    input = b.read_input_group(foo=url1)
    j = b.new_job()
    j.storage('10Gi')
    j.command(f'cat {input.foo}')
    j.command(f'cat {input}.foo')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_output(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job(attributes={'a': 'bar', 'b': 'foo'})
    j.command(f'echo hello > {j.ofile}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_write_output(backend: ServiceBackend, output_tmpdir: str):
    b = batch(backend)
    j = b.new_job()
    j.command(f'echo hello > {j.ofile}')
    b.write_output(j.ofile, os.path.join(output_tmpdir, 'test_single_task_output.txt'))
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_resource_group(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job()
    j.declare_resource_group(output={'foo': '{root}.foo'})
    assert isinstance(j.output, ResourceGroup)
    j.command(f'echo "hello" > {j.output.foo}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_write_resource_group(backend: ServiceBackend, output_tmpdir: str):
    b = batch(backend)
    j = b.new_job()
    j.declare_resource_group(output={'foo': '{root}.foo'})
    assert isinstance(j.output, ResourceGroup)
    j.command(f'echo "hello" > {j.output.foo}')
    b.write_output(j.output, os.path.join(output_tmpdir, 'test_single_task_write_resource_group'))
    b.write_output(j.output.foo, os.path.join(output_tmpdir, 'test_single_task_write_resource_group_file.txt'))
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_multiple_dependent_tasks(backend: ServiceBackend, output_tmpdir: str):
    output_file = os.path.join(output_tmpdir, 'test_multiple_dependent_tasks.txt')
    b = batch(backend)
    j = b.new_job()
    j.command(f'echo "0" >> {j.ofile}')

    for i in range(1, 3):
        j2 = b.new_job()
        j2.command(f'echo "{i}" > {j2.tmp1}')
        j2.command(f'cat {j.ofile} {j2.tmp1} > {j2.ofile}')
        j = j2

    b.write_output(j.ofile, output_file)
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_specify_cpu(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job()
    j.cpu('0.5')
    j.command(f'echo "hello" > {j.ofile}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_specify_memory(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job()
    j.memory('100M')
    j.command(f'echo "hello" > {j.ofile}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_scatter_gather(backend: ServiceBackend):
    b = batch(backend)

    for i in range(3):
        j = b.new_job(name=f'foo{i}')
        j.command(f'echo "{i}" > {j.ofile}')

    merger = b.new_job()
    merger.command(
        'cat {files} > {ofile}'.format(
            files=' '.join(
                [j.ofile for j in sorted(b.select_jobs('foo'), key=lambda x: x.name, reverse=True)]  # type: ignore
            ),
            ofile=merger.ofile,
        )
    )

    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_file_name_space(
    backend: ServiceBackend,
    upload_test_files: Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]],
    output_tmpdir: str,
):
    _, _, (url3, data3) = upload_test_files
    b = batch(backend)
    input = b.read_input(url3)
    j = b.new_job()
    j.command(f'cat {input} > {j.ofile}')
    b.write_output(j.ofile, os.path.join(output_tmpdir, 'hello (foo) spaces.txt'))
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_dry_run(backend: ServiceBackend, output_tmpdir: str):
    b = batch(backend)
    j = b.new_job()
    j.command(f'echo hello > {j.ofile}')
    b.write_output(j.ofile, os.path.join(output_tmpdir, 'test_single_job_output.txt'))
    b.run(dry_run=True)


def test_verbose(
    backend: ServiceBackend,
    upload_test_files: Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]],
    output_tmpdir: str,
):
    (url1, data1), _, _ = upload_test_files
    b = batch(backend)
    input = b.read_input(url1)
    j = b.new_job()
    j.command(f'cat {input}')
    b.write_output(input, os.path.join(output_tmpdir, 'hello.txt'))
    res = b.run(verbose=True)
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_cloudfuse_fails_with_read_write_mount_option(fs: RouterAsyncFS, backend: ServiceBackend, output_bucket_path):
    bucket, path, output_tmpdir = output_bucket_path

    b = batch(backend)
    j = b.new_job()
    j.command(f'mkdir -p {path}; echo head > {path}/cloudfuse_test_1')
    j.cloudfuse(bucket, f'/{bucket}', read_only=False)

    try:
        b.run()
    except ClientResponseError as e:
        assert 'Only read-only cloudfuse requests are supported' in e.body, e.body
    else:
        assert False


def test_cloudfuse_fails_with_io_mount_point(fs: RouterAsyncFS, backend: ServiceBackend, output_bucket_path):
    bucket, path, output_tmpdir = output_bucket_path

    b = batch(backend)
    j = b.new_job()
    j.command(f'mkdir -p {path}; echo head > {path}/cloudfuse_test_1')
    j.cloudfuse(bucket, '/io', read_only=True)

    try:
        b.run()
    except ClientResponseError as e:
        assert 'Cloudfuse requests with mount_path=/io are not supported' in e.body, e.body
    else:
        assert False


def test_cloudfuse_read_only(backend: ServiceBackend, output_bucket_path):
    bucket, path, output_tmpdir = output_bucket_path

    b = batch(backend)
    j = b.new_job()
    j.command(f'mkdir -p {path}; echo head > {path}/cloudfuse_test_1')
    j.cloudfuse(bucket, f'/{bucket}', read_only=True)

    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))


def test_cloudfuse_implicit_dirs(fs: RouterAsyncFS, backend: ServiceBackend, upload_test_files):
    (url1, data1), _, _ = upload_test_files
    parsed_url1 = fs.parse_url(url1)
    object_name = parsed_url1.path
    bucket_name = '/'.join(parsed_url1.bucket_parts)

    b = batch(backend)
    j = b.new_job()
    j.command('cat ' + os.path.join('/cloudfuse', object_name))
    j.cloudfuse(bucket_name, '/cloudfuse', read_only=True)

    res = b.run()
    assert res
    res_status = res.status()
    assert res.get_job_log(1)['main'] == data1.decode()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_cloudfuse_empty_string_bucket_fails(backend: ServiceBackend, output_bucket_path):
    bucket, path, output_tmpdir = output_bucket_path

    b = batch(backend)
    j = b.new_job()
    with pytest.raises(BatchException):
        j.cloudfuse('', '/empty_bucket')
    with pytest.raises(BatchException):
        j.cloudfuse(bucket, '')


async def test_cloudfuse_submount_in_io_doesnt_rm_bucket(
    fs: RouterAsyncFS, backend: ServiceBackend, output_bucket_path
):
    bucket, path, output_tmpdir = output_bucket_path

    should_still_exist_url = os.path.join(output_tmpdir, 'should-still-exist')
    await fs.write(should_still_exist_url, b'should-still-exist')

    b = batch(backend)
    j = b.new_job()
    j.cloudfuse(bucket, '/io/cloudfuse')
    j.command('ls /io/cloudfuse/')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))
    assert await fs.read(should_still_exist_url) == b'should-still-exist'


@skip_in_azure
def test_fuse_requester_pays(backend: ServiceBackend):
    assert REQUESTER_PAYS_PROJECT
    b = batch(backend, requester_pays_project=REQUESTER_PAYS_PROJECT)
    j = b.new_job()
    j.cloudfuse('hail-test-requester-pays-fds32', '/fuse-bucket')
    j.command('cat /fuse-bucket/hello')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


@skip_in_azure
def test_fuse_non_requester_pays_bucket_when_requester_pays_project_specified(
    backend: ServiceBackend, output_bucket_path
):
    bucket, path, output_tmpdir = output_bucket_path
    assert REQUESTER_PAYS_PROJECT

    b = batch(backend, requester_pays_project=REQUESTER_PAYS_PROJECT)
    j = b.new_job()
    j.command('ls /fuse-bucket')
    j.cloudfuse(bucket, '/fuse-bucket', read_only=True)

    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


@skip_in_azure
def test_requester_pays(backend: ServiceBackend):
    assert REQUESTER_PAYS_PROJECT
    b = batch(backend, requester_pays_project=REQUESTER_PAYS_PROJECT)
    input = b.read_input('gs://hail-test-requester-pays-fds32/hello')
    j = b.new_job()
    j.command(f'cat {input}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_benchmark_lookalike_workflow(backend: ServiceBackend, output_tmpdir):
    b = batch(backend)

    setup_jobs = []
    for i in range(10):
        j = b.new_job(f'setup_{i}').cpu(0.25)
        j.command(f'echo "foo" > {j.ofile}')
        setup_jobs.append(j)

    jobs = []
    for i in range(500):
        j = b.new_job(f'create_file_{i}').cpu(0.25)
        j.command(f'echo {setup_jobs[i % len(setup_jobs)].ofile} > {j.ofile}')
        j.command(f'echo "bar" >> {j.ofile}')
        jobs.append(j)

    combine = b.new_job('combine_output').cpu(0.25)
    for _ in grouped(arg_max(), jobs):
        combine.command(f'cat {" ".join(shq(j.ofile) for j in jobs)} >> {combine.ofile}')
    b.write_output(combine.ofile, os.path.join(output_tmpdir, 'pipeline_benchmark_test.txt'))
    # too slow
    # assert b.run().status()['state'] == 'success'


def test_envvar(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job()
    j.env('SOME_VARIABLE', '123abcdef')
    j.command('[ $SOME_VARIABLE = "123abcdef" ]')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_job_with_shell(backend: ServiceBackend):
    msg = 'hello world'
    b = batch(backend)
    j = b.new_job(shell='/bin/sh')
    j.command(f'echo "{msg}"')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_job_with_nonsense_shell(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job(shell='/bin/ajdsfoijasidojf')
    j.command('echo "hello"')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))


def test_single_job_with_intermediate_failure(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job()
    j.command('echoddd "hello"')
    j2 = b.new_job()
    j2.command('echo "world"')

    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))


def test_input_directory(
    backend: ServiceBackend, upload_test_files: Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]]
):
    (url1, data1), _, _ = upload_test_files
    b = batch(backend)
    containing_folder = '/'.join(url1.rstrip('/').split('/')[:-1])
    input1 = b.read_input(containing_folder)
    input2 = b.read_input(containing_folder + '/')
    j = b.new_job()
    j.command(f'ls {input1}/hello.txt')
    j.command(f'ls {input2}/hello.txt')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_python_job(backend: ServiceBackend):
    b = batch(backend, default_python_image=PYTHON_DILL_IMAGE)
    head = b.new_job()
    head.command(f'echo "5" > {head.r5}')
    head.command(f'echo "3" > {head.r3}')

    def read(path):
        with open(path, 'r') as f:
            i = f.read()
        return int(i)

    def multiply(x, y):
        return x * y

    def reformat(x, y):
        return {'x': x, 'y': y}

    middle = b.new_python_job()
    r3 = middle.call(read, head.r3)
    r5 = middle.call(read, head.r5)
    r_mult = middle.call(multiply, r3, r5)

    middle2 = b.new_python_job()
    r_mult = middle2.call(multiply, r_mult, 2)
    r_dict = middle2.call(reformat, r3, r5)

    tail = b.new_job()
    tail.command(f'cat {r3.as_str()} {r5.as_repr()} {r_mult.as_str()} {r_dict.as_json()}')

    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))
    assert res.get_job_log(4)['main'] == "3\n5\n30\n{\"x\": 3, \"y\": 5}\n", str(res.debug_info())


def test_python_job_w_resource_group_unpack_individually(backend: ServiceBackend):
    b = batch(backend, default_python_image=PYTHON_DILL_IMAGE)
    head = b.new_job()
    head.declare_resource_group(count={'r5': '{root}.r5', 'r3': '{root}.r3'})
    assert isinstance(head.count, ResourceGroup)

    head.command(f'echo "5" > {head.count.r5}')
    head.command(f'echo "3" > {head.count.r3}')

    def read(path):
        with open(path, 'r') as f:
            r = int(f.read())
        return r

    def multiply(x, y):
        return x * y

    def reformat(x, y):
        return {'x': x, 'y': y}

    middle = b.new_python_job()
    r3 = middle.call(read, head.count.r3)
    r5 = middle.call(read, head.count.r5)
    r_mult = middle.call(multiply, r3, r5)

    middle2 = b.new_python_job()
    r_mult = middle2.call(multiply, r_mult, 2)
    r_dict = middle2.call(reformat, r3, r5)

    tail = b.new_job()
    tail.command(f'cat {r3.as_str()} {r5.as_repr()} {r_mult.as_str()} {r_dict.as_json()}')

    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))
    assert res.get_job_log(4)['main'] == "3\n5\n30\n{\"x\": 3, \"y\": 5}\n", str(res.debug_info())


def test_python_job_can_write_to_resource_path(backend: ServiceBackend):
    b = batch(backend, default_python_image=PYTHON_DILL_IMAGE)

    def write(path):
        with open(path, 'w') as f:
            f.write('foo')

    head = b.new_python_job()
    head.call(write, head.ofile)

    tail = b.new_bash_job()
    tail.command(f'cat {head.ofile}')

    res = b.run()
    assert res
    assert tail._job_id
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))
    assert res.get_job_log(tail._job_id)['main'] == 'foo', str(res.debug_info())


def test_python_job_w_resource_group_unpack_jointly(backend: ServiceBackend):
    b = batch(backend, default_python_image=PYTHON_DILL_IMAGE)
    head = b.new_job()
    head.declare_resource_group(count={'r5': '{root}.r5', 'r3': '{root}.r3'})
    assert isinstance(head.count, ResourceGroup)

    head.command(f'echo "5" > {head.count.r5}')
    head.command(f'echo "3" > {head.count.r3}')

    def read_rg(root):
        with open(root['r3'], 'r') as f:
            r3 = int(f.read())
        with open(root['r5'], 'r') as f:
            r5 = int(f.read())
        return (r3, r5)

    def multiply(r):
        x, y = r
        return x * y

    middle = b.new_python_job()
    r = middle.call(read_rg, head.count)
    r_mult = middle.call(multiply, r)

    tail = b.new_job()
    tail.command(f'cat {r_mult.as_str()}')

    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))
    job_log_3 = res.get_job_log(3)
    assert job_log_3['main'] == "15\n", str((job_log_3, res.debug_info()))


def test_python_job_w_non_zero_ec(backend: ServiceBackend):
    b = batch(backend, default_python_image=PYTHON_DILL_IMAGE)
    j = b.new_python_job()

    def error():
        raise Exception("this should fail")

    j.call(error)
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))


def test_python_job_incorrect_signature(backend: ServiceBackend):
    b = batch(backend, default_python_image=PYTHON_DILL_IMAGE)

    def foo(pos_arg1, pos_arg2, *, kwarg1, kwarg2=1):
        print(pos_arg1, pos_arg2, kwarg1, kwarg2)

    j = b.new_python_job()

    with pytest.raises(BatchException):
        j.call(foo)
    with pytest.raises(BatchException):
        j.call(foo, 1)
    with pytest.raises(BatchException):
        j.call(foo, 1, 2)
    with pytest.raises(BatchException):
        j.call(foo, 1, kwarg1=2)
    with pytest.raises(BatchException):
        j.call(foo, 1, 2, 3)
    with pytest.raises(BatchException):
        j.call(foo, 1, 2, kwarg1=3, kwarg2=4, kwarg3=5)

    j.call(foo, 1, 2, kwarg1=3)
    j.call(foo, 1, 2, kwarg1=3, kwarg2=4)

    # `print` doesn't have a signature but other builtins like `abs` do
    j.call(print, 5)
    j.call(abs, -1)
    with pytest.raises(BatchException):
        j.call(abs, -1, 5)


def test_fail_fast(backend: ServiceBackend):
    b = batch(backend, cancel_after_n_failures=1)

    j1 = b.new_job()
    j1.command('false')

    j2 = b.new_job()
    j2.command('sleep 300')

    res = b.run()
    assert res
    job_status = res.get_job(2).status()
    assert job_status['state'] == 'Cancelled', str((job_status, res.debug_info()))


def test_service_backend_remote_tempdir_with_trailing_slash(backend):
    b = Batch(backend=backend)
    j1 = b.new_job()
    j1.command(f'echo hello > {j1.ofile}')
    j2 = b.new_job()
    j2.command(f'cat {j1.ofile}')
    b.run()


def test_service_backend_remote_tempdir_with_no_trailing_slash(backend):
    b = Batch(backend=backend)
    j1 = b.new_job()
    j1.command(f'echo hello > {j1.ofile}')
    j2 = b.new_job()
    j2.command(f'cat {j1.ofile}')
    b.run()


def test_large_command(backend: ServiceBackend):
    b = Batch(backend=backend)
    j1 = b.new_job()
    long_str = secrets.token_urlsafe(15 * 1024)
    j1.command(f'echo "{long_str}"')
    b.run()


def test_big_batch_which_uses_slow_path(backend: ServiceBackend):
    b = Batch(backend=backend)
    # 8 * 256 * 1024 = 2 MiB > 1 MiB max bunch size
    for _ in range(8):
        j1 = b.new_job()
        long_str = secrets.token_urlsafe(256 * 1024)
        j1.command(f'echo "{long_str}" > /dev/null')
    res = b.run()
    assert res
    assert not res._submission_info.used_fast_path
    batch_status = res.status()
    assert batch_status['state'] == 'success', str((res.debug_info()))


def test_query_on_batch_in_batch(backend: ServiceBackend, output_tmpdir: str):
    bb = Batch(backend=backend, default_python_image=HAIL_GENETICS_HAIL_IMAGE)

    tmp_ht_path = os.path.join(output_tmpdir, secrets.token_urlsafe(32))

    def qob_in_batch():
        import hail as hl

        hl.utils.range_table(10).write(tmp_ht_path, overwrite=True)

    j = bb.new_python_job()
    j.env('HAIL_QUERY_BACKEND', 'batch')
    j.env('HAIL_BATCH_BILLING_PROJECT', configuration_of(ConfigVariable.BATCH_BILLING_PROJECT, None, ''))
    j.env('HAIL_BATCH_REMOTE_TMPDIR', output_tmpdir)
    j.call(qob_in_batch)

    bb.run()


def test_basic_async_fun(backend: ServiceBackend):
    b = Batch(backend=backend)

    j = b.new_python_job()
    j.call(asyncio.sleep, 1)

    res = b.run()
    assert res
    batch_status = res.status()
    assert batch_status['state'] == 'success', str((res.debug_info()))


def test_async_fun_returns_value(backend: ServiceBackend):
    b = Batch(backend=backend)

    async def foo(i, j):
        await asyncio.sleep(1)
        return i * j

    j = b.new_python_job()
    result = j.call(foo, 2, 3)

    j = b.new_job()
    j.command(f'cat {result.as_str()}')

    res = b.run()
    assert res
    batch_status = res.status()
    assert batch_status['state'] == 'success', str((batch_status, res.debug_info()))
    job_log_2 = res.get_job_log(2)
    assert job_log_2['main'] == "6\n", str((job_log_2, res.debug_info()))


def test_specify_job_region(backend: ServiceBackend):
    b = batch(backend, cancel_after_n_failures=1)
    j = b.new_job('region')
    possible_regions = backend.supported_regions()
    j.regions(possible_regions)
    j.command('true')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_always_copy_output(backend: ServiceBackend, output_tmpdir: str):
    output_path = os.path.join(output_tmpdir, 'test_always_copy_output.txt')

    b = batch(backend)
    j = b.new_job()
    j.always_copy_output()
    j.command(f'echo "hello" > {j.ofile} && false')

    b.write_output(j.ofile, output_path)
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))

    b2 = batch(backend)
    input = b2.read_input(output_path)
    file_exists_j = b2.new_job()
    file_exists_j.command(f'cat {input}')

    res = b2.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))
    assert res.get_job_log(1)['main'] == "hello\n", str(res.debug_info())


def test_no_copy_output_on_failure(backend: ServiceBackend, output_tmpdir: str):
    output_path = os.path.join(output_tmpdir, 'test_no_copy_output.txt')

    b = batch(backend)
    j = b.new_job()
    j.command(f'echo "hello" > {j.ofile} && false')

    b.write_output(j.ofile, output_path)
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))

    b2 = batch(backend)
    input = b2.read_input(output_path)
    file_exists_j = b2.new_job()
    file_exists_j.command(f'cat {input}')

    res = b2.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))


def test_update_batch(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job()
    j.command('true')
    res = b.run()
    assert res

    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    j2 = b.new_job()
    j2.command('true')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_update_batch_with_dependencies(backend: ServiceBackend):
    b = batch(backend)
    j1 = b.new_job()
    j1.command('true')
    j2 = b.new_job()
    j2.command('false')
    res = b.run()
    assert res

    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))

    j3 = b.new_job()
    j3.command('true')
    j3.depends_on(j1)

    j4 = b.new_job()
    j4.command('true')
    j4.depends_on(j2)

    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))

    assert res.get_job(3).status()['state'] == 'Success', str((res_status, res.debug_info()))
    assert res.get_job(4).status()['state'] == 'Cancelled', str((res_status, res.debug_info()))


def test_update_batch_with_python_job_dependencies(backend: ServiceBackend):
    b = batch(backend)

    async def foo(i, j):
        await asyncio.sleep(1)
        return i * j

    j1 = b.new_python_job()
    j1.call(foo, 2, 3)

    res = b.run()
    assert res
    batch_status = res.status()
    assert batch_status['state'] == 'success', str((batch_status, res.debug_info()))

    j2 = b.new_python_job()
    j2.call(foo, 2, 3)

    res = b.run()
    assert res
    batch_status = res.status()
    assert batch_status['state'] == 'success', str((batch_status, res.debug_info()))

    j3 = b.new_python_job()
    j3.depends_on(j2)
    j3.call(foo, 2, 3)

    res = b.run()
    assert res
    batch_status = res.status()
    assert batch_status['state'] == 'success', str((batch_status, res.debug_info()))


def test_update_batch_from_batch_id(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job()
    j.command('true')
    res = b.run()
    assert res

    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    b2 = Batch.from_batch_id(res.id, backend=b._backend)
    j2 = b2.new_job()
    j2.command('true')
    res = b2.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


async def test_python_job_with_kwarg(fs: RouterAsyncFS, backend: ServiceBackend, output_tmpdir: str):
    def foo(*, kwarg):
        return kwarg

    b = batch(backend, default_python_image=PYTHON_DILL_IMAGE)
    j = b.new_python_job()
    r = j.call(foo, kwarg='hello world')

    output_path = os.path.join(output_tmpdir, 'test_python_job_with_kwarg')
    b.write_output(r.as_json(), output_path)
    res = b.run()
    assert isinstance(res, bc.Batch)

    assert res.status()['state'] == 'success', str((res, res.debug_info()))
    assert orjson.loads(await fs.read(output_path)) == 'hello world'


def test_tuple_recursive_resource_extraction_in_python_jobs(backend: ServiceBackend):
    b = batch(backend, default_python_image=PYTHON_DILL_IMAGE)

    def write(paths):
        if not isinstance(paths, tuple):
            raise ValueError('paths must be a tuple')
        for i, path in enumerate(paths):
            with open(path, 'w') as f:
                f.write(f'{i}')

    head = b.new_python_job()
    head.call(write, (head.ofile1, head.ofile2))

    tail = b.new_bash_job()
    tail.command(f'cat {head.ofile1}')
    tail.command(f'cat {head.ofile2}')

    res = b.run()
    assert res
    assert tail._job_id
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))
    assert res.get_job_log(tail._job_id)['main'] == '01', str(res.debug_info())


def test_list_recursive_resource_extraction_in_python_jobs(backend: ServiceBackend):
    b = batch(backend, default_python_image=PYTHON_DILL_IMAGE)

    def write(paths):
        for i, path in enumerate(paths):
            with open(path, 'w') as f:
                f.write(f'{i}')

    head = b.new_python_job()
    head.call(write, [head.ofile1, head.ofile2])

    tail = b.new_bash_job()
    tail.command(f'cat {head.ofile1}')
    tail.command(f'cat {head.ofile2}')

    res = b.run()
    assert res
    assert tail._job_id
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))
    assert res.get_job_log(tail._job_id)['main'] == '01', str(res.debug_info())


def test_dict_recursive_resource_extraction_in_python_jobs(backend: ServiceBackend):
    b = batch(backend, default_python_image=PYTHON_DILL_IMAGE)

    def write(kwargs):
        for k, v in kwargs.items():
            with open(v, 'w') as f:
                f.write(k)

    head = b.new_python_job()
    head.call(write, {'a': head.ofile1, 'b': head.ofile2})

    tail = b.new_bash_job()
    tail.command(f'cat {head.ofile1}')
    tail.command(f'cat {head.ofile2}')

    res = b.run()
    assert res
    assert tail._job_id
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))
    assert res.get_job_log(tail._job_id)['main'] == 'ab', str(res.debug_info())


def test_wait_on_empty_batch_update(backend: ServiceBackend):
    b = batch(backend)
    b.run(wait=True)
    b.run(wait=True)


def test_non_spot_job(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job()
    j.spot(False)
    j.command('echo hello')
    res = b.run()
    assert res
    assert res.get_job(1).status()['spec']['resources']['preemptible'] is False


def test_spot_unspecified_job(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job()
    j.command('echo hello')
    res = b.run()
    assert res
    assert res.get_job(1).status()['spec']['resources']['preemptible'] is True


def test_spot_true_job(backend: ServiceBackend):
    b = batch(backend)
    j = b.new_job()
    j.spot(True)
    j.command('echo hello')
    res = b.run()
    assert res
    assert res.get_job(1).status()['spec']['resources']['preemptible'] is True


def test_non_spot_batch(backend: ServiceBackend):
    b = batch(backend, default_spot=False)
    j1 = b.new_job()
    j1.command('echo hello')
    j2 = b.new_job()
    j2.command('echo hello')
    j3 = b.new_job()
    j3.spot(True)
    j3.command('echo hello')
    res = b.run()
    assert res
    assert res.get_job(1).status()['spec']['resources']['preemptible'] is False
    assert res.get_job(2).status()['spec']['resources']['preemptible'] is False
    assert res.get_job(3).status()['spec']['resources']['preemptible'] is True


def test_local_file_paths_error(backend: ServiceBackend):
    b = batch(backend)
    b.new_job()
    for input in ["hi.txt", "~/hello.csv", "./hey.tsv", "/sup.json", "file://yo.yaml"]:
        with pytest.raises(ValueError) as e:
            b.read_input(input)
        assert str(e.value).startswith("Local filepath detected")


@skip_in_azure
def test_validate_cloud_storage_policy(backend, monkeypatch):
    # buckets do not exist (bucket names can't contain the string "google" per
    # https://cloud.google.com/storage/docs/buckets)
    fake_bucket1 = "google"
    fake_bucket2 = "google1"
    no_bucket_error = "bucket does not exist"
    # bucket exists, but account does not have permissions on it
    no_perms_bucket = "test"
    no_perms_error = "does not have storage.buckets.get access"
    # bucket exists and account has permissions, but is set to use cold storage by default
    cold_bucket = "hail-test-cold-storage"
    cold_error = "configured to use cold storage by default"
    fake_uri1, fake_uri2, no_perms_uri, cold_uri = [
        f"gs://{bucket}/test" for bucket in [fake_bucket1, fake_bucket2, no_perms_bucket, cold_bucket]
    ]

    def _test_raises(exception_type, exception_msg, func):
        with pytest.raises(exception_type) as e:
            func()
        assert exception_msg in str(e.value)

    def _test_raises_no_bucket_error(remote_tmpdir, arg=None):
        _test_raises(
            ClientResponseError,
            no_bucket_error,
            lambda: ServiceBackend(remote_tmpdir=remote_tmpdir, gcs_bucket_allow_list=arg),
        )

    def _test_raises_cold_error(func):
        _test_raises(ValueError, cold_error, func)

    # no configuration, nonexistent buckets error
    _test_raises_no_bucket_error(fake_uri1)
    _test_raises_no_bucket_error(fake_uri2)

    # no configuration, no perms bucket errors
    _test_raises(ClientResponseError, no_perms_error, lambda: ServiceBackend(remote_tmpdir=no_perms_uri))

    # no configuration, cold bucket errors
    _test_raises_cold_error(lambda: ServiceBackend(remote_tmpdir=cold_uri))
    b = batch(backend)
    _test_raises_cold_error(lambda: b.read_input(cold_uri))
    j = b.new_job()
    j.command(f"echo hello > {j.ofile}")
    _test_raises_cold_error(lambda: b.write_output(j.ofile, cold_uri))

    # hailctl config, allowlisted nonexistent buckets don't error
    base_config = get_user_config()
    local_config = ConfigParser()
    local_config.read_dict({
        **{section: {key: val for key, val in base_config[section].items()} for section in base_config.sections()},
        **{"gcs": {"bucket_allow_list": f"{fake_bucket1},{fake_bucket2}"}},
    })

    def _get_user_config():
        return local_config

    monkeypatch.setattr(user_config, "get_user_config", _get_user_config)
    ServiceBackend(remote_tmpdir=fake_uri1)
    ServiceBackend(remote_tmpdir=fake_uri2)

    # environment variable config, only allowlisted nonexistent buckets don't error
    monkeypatch.setenv("HAIL_GCS_BUCKET_ALLOW_LIST", fake_bucket2)
    _test_raises_no_bucket_error(fake_uri1)
    ServiceBackend(remote_tmpdir=fake_uri2)

    # arg to constructor config, only allowlisted nonexistent buckets don't error
    arg = [fake_bucket1]
    ServiceBackend(remote_tmpdir=fake_uri1, gcs_bucket_allow_list=arg)
    _test_raises_no_bucket_error(fake_uri2, arg)
