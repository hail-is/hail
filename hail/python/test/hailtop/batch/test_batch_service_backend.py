import os
import secrets
from configparser import ConfigParser
from shlex import quote as shq
from typing import Tuple

import pytest

from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.aiotools.validators import validate_file
from hailtop.batch import Batch, ResourceGroup, ServiceBackend
from hailtop.batch.exceptions import BatchException
from hailtop.batch.globals import arg_max
from hailtop.config import get_user_config, user_config
from hailtop.httpx import ClientResponseError
from hailtop.test_utils import skip_in_azure
from hailtop.utils import grouped

from .utils import (
    REQUESTER_PAYS_PROJECT,
    batch,
)


def test_single_task_no_io(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job()
    j.command('echo hello')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_input(
    service_backend: ServiceBackend, upload_test_files: Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]]
):
    (url1, data1), _, _ = upload_test_files
    b = batch(service_backend)
    input = b.read_input(url1)
    j = b.new_job()
    j.command(f'cat {input}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_input_resource_group(
    service_backend: ServiceBackend, upload_test_files: Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]]
):
    (url1, data1), _, _ = upload_test_files
    b = batch(service_backend)
    input = b.read_input_group(foo=url1)
    j = b.new_job()
    j.storage('10Gi')
    j.command(f'cat {input.foo}')
    j.command(f'cat {input}.foo')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_output(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job(attributes={'a': 'bar', 'b': 'foo'})
    j.command(f'echo hello > {j.ofile}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_write_output(service_backend: ServiceBackend, output_tmpdir: str):
    b = batch(service_backend)
    j = b.new_job()
    j.command(f'echo hello > {j.ofile}')
    b.write_output(j.ofile, os.path.join(output_tmpdir, 'test_single_task_output.txt'))
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_resource_group(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job()
    j.declare_resource_group(output={'foo': '{root}.foo'})
    assert isinstance(j.output, ResourceGroup)
    j.command(f'echo "hello" > {j.output.foo}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_task_write_resource_group(service_backend: ServiceBackend, output_tmpdir: str):
    b = batch(service_backend)
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


def test_multiple_dependent_tasks(service_backend: ServiceBackend, output_tmpdir: str):
    output_file = os.path.join(output_tmpdir, 'test_multiple_dependent_tasks.txt')
    b = batch(service_backend)
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


def test_specify_cpu(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job()
    j.cpu('0.5')
    j.command(f'echo "hello" > {j.ofile}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_specify_memory(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job()
    j.memory('100M')
    j.command(f'echo "hello" > {j.ofile}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_scatter_gather(service_backend: ServiceBackend):
    b = batch(service_backend)

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
    service_backend: ServiceBackend,
    upload_test_files: Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]],
    output_tmpdir: str,
):
    _, _, (url3, data3) = upload_test_files
    b = batch(service_backend)
    input = b.read_input(url3)
    j = b.new_job()
    j.command(f'cat {input} > {j.ofile}')
    b.write_output(j.ofile, os.path.join(output_tmpdir, 'hello (foo) spaces.txt'))
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_dry_run(service_backend: ServiceBackend, output_tmpdir: str):
    b = batch(service_backend)
    j = b.new_job()
    j.command(f'echo hello > {j.ofile}')
    b.write_output(j.ofile, os.path.join(output_tmpdir, 'test_single_job_output.txt'))
    b.run(dry_run=True)


def test_verbose(
    service_backend: ServiceBackend,
    upload_test_files: Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]],
    output_tmpdir: str,
):
    (url1, data1), _, _ = upload_test_files
    b = batch(service_backend)
    input = b.read_input(url1)
    j = b.new_job()
    j.command(f'cat {input}')
    b.write_output(input, os.path.join(output_tmpdir, 'hello.txt'))
    res = b.run(verbose=True)
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_cloudfuse_fails_with_read_write_mount_option(
    fs: RouterAsyncFS, service_backend: ServiceBackend, output_bucket_path
):
    bucket, path, output_tmpdir = output_bucket_path

    b = batch(service_backend)
    j = b.new_job()
    j.command(f'mkdir -p {path}; echo head > {path}/cloudfuse_test_1')
    j.cloudfuse(bucket, f'/{bucket}', read_only=False)

    try:
        b.run()
    except ClientResponseError as e:
        assert 'Only read-only cloudfuse requests are supported' in e.body, e.body
    else:
        assert False


def test_cloudfuse_fails_with_io_mount_point(fs: RouterAsyncFS, service_backend: ServiceBackend, output_bucket_path):
    bucket, path, output_tmpdir = output_bucket_path

    b = batch(service_backend)
    j = b.new_job()
    j.command(f'mkdir -p {path}; echo head > {path}/cloudfuse_test_1')
    j.cloudfuse(bucket, '/io', read_only=True)

    try:
        b.run()
    except ClientResponseError as e:
        assert 'Cloudfuse requests with mount_path=/io are not supported' in e.body, e.body
    else:
        assert False


def test_cloudfuse_read_only(service_backend: ServiceBackend, output_bucket_path):
    bucket, path, output_tmpdir = output_bucket_path

    b = batch(service_backend)
    j = b.new_job()
    j.command(f'mkdir -p {path}; echo head > {path}/cloudfuse_test_1')
    j.cloudfuse(bucket, f'/{bucket}', read_only=True)

    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))


def test_cloudfuse_implicit_dirs(fs: RouterAsyncFS, service_backend: ServiceBackend, upload_test_files):
    (url1, data1), _, _ = upload_test_files
    parsed_url1 = fs.parse_url(url1)
    object_name = parsed_url1.path
    bucket_name = '/'.join(parsed_url1.bucket_parts)

    b = batch(service_backend)
    j = b.new_job()
    j.command('cat ' + os.path.join('/cloudfuse', object_name))
    j.cloudfuse(bucket_name, '/cloudfuse', read_only=True)

    res = b.run()
    assert res
    res_status = res.status()
    assert res.get_job_log(1)['main'] == data1.decode()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_cloudfuse_empty_string_bucket_fails(service_backend: ServiceBackend, output_bucket_path):
    bucket, path, output_tmpdir = output_bucket_path

    b = batch(service_backend)
    j = b.new_job()
    with pytest.raises(BatchException):
        j.cloudfuse('', '/empty_bucket')
    with pytest.raises(BatchException):
        j.cloudfuse(bucket, '')


async def test_cloudfuse_submount_in_io_doesnt_rm_bucket(
    fs: RouterAsyncFS, service_backend: ServiceBackend, output_bucket_path
):
    bucket, path, output_tmpdir = output_bucket_path

    should_still_exist_url = os.path.join(output_tmpdir, 'should-still-exist')
    await fs.write(should_still_exist_url, b'should-still-exist')

    b = batch(service_backend)
    j = b.new_job()
    j.cloudfuse(bucket, '/io/cloudfuse')
    j.command('ls /io/cloudfuse/')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))
    assert await fs.read(should_still_exist_url) == b'should-still-exist'


@skip_in_azure
def test_fuse_requester_pays(service_backend: ServiceBackend):
    assert REQUESTER_PAYS_PROJECT
    b = batch(service_backend, requester_pays_project=REQUESTER_PAYS_PROJECT)
    j = b.new_job()
    j.cloudfuse('hail-test-requester-pays-fds32', '/fuse-bucket')
    j.command('cat /fuse-bucket/hello')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


@skip_in_azure
def test_fuse_non_requester_pays_bucket_when_requester_pays_project_specified(
    service_backend: ServiceBackend, output_bucket_path
):
    bucket, path, output_tmpdir = output_bucket_path
    assert REQUESTER_PAYS_PROJECT

    b = batch(service_backend, requester_pays_project=REQUESTER_PAYS_PROJECT)
    j = b.new_job()
    j.command('ls /fuse-bucket')
    j.cloudfuse(bucket, '/fuse-bucket', read_only=True)

    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


@skip_in_azure
def test_requester_pays(service_backend: ServiceBackend):
    assert REQUESTER_PAYS_PROJECT
    b = batch(service_backend, requester_pays_project=REQUESTER_PAYS_PROJECT)
    input = b.read_input('gs://hail-test-requester-pays-fds32/hello')
    j = b.new_job()
    j.command(f'cat {input}')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_benchmark_lookalike_workflow(service_backend: ServiceBackend, output_tmpdir):
    b = batch(service_backend)

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


def test_envvar(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job()
    j.env('SOME_VARIABLE', '123abcdef')
    j.command('[ $SOME_VARIABLE = "123abcdef" ]')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_job_with_shell(service_backend: ServiceBackend):
    msg = 'hello world'
    b = batch(service_backend)
    j = b.new_job(shell='/bin/sh')
    j.command(f'echo "{msg}"')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_single_job_with_nonsense_shell(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job(shell='/bin/ajdsfoijasidojf')
    j.command('echo "hello"')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))


def test_single_job_with_intermediate_failure(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job()
    j.command('echoddd "hello"')
    j2 = b.new_job()
    j2.command('echo "world"')

    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))


def test_input_directory(
    service_backend: ServiceBackend, upload_test_files: Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]]
):
    (url1, data1), _, _ = upload_test_files
    b = batch(service_backend)
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


def test_fail_fast(service_backend: ServiceBackend):
    b = batch(service_backend, cancel_after_n_failures=1)
    j1 = b.new_job()
    j1.command('false')

    j2 = b.new_job()
    j2.command('sleep 300')

    res = b.run()
    assert res
    job_status = res.get_job(2).status()
    assert job_status['state'] == 'Cancelled', str((job_status, res.debug_info()))


def test_service_backend_remote_tempdir_with_trailing_slash(service_backend: ServiceBackend):
    b = Batch(backend=service_backend)
    j1 = b.new_job()
    j1.command(f'echo hello > {j1.ofile}')
    j2 = b.new_job()
    j2.command(f'cat {j1.ofile}')
    b.run()


def test_service_backend_remote_tempdir_with_no_trailing_slash(service_backend: ServiceBackend):
    b = Batch(backend=service_backend)
    j1 = b.new_job()
    j1.command(f'echo hello > {j1.ofile}')
    j2 = b.new_job()
    j2.command(f'cat {j1.ofile}')
    b.run()


def test_large_command(service_backend: ServiceBackend):
    b = Batch(backend=service_backend)
    j1 = b.new_job()
    long_str = secrets.token_urlsafe(15 * 1024)
    j1.command(f'echo "{long_str}"')
    b.run()


def test_big_batch_which_uses_slow_path(service_backend: ServiceBackend):
    b = Batch(backend=service_backend)
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


def test_specify_job_region(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job('region')
    possible_regions = service_backend.supported_regions()
    j.regions(possible_regions)
    j.command('true')
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))


def test_job_regions_controls_job_execution_region(service_backend: ServiceBackend):
    the_region = service_backend.supported_regions()[0]

    b = batch(service_backend)
    j = b.new_job()
    j.regions([the_region])
    j.command('true')
    res = b.run()

    assert res
    job_status = res.get_job(1).status()
    assert job_status['status']['region'] == the_region, str((job_status, res.debug_info()))


def test_job_regions_overrides_batch_regions(service_backend: ServiceBackend):
    the_region = service_backend.supported_regions()[0]

    b = batch(service_backend, default_regions=['some-other-region'])
    j = b.new_job()
    j.regions([the_region])
    j.command('true')
    res = b.run()

    assert res
    job_status = res.get_job(1).status()
    assert job_status['status']['region'] == the_region, str((job_status, res.debug_info()))


def test_always_copy_output(service_backend: ServiceBackend, output_tmpdir: str):
    output_path = os.path.join(output_tmpdir, 'test_always_copy_output.txt')

    b = batch(service_backend)
    j = b.new_job()
    j.always_copy_output()
    j.command(f'echo "hello" > {j.ofile} && false')

    b.write_output(j.ofile, output_path)
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))

    b2 = batch(service_backend)
    input = b2.read_input(output_path)
    file_exists_j = b2.new_job()
    file_exists_j.command(f'cat {input}')

    res = b2.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'success', str((res_status, res.debug_info()))
    assert res.get_job_log(1)['main'] == "hello\n", str(res.debug_info())


def test_no_copy_output_on_failure(service_backend: ServiceBackend, output_tmpdir: str):
    output_path = os.path.join(output_tmpdir, 'test_no_copy_output.txt')

    b = batch(service_backend)
    j = b.new_job()
    j.command(f'echo "hello" > {j.ofile} && false')

    b.write_output(j.ofile, output_path)
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))

    b2 = batch(service_backend)
    input = b2.read_input(output_path)
    file_exists_j = b2.new_job()
    file_exists_j.command(f'cat {input}')

    res = b2.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))


def test_update_batch(service_backend: ServiceBackend):
    b = batch(service_backend)
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


def test_update_batch_with_dependencies(service_backend: ServiceBackend):
    b = batch(service_backend)
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


def test_update_batch_from_batch_id(service_backend: ServiceBackend):
    b = batch(service_backend)
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


def test_wait_on_empty_batch_update(service_backend: ServiceBackend):
    b = batch(service_backend)
    b.run(wait=True)
    b.run(wait=True)


def test_non_spot_job(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job()
    j.spot(False)
    j.command('echo hello')
    res = b.run()
    assert res
    assert res.get_job(1).status()['spec']['resources']['preemptible'] is False


def test_spot_unspecified_job(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job()
    j.command('echo hello')
    res = b.run()
    assert res
    assert res.get_job(1).status()['spec']['resources']['preemptible'] is True


def test_spot_true_job(service_backend: ServiceBackend):
    b = batch(service_backend)
    j = b.new_job()
    j.spot(True)
    j.command('echo hello')
    res = b.run()
    assert res
    assert res.get_job(1).status()['spec']['resources']['preemptible'] is True


def test_non_spot_batch(service_backend: ServiceBackend):
    b = batch(service_backend, default_spot=False)
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


def test_local_file_paths_error(service_backend: ServiceBackend):
    b = batch(service_backend)
    b.new_job()
    for input in ["hi.txt", "~/hello.csv", "./hey.tsv", "/sup.json", "file://yo.yaml"]:
        with pytest.raises(ValueError) as e:
            b.read_input(input)
        assert str(e.value).startswith("Local filepath detected")


@skip_in_azure
async def test_validate_cloud_storage_policy(service_backend: ServiceBackend, monkeypatch):
    # buckets do not exist (bucket names can't contain the string "google" per
    # https://cloud.google.com/storage/docs/buckets)
    fake_bucket1 = "google"
    fake_bucket2 = "google1"
    no_bucket_error = "bucket does not exist"
    # bucket exists, but account does not have permissions on it
    no_perms_bucket = "hail-test-no-perms"
    no_perms_error = "does not have storage.objects.get access"
    # bucket is a public access bucket (https://cloud.google.com/storage/docs/access-public-data)
    public_access_bucket = "hail-common"
    # bucket exists and account has permissions, but is set to use cold storage by default
    cold_bucket = "hail-test-cold-storage"
    cold_error = "configured to use cold storage by default"
    fake_uri1, fake_uri2, no_perms_uri, cold_uri = [
        f"gs://{bucket}/test" for bucket in [fake_bucket1, fake_bucket2, no_perms_bucket, cold_bucket]
    ]
    public_access_uri1 = f"gs://{public_access_bucket}/references"
    public_access_uri2 = f"{public_access_uri1}/human_g1k_v37.fasta.gz"
    public_access_uri3 = f"gs://{public_access_bucket}/36bbda16-2d47-4be8-ad9e-1c6ef5b7c216"

    async def _test_raises(exception_type, exception_msg, func):
        with pytest.raises(exception_type) as e:
            await func()
        assert exception_msg in str(e.value)

    async def _test_raises_no_bucket_error(remote_tmpdir, arg=None):
        await _test_raises(
            ClientResponseError,
            no_bucket_error,
            lambda: ServiceBackend(remote_tmpdir=remote_tmpdir, gcs_bucket_allow_list=arg),
        )

    async def _test_raises_cold_error(func):
        await _test_raises(ValueError, cold_error, func)

    async def _with_temp_fs(func):
        async def inner():
            async with RouterAsyncFS() as fs:
                await func(fs)

        await inner()

    # no configuration, nonexistent buckets error
    await _test_raises_no_bucket_error(fake_uri1)
    await _test_raises_no_bucket_error(fake_uri2)

    # no configuration, public access bucket doesn't error unless the object doesn't exist
    await _with_temp_fs(lambda fs: validate_file(public_access_uri1, fs))
    await _with_temp_fs(lambda fs: validate_file(public_access_uri2, fs))
    await _with_temp_fs(
        lambda fs: _test_raises(FileNotFoundError, public_access_uri3, lambda: validate_file(public_access_uri3, fs))
    )

    # no configuration, no perms bucket errors
    await _test_raises(ClientResponseError, no_perms_error, lambda: ServiceBackend(remote_tmpdir=no_perms_uri))

    # no configuration, cold bucket errors
    await _test_raises_cold_error(lambda: ServiceBackend(remote_tmpdir=cold_uri))
    b = batch(service_backend)
    await _test_raises_cold_error(lambda: b.read_input(cold_uri))
    j = b.new_job()
    j.command(f"echo hello > {j.ofile}")
    await _test_raises_cold_error(lambda: b.write_output(j.ofile, cold_uri))

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
    await _test_raises_no_bucket_error(fake_uri1)
    ServiceBackend(remote_tmpdir=fake_uri2)

    # arg to constructor config, only allowlisted nonexistent buckets don't error
    arg = [fake_bucket1]
    ServiceBackend(remote_tmpdir=fake_uri1, gcs_bucket_allow_list=arg)
    await _test_raises_no_bucket_error(fake_uri2, arg)
