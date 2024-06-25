import asyncio
import os
import secrets

import orjson
import pytest

import hailtop.batch_client.client as bc
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.batch import Batch, ResourceGroup, ServiceBackend
from hailtop.batch.exceptions import BatchException
from hailtop.config import configuration_of
from hailtop.config.variables import ConfigVariable

from .utils import (
    HAIL_GENETICS_HAIL_IMAGE,
    PYTHON_DILL_IMAGE,
    batch,
)


def test_python_job(service_backend: ServiceBackend):
    b = batch(service_backend, default_python_image=PYTHON_DILL_IMAGE)
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


def test_python_job_w_resource_group_unpack_individually(service_backend: ServiceBackend):
    b = batch(service_backend, default_python_image=PYTHON_DILL_IMAGE)
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


def test_python_job_can_write_to_resource_path(service_backend: ServiceBackend):
    b = batch(service_backend, default_python_image=PYTHON_DILL_IMAGE)

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


def test_python_job_w_resource_group_unpack_jointly(service_backend: ServiceBackend):
    b = batch(service_backend, default_python_image=PYTHON_DILL_IMAGE)
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


def test_python_job_w_non_zero_ec(service_backend: ServiceBackend):
    b = batch(service_backend, default_python_image=PYTHON_DILL_IMAGE)
    j = b.new_python_job()

    def error():
        raise Exception("this should fail")

    j.call(error)
    res = b.run()
    assert res
    res_status = res.status()
    assert res_status['state'] == 'failure', str((res_status, res.debug_info()))


def test_python_job_incorrect_signature(service_backend: ServiceBackend):
    b = batch(service_backend, default_python_image=PYTHON_DILL_IMAGE)

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


def test_query_on_batch_in_batch(service_backend: ServiceBackend, output_tmpdir: str):
    bb = Batch(backend=service_backend, default_python_image=HAIL_GENETICS_HAIL_IMAGE)

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


def test_basic_async_fun(service_backend: ServiceBackend):
    b = Batch(backend=service_backend)

    j = b.new_python_job()
    j.call(asyncio.sleep, 1)

    res = b.run()
    assert res
    batch_status = res.status()
    assert batch_status['state'] == 'success', str((res.debug_info()))


def test_async_fun_returns_value(service_backend: ServiceBackend):
    b = Batch(backend=service_backend)

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


def test_update_batch_with_python_job_dependencies(service_backend: ServiceBackend):
    b = batch(service_backend)

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


async def test_python_job_with_kwarg(fs: RouterAsyncFS, service_backend: ServiceBackend, output_tmpdir: str):
    def foo(*, kwarg):
        return kwarg

    b = batch(service_backend, default_python_image=PYTHON_DILL_IMAGE)
    j = b.new_python_job()
    r = j.call(foo, kwarg='hello world')

    output_path = os.path.join(output_tmpdir, 'test_python_job_with_kwarg')
    b.write_output(r.as_json(), output_path)
    res = b.run()
    assert isinstance(res, bc.Batch)

    assert res.status()['state'] == 'success', str((res, res.debug_info()))
    assert orjson.loads(await fs.read(output_path)) == 'hello world'


def test_tuple_recursive_resource_extraction_in_python_jobs(service_backend: ServiceBackend):
    b = batch(service_backend, default_python_image=PYTHON_DILL_IMAGE)

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


def test_list_recursive_resource_extraction_in_python_jobs(service_backend: ServiceBackend):
    b = batch(service_backend, default_python_image=PYTHON_DILL_IMAGE)

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


def test_dict_recursive_resource_extraction_in_python_jobs(service_backend: ServiceBackend):
    b = batch(service_backend, default_python_image=PYTHON_DILL_IMAGE)

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
