from typing import AsyncIterator
import os
import pytest
import subprocess as sp
import tempfile

from hailtop import pip_version
from hailtop.batch import Batch, LocalBackend, ResourceGroup
from hailtop.batch.resource import JobResourceFile
from hailtop.batch.utils import concatenate


DOCKER_ROOT_IMAGE = os.environ.get('DOCKER_ROOT_IMAGE', 'ubuntu:22.04')
PYTHON_DILL_IMAGE = 'hailgenetics/python-dill:3.9-slim'
HAIL_GENETICS_HAIL_IMAGE = os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{pip_version()}')
REQUESTER_PAYS_PROJECT = os.environ.get('GCS_REQUESTER_PAYS_PROJECT')


@pytest.fixture(scope="session")
async def backend() -> AsyncIterator[LocalBackend]:
    lb = LocalBackend()
    try:
        yield lb
    finally:
        await lb.async_close()


@pytest.fixture
def batch(self, backend, requester_pays_project=None):
    return Batch(
        backend=backend,
        requester_pays_project=requester_pays_project
    )


def test_read_input_and_write_output(batch):
    with tempfile.NamedTemporaryFile('w') as input_file, \
            tempfile.NamedTemporaryFile('w') as output_file:
        input_file.write('abc')
        input_file.flush()

        b = batch
        input = b.read_input(input_file.name)
        b.write_output(input, output_file.name)
        b.run()

        assert open(input_file.name).read() == open(output_file.name).read()


def test_read_input_group(batch):
    with tempfile.NamedTemporaryFile('w') as input_file1, \
            tempfile.NamedTemporaryFile('w') as input_file2, \
            tempfile.NamedTemporaryFile('w') as output_file1, \
            tempfile.NamedTemporaryFile('w') as output_file2:

        input_file1.write('abc')
        input_file2.write('123')
        input_file1.flush()
        input_file2.flush()

        b = batch
        input = b.read_input_group(in1=input_file1.name,
                                   in2=input_file2.name)

        b.write_output(input.in1, output_file1.name)
        b.write_output(input.in2, output_file2.name)
        b.run()

        assert open(input_file1.name).read() == open(output_file1.name).read()
        assert open(input_file2.name).read() == open(output_file2.name).read()


def test_write_resource_group(batch):
    with tempfile.NamedTemporaryFile('w') as input_file1, \
            tempfile.NamedTemporaryFile('w') as input_file2, \
            tempfile.TemporaryDirectory() as output_dir:

        b = batch
        input = b.read_input_group(in1=input_file1.name,
                                   in2=input_file2.name)

        b.write_output(input, output_dir + '/foo')
        b.run()

        assert open(input_file1.name).read() == open(output_dir + '/foo.in1').read()
        assert open(input_file2.name).read() == open(output_dir + '/foo.in2').read()


def test_single_job(batch):
    with tempfile.NamedTemporaryFile('w') as output_file:
        msg = 'hello world'

        b = batch
        j = b.new_job()
        j.command(f'echo "{msg}" > {j.ofile}')
        b.write_output(j.ofile, output_file.name)
        b.run()

        assert open(output_file.name).read() == msg


def test_single_job_with_shell(batch):
    with tempfile.NamedTemporaryFile('w') as output_file:
        msg = 'hello world'

        b = batch
        j = b.new_job(shell='/bin/bash')
        j.command(f'echo "{msg}" > {j.ofile}')

        b.write_output(j.ofile, output_file.name)
        b.run()

        assert open(output_file.name).read() == msg


def test_single_job_with_nonsense_shell(batch):
    b = batch
    j = b.new_job(shell='/bin/ajdsfoijasidojf')
    j.image(DOCKER_ROOT_IMAGE)
    j.command(f'echo "hello"')
    with pytest.raises(Exception):
        b.run()

    b = batch
    j = b.new_job(shell='/bin/nonexistent')
    j.command(f'echo "hello"')
    with pytest.raises(Exception):
        b.run()


def test_single_job_with_intermediate_failure(batch):
    b = batch
    j = b.new_job()
    j.command(f'echoddd "hello"')
    j2 = b.new_job()
    j2.command(f'echo "world"')

    with pytest.raises(Exception):
        b.run()


def test_single_job_w_input(batch):
    with tempfile.NamedTemporaryFile('w') as input_file, \
            tempfile.NamedTemporaryFile('w') as output_file:
        msg = 'abc'
        input_file.write(msg)
        input_file.flush()

        b = batch
        input = b.read_input(input_file.name)
        j = b.new_job()
        j.command(f'cat {input} > {j.ofile}')
        b.write_output(j.ofile, output_file.name)
        b.run()

        assert open(output_file.name).read() == msg


def test_single_job_w_input_group(batch):
    with tempfile.NamedTemporaryFile('w') as input_file1, \
            tempfile.NamedTemporaryFile('w') as input_file2, \
            tempfile.NamedTemporaryFile('w') as output_file:
        msg1 = 'abc'
        msg2 = '123'

        input_file1.write(msg1)
        input_file2.write(msg2)
        input_file1.flush()
        input_file2.flush()

        b = batch
        input = b.read_input_group(in1=input_file1.name,
                                   in2=input_file2.name)
        j = b.new_job()
        j.command(f'cat {input.in1} {input.in2} > {j.ofile}')
        j.command(f'cat {input}.in1 {input}.in2')
        b.write_output(j.ofile, output_file.name)
        b.run()

        assert open(output_file.name).read() == msg1 + msg2


def test_single_job_bad_command(batch):
    b = batch
    j = b.new_job()
    j.command("foo")  # this should fail!
    with pytest.raises(sp.CalledProcessError):
        b.run()


def test_declare_resource_group(batch):
    with tempfile.NamedTemporaryFile('w') as output_file:
        msg = 'hello world'
        b = batch
        j = b.new_job()
        j.declare_resource_group(ofile={'log': "{root}.txt"})
        assert isinstance(j.ofile, ResourceGroup)
        j.command(f'echo "{msg}" > {j.ofile.log}')
        b.write_output(j.ofile.log, output_file.name)
        b.run()

        assert open(output_file.name).read() == msg


def test_resource_group_get_all_inputs(batch):
    b = batch
    input = b.read_input_group(fasta="foo",
                               idx="bar")
    j = b.new_job()
    j.command(f"cat {input.fasta}")
    assert input.fasta in j._inputs
    assert input.idx in j._inputs


def test_resource_group_get_all_mentioned(batch):
    b = batch
    j = b.new_job()
    j.declare_resource_group(foo={'bed': '{root}.bed', 'bim': '{root}.bim'})
    assert isinstance(j.foo, ResourceGroup)
    j.command(f"cat {j.foo.bed}")
    assert j.foo.bed in j._mentioned
    assert j.foo.bim not in j._mentioned


def test_resource_group_get_all_mentioned_dependent_jobs(batch):
    b = batch
    j = b.new_job()
    j.declare_resource_group(foo={'bed': '{root}.bed', 'bim': '{root}.bim'})
    j.command(f"cat")
    j2 = b.new_job()
    j2.command(f"cat {j.foo}")


def test_resource_group_get_all_outputs(batch):
    b = batch
    j1 = b.new_job()
    j1.declare_resource_group(foo={'bed': '{root}.bed', 'bim': '{root}.bim'})
    assert isinstance(j1.foo, ResourceGroup)
    j1.command(f"cat {j1.foo.bed}")
    j2 = b.new_job()
    j2.command(f"cat {j1.foo.bed}")

    for r in [j1.foo.bed, j1.foo.bim]:
        assert r in j1._internal_outputs
        assert r in j2._inputs

    assert j1.foo.bed in j1._mentioned
    assert j1.foo.bim not in j1._mentioned

    assert j1.foo.bed in j2._mentioned
    assert j1.foo.bim not in j2._mentioned

    assert j1.foo not in j1._mentioned


def test_multiple_isolated_jobs(batch):
    b = batch

    output_files = []
    try:
        output_files = [tempfile.NamedTemporaryFile('w') for _ in range(5)]

        for i, ofile in enumerate(output_files):
            msg = f'hello world {i}'
            j = b.new_job()
            j.command(f'echo "{msg}" > {j.ofile}')
            b.write_output(j.ofile, ofile.name)
        b.run()

        for i, ofile in enumerate(output_files):
            msg = f'hello world {i}'
            assert open(ofile.name).read() == msg
    finally:
        [ofile.close() for ofile in output_files]


def test_multiple_dependent_jobs(batch):
    with tempfile.NamedTemporaryFile('w') as output_file:
        b = batch
        j = b.new_job()
        j.command(f'echo "0" >> {j.ofile}')

        for i in range(1, 3):
            j2 = b.new_job()
            j2.command(f'echo "{i}" > {j2.tmp1}')
            j2.command(f'cat {j.ofile} {j2.tmp1} > {j2.ofile}')
            j = j2

        b.write_output(j.ofile, output_file.name)
        b.run()

        assert open(output_file.name).read() == "0\n1\n2"


def test_select_jobs(batch):
    b = batch
    for i in range(3):
        b.new_job(name=f'foo{i}')
    assert len(b.select_jobs('foo')) == 3


def test_scatter_gather(batch):
    with tempfile.NamedTemporaryFile('w') as output_file:
        b = batch

        for i in range(3):
            j = b.new_job(name=f'foo{i}')
            j.command(f'echo "{i}" > {j.ofile}')

        merger = b.new_job()
        merger.command('cat {files} > {ofile}'.format(files=' '.join([j.ofile for j in sorted(b.select_jobs('foo'),
                                                                                              key=lambda x: x.name,  # type: ignore
                                                                                              reverse=True)]),
                                                      ofile=merger.ofile))

        b.write_output(merger.ofile, output_file.name)
        b.run()

        assert open(output_file.name).read() == '2\n1\n0'


def test_add_extension_job_resource_file(batch):
    b = batch
    j = b.new_job()
    j.command(f'echo "hello" > {j.ofile}')
    assert isinstance(j.ofile, JobResourceFile)
    j.ofile.add_extension('.txt.bgz')
    assert j.ofile._value
    assert j.ofile._value.endswith('.txt.bgz')


def test_add_extension_input_resource_file(batch):
    input_file1 = '/tmp/data/example1.txt.bgz.foo'
    b = batch
    in1 = b.read_input(input_file1)
    assert in1._value
    assert in1._value.endswith('.txt.bgz.foo')


def test_file_name_space(batch):
    with tempfile.NamedTemporaryFile('w', prefix="some file name with (foo) spaces") as input_file, \
            tempfile.NamedTemporaryFile('w', prefix="another file name with (foo) spaces") as output_file:

        input_file.write('abc')
        input_file.flush()

        b = batch
        input = b.read_input(input_file.name)
        j = b.new_job()
        j.command(f'cat {input} > {j.ofile}')
        b.write_output(j.ofile, output_file.name)
        b.run()

        assert open(input_file.name).read() == open(output_file.name).read()


def test_resource_group_mentioned(batch):
    b = batch
    j = b.new_job()
    j.declare_resource_group(foo={'bed': '{root}.bed'})
    assert isinstance(j.foo, ResourceGroup)
    j.command(f'echo "hello" > {j.foo}')

    t2 = b.new_job()
    t2.command(f'echo "hello" >> {j.foo.bed}')
    b.run()


def test_envvar(batch):
    with tempfile.NamedTemporaryFile('w') as output_file:
        b = batch
        j = b.new_job()
        j.env('SOME_VARIABLE', '123abcdef')
        j.command(f'echo $SOME_VARIABLE > {j.ofile}')
        b.write_output(j.ofile, output_file.name)
        b.run()
        assert open(output_file.name).read() == '123abcdef'


def test_concatenate(batch):
    b = batch
    files = []
    for _ in range(10):
        j = b.new_job()
        j.command(f'touch {j.ofile}')
        files.append(j.ofile)
    concatenate(b, files, branching_factor=2)
    assert len(b._jobs) == 10 + (5 + 3 + 2 + 1)
    b.run()


def test_python_job(batch):
    with tempfile.NamedTemporaryFile('w') as output_file:
        b = batch
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
        tail.command(f'cat {r3.as_str()} {r5.as_repr()} {r_mult.as_str()} {r_dict.as_json()} > {tail.ofile}')

        b.write_output(tail.ofile, output_file.name)
        b.run()
        assert open(output_file.name).read() == '3\n5\n30\n{\"x\": 3, \"y\": 5}'


def test_backend_context_manager():
    with LocalBackend() as backend:
        b = Batch(backend=backend)
        b.run()


def test_failed_jobs_dont_stop_non_dependent_jobs(batch):
    with tempfile.NamedTemporaryFile('w') as output_file:
        b = batch

        head = b.new_job()
        head.command(f'echo 1 > {head.ofile}')

        head2 = b.new_job()
        head2.command('false')

        tail = b.new_job()
        tail.command(f'cat {head.ofile} > {tail.ofile}')
        b.write_output(tail.ofile, output_file.name)
        with pytest.raises(Exception):
            b.run()
        assert open(output_file.name).read() == '1'


def test_failed_jobs_stop_child_jobs(batch):
    with tempfile.NamedTemporaryFile('w') as output_file:
        b = batch

        head = b.new_job()
        head.command(f'echo 1 > {head.ofile}')
        head.command('false')

        head2 = b.new_job()
        head2.command(f'echo 2 > {head2.ofile}')

        tail = b.new_job()
        tail.command(f'cat {head.ofile} > {tail.ofile}')

        b.write_output(head2.ofile, output_file.name)
        b.write_output(tail.ofile, output_file.name)
        with pytest.raises(Exception):
            b.run()
        assert open(output_file.name).read() == '2'


def test_failed_jobs_stop_grandchild_jobs(batch):
    with tempfile.NamedTemporaryFile('w') as output_file:
        b = batch

        head = b.new_job()
        head.command(f'echo 1 > {head.ofile}')
        head.command('false')

        head2 = b.new_job()
        head2.command(f'echo 2 > {head2.ofile}')

        tail = b.new_job()
        tail.command(f'cat {head.ofile} > {tail.ofile}')

        tail2 = b.new_job()
        tail2.depends_on(tail)
        tail2.command(f'echo foo > {tail2.ofile}')

        b.write_output(head2.ofile, output_file.name)
        b.write_output(tail2.ofile, output_file.name)
        with pytest.raises(Exception):
            b.run()
        assert open(output_file.name).read() == '2'


def test_failed_jobs_dont_stop_always_run_jobs(batch):
    with tempfile.NamedTemporaryFile('w') as output_file:
        b = batch

        head = b.new_job()
        head.command(f'echo 1 > {head.ofile}')
        head.command('false')

        tail = b.new_job()
        tail.command(f'cat {head.ofile} > {tail.ofile}')
        tail.always_run()

        b.write_output(tail.ofile, output_file.name)
        with pytest.raises(Exception):
            b.run()
        assert open(output_file.name).read() == '1'
