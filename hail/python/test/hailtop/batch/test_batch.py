import secrets
import unittest
import os
import subprocess as sp
import tempfile
from shlex import quote as shq
import uuid
import re

from hailtop.batch import Batch, ServiceBackend, LocalBackend
from hailtop.batch.exceptions import BatchException
from hailtop.batch.globals import arg_max
from hailtop.utils import grouped, async_to_blocking
from hailtop.config import get_user_config
from hailtop.batch.utils import concatenate
from hailtop.aiotools.router_fs import RouterAsyncFS

from ..utils import fails_in_azure, skip_in_azure


DOCKER_ROOT_IMAGE = os.environ['DOCKER_ROOT_IMAGE']
PYTHON_DILL_IMAGE = os.environ['PYTHON_DILL_IMAGE']


class LocalTests(unittest.TestCase):
    def batch(self, requester_pays_project=None):
        return Batch(backend=LocalBackend(),
                     requester_pays_project=requester_pays_project)

    def read(self, file):
        with open(file, 'r') as f:
            result = f.read().rstrip()
        return result

    def assert_same_file(self, file1, file2):
        assert self.read(file1).rstrip() == self.read(file2).rstrip()

    def test_read_input_and_write_output(self):
        with tempfile.NamedTemporaryFile('w') as input_file, \
                tempfile.NamedTemporaryFile('w') as output_file:
            input_file.write('abc')
            input_file.flush()

            b = self.batch()
            input = b.read_input(input_file.name)
            b.write_output(input, output_file.name)
            b.run()

            self.assert_same_file(input_file.name, output_file.name)

    def test_read_input_group(self):
        with tempfile.NamedTemporaryFile('w') as input_file1, \
                tempfile.NamedTemporaryFile('w') as input_file2, \
                tempfile.NamedTemporaryFile('w') as output_file1, \
                tempfile.NamedTemporaryFile('w') as output_file2:

            input_file1.write('abc')
            input_file2.write('123')
            input_file1.flush()
            input_file2.flush()

            b = self.batch()
            input = b.read_input_group(in1=input_file1.name,
                                       in2=input_file2.name)

            b.write_output(input.in1, output_file1.name)
            b.write_output(input.in2, output_file2.name)
            b.run()

            self.assert_same_file(input_file1.name, output_file1.name)
            self.assert_same_file(input_file2.name, output_file2.name)

    def test_write_resource_group(self):
        with tempfile.NamedTemporaryFile('w') as input_file1, \
                tempfile.NamedTemporaryFile('w') as input_file2, \
                tempfile.TemporaryDirectory() as output_dir:

            b = self.batch()
            input = b.read_input_group(in1=input_file1.name,
                                       in2=input_file2.name)

            b.write_output(input, output_dir + '/foo')
            b.run()

            self.assert_same_file(input_file1.name, output_dir + '/foo.in1')
            self.assert_same_file(input_file2.name, output_dir + '/foo.in2')

    def test_single_job(self):
        with tempfile.NamedTemporaryFile('w') as output_file:
            msg = 'hello world'

            b = self.batch()
            j = b.new_job()
            j.command(f'echo "{msg}" > {j.ofile}')
            b.write_output(j.ofile, output_file.name)
            b.run()

            assert self.read(output_file.name) == msg

    def test_single_job_with_shell(self):
        with tempfile.NamedTemporaryFile('w') as output_file:
            msg = 'hello world'

            b = self.batch()
            j = b.new_job(shell='/bin/bash')
            j.command(f'echo "{msg}" > {j.ofile}')

            b.write_output(j.ofile, output_file.name)
            b.run()

            assert self.read(output_file.name) == msg

    def test_single_job_with_nonsense_shell(self):
        b = self.batch()
        j = b.new_job(shell='/bin/ajdsfoijasidojf')
        j.image(DOCKER_ROOT_IMAGE)
        j.command(f'echo "hello"')
        self.assertRaises(Exception, b.run)

        b = self.batch()
        j = b.new_job(shell='/bin/nonexistent')
        j.command(f'echo "hello"')
        self.assertRaises(Exception, b.run)

    def test_single_job_with_intermediate_failure(self):
        b = self.batch()
        j = b.new_job()
        j.command(f'echoddd "hello"')
        j2 = b.new_job()
        j2.command(f'echo "world"')

        self.assertRaises(Exception, b.run)

    def test_single_job_w_input(self):
        with tempfile.NamedTemporaryFile('w') as input_file, \
                tempfile.NamedTemporaryFile('w') as output_file:
            msg = 'abc'
            input_file.write(msg)
            input_file.flush()

            b = self.batch()
            input = b.read_input(input_file.name)
            j = b.new_job()
            j.command(f'cat {input} > {j.ofile}')
            b.write_output(j.ofile, output_file.name)
            b.run()

            assert self.read(output_file.name) == msg

    def test_single_job_w_input_group(self):
        with tempfile.NamedTemporaryFile('w') as input_file1, \
                tempfile.NamedTemporaryFile('w') as input_file2, \
                tempfile.NamedTemporaryFile('w') as output_file:
            msg1 = 'abc'
            msg2 = '123'

            input_file1.write(msg1)
            input_file2.write(msg2)
            input_file1.flush()
            input_file2.flush()

            b = self.batch()
            input = b.read_input_group(in1=input_file1.name,
                                       in2=input_file2.name)
            j = b.new_job()
            j.command(f'cat {input.in1} {input.in2} > {j.ofile}')
            j.command(f'cat {input}.in1 {input}.in2')
            b.write_output(j.ofile, output_file.name)
            b.run()

            assert self.read(output_file.name) == msg1 + msg2

    def test_single_job_bad_command(self):
        b = self.batch()
        j = b.new_job()
        j.command("foo")  # this should fail!
        with self.assertRaises(sp.CalledProcessError):
            b.run()

    def test_declare_resource_group(self):
        with tempfile.NamedTemporaryFile('w') as output_file:
            msg = 'hello world'
            b = self.batch()
            j = b.new_job()
            j.declare_resource_group(ofile={'log': "{root}.txt"})
            j.command(f'echo "{msg}" > {j.ofile.log}')
            b.write_output(j.ofile.log, output_file.name)
            b.run()

            assert self.read(output_file.name) == msg

    def test_resource_group_get_all_inputs(self):
        b = self.batch()
        input = b.read_input_group(fasta="foo",
                                   idx="bar")
        j = b.new_job()
        j.command(f"cat {input.fasta}")
        assert input.fasta in j._inputs
        assert input.idx in j._inputs

    def test_resource_group_get_all_mentioned(self):
        b = self.batch()
        j = b.new_job()
        j.declare_resource_group(foo={'bed': '{root}.bed', 'bim': '{root}.bim'})
        j.command(f"cat {j.foo.bed}")
        assert j.foo.bed in j._mentioned
        assert j.foo.bim not in j._mentioned

    def test_resource_group_get_all_mentioned_dependent_jobs(self):
        b = self.batch()
        j = b.new_job()
        j.declare_resource_group(foo={'bed': '{root}.bed', 'bim': '{root}.bim'})
        j.command(f"cat")
        j2 = b.new_job()
        j2.command(f"cat {j.foo}")

    def test_resource_group_get_all_outputs(self):
        b = self.batch()
        j1 = b.new_job()
        j1.declare_resource_group(foo={'bed': '{root}.bed', 'bim': '{root}.bim'})
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

    def test_multiple_isolated_jobs(self):
        b = self.batch()

        output_files = []
        try:
            output_files = [tempfile.NamedTemporaryFile('w') for i in range(5)]

            for i, ofile in enumerate(output_files):
                msg = f'hello world {i}'
                j = b.new_job()
                j.command(f'echo "{msg}" > {j.ofile}')
                b.write_output(j.ofile, ofile.name)
            b.run()

            for i, ofile in enumerate(output_files):
                msg = f'hello world {i}'
                assert self.read(ofile.name) == msg
        finally:
            [ofile.close() for ofile in output_files]

    def test_multiple_dependent_jobs(self):
        with tempfile.NamedTemporaryFile('w') as output_file:
            b = self.batch()
            j = b.new_job()
            j.command(f'echo "0" >> {j.ofile}')

            for i in range(1, 3):
                j2 = b.new_job()
                j2.command(f'echo "{i}" > {j2.tmp1}')
                j2.command(f'cat {j.ofile} {j2.tmp1} > {j2.ofile}')
                j = j2

            b.write_output(j.ofile, output_file.name)
            b.run()

            assert self.read(output_file.name) == "0\n1\n2"

    def test_select_jobs(self):
        b = self.batch()
        for i in range(3):
            b.new_job(name=f'foo{i}')
        self.assertTrue(len(b.select_jobs('foo')) == 3)

    def test_scatter_gather(self):
        with tempfile.NamedTemporaryFile('w') as output_file:
            b = self.batch()

            for i in range(3):
                j = b.new_job(name=f'foo{i}')
                j.command(f'echo "{i}" > {j.ofile}')

            merger = b.new_job()
            merger.command('cat {files} > {ofile}'.format(files=' '.join([j.ofile for j in sorted(b.select_jobs('foo'),
                                                                                                  key=lambda x: x.name,
                                                                                                  reverse=True)]),
                                                          ofile=merger.ofile))

            b.write_output(merger.ofile, output_file.name)
            b.run()

            assert self.read(output_file.name) == '2\n1\n0'

    def test_add_extension_job_resource_file(self):
        b = self.batch()
        j = b.new_job()
        j.command(f'echo "hello" > {j.ofile}')
        j.ofile.add_extension('.txt.bgz')
        assert j.ofile._value.endswith('.txt.bgz')

    def test_add_extension_input_resource_file(self):
        input_file1 = '/tmp/data/example1.txt.bgz.foo'
        b = self.batch()
        in1 = b.read_input(input_file1)
        assert in1._value.endswith('.txt.bgz.foo')

    def test_file_name_space(self):
        with tempfile.NamedTemporaryFile('w', prefix="some file name with (foo) spaces") as input_file, \
                tempfile.NamedTemporaryFile('w', prefix="another file name with (foo) spaces") as output_file:

            input_file.write('abc')
            input_file.flush()

            b = self.batch()
            input = b.read_input(input_file.name)
            j = b.new_job()
            j.command(f'cat {input} > {j.ofile}')
            b.write_output(j.ofile, output_file.name)
            b.run()

            self.assert_same_file(input_file.name, output_file.name)

    def test_resource_group_mentioned(self):
        b = self.batch()
        j = b.new_job()
        j.declare_resource_group(foo={'bed': '{root}.bed'})
        j.command(f'echo "hello" > {j.foo}')

        t2 = b.new_job()
        t2.command(f'echo "hello" >> {j.foo.bed}')
        b.run()

    def test_envvar(self):
        with tempfile.NamedTemporaryFile('w') as output_file:
            b = self.batch()
            j = b.new_job()
            j.env('SOME_VARIABLE', '123abcdef')
            j.command(f'echo $SOME_VARIABLE > {j.ofile}')
            b.write_output(j.ofile, output_file.name)
            b.run()
            assert self.read(output_file.name) == '123abcdef'

    def test_concatenate(self):
        b = self.batch()
        files = []
        for i in range(10):
            j = b.new_job()
            j.command(f'touch {j.ofile}')
            files.append(j.ofile)
        concatenate(b, files, branching_factor=2)
        assert len(b._jobs) == 10 + (5 + 3 + 2 + 1)
        b.run()

    def test_python_job(self):
        with tempfile.NamedTemporaryFile('w') as output_file:
            b = self.batch()
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
            res = b.run()
            assert self.read(output_file.name) == '3\n5\n30\n{\"x\": 3, \"y\": 5}'

    def test_backend_context_manager(self):
        with LocalBackend() as backend:
            b = Batch(backend=backend)
            b.run()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.backend = ServiceBackend()

        remote_tmpdir = get_user_config().get('batch', 'remote_tmpdir')
        if not remote_tmpdir.endswith('/'):
            remote_tmpdir += '/'
        self.remote_tmpdir = remote_tmpdir

        if remote_tmpdir.startswith('gs://'):
            self.bucket_name = re.fullmatch('gs://(?P<bucket_name>[^/]+).*', remote_tmpdir).groupdict()['bucket_name']
        else:
            self.bucket_name = None

        self.cloud_input_dir = f'{self.remote_tmpdir}batch-tests/resources'

        token = uuid.uuid4()
        self.cloud_output_path = f'/batch-tests/{token}'
        self.cloud_output_dir = f'{self.remote_tmpdir}{self.cloud_output_path}'

        in_cluster_key_file = '/test-gsa-key/key.json'
        if not os.path.exists(in_cluster_key_file):
            in_cluster_key_file = None

        router_fs = RouterAsyncFS('gs',
                                  gcs_kwargs={'project': 'hail-vdc', 'credentials_file': in_cluster_key_file},
                                  azure_kwargs={'credential_file': in_cluster_key_file})

        def sync_exists(url):
            return async_to_blocking(router_fs.exists(url))

        def sync_write(url, data):
            return async_to_blocking(router_fs.write(url, data))

        if not sync_exists(f'{self.remote_tmpdir}batch-tests/resources/hello.txt'):
            sync_write(f'{self.remote_tmpdir}batch-tests/resources/hello.txt', b'hello world')
        if not sync_exists(f'{self.remote_tmpdir}batch-tests/resources/hello spaces.txt'):
            sync_write(f'{self.remote_tmpdir}batch-tests/resources/hello spaces.txt', b'hello')
        if not sync_exists(f'{self.remote_tmpdir}batch-tests/resources/hello (foo) spaces.txt'):
            sync_write(f'{self.remote_tmpdir}batch-tests/resources/hello (foo) spaces.txt', b'hello')

    def tearDown(self):
        self.backend.close()

    def batch(self, requester_pays_project=None, default_python_image=None,
              cancel_after_n_failures=None):
        return Batch(backend=self.backend,
                     default_image=DOCKER_ROOT_IMAGE,
                     attributes={'foo': 'a', 'bar': 'b'},
                     requester_pays_project=requester_pays_project,
                     default_python_image=default_python_image,
                     cancel_after_n_failures=cancel_after_n_failures)

    def test_single_task_no_io(self):
        b = self.batch()
        j = b.new_job()
        j.command('echo hello')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_single_task_input(self):
        b = self.batch()
        input = b.read_input(f'{self.cloud_input_dir}/hello.txt')
        j = b.new_job()
        j.command(f'cat {input}')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_single_task_input_resource_group(self):
        b = self.batch()
        input = b.read_input_group(foo=f'{self.cloud_input_dir}/hello.txt')
        j = b.new_job()
        j.storage('10Gi')
        j.command(f'cat {input.foo}')
        j.command(f'cat {input}.foo')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_single_task_output(self):
        b = self.batch()
        j = b.new_job(attributes={'a': 'bar', 'b': 'foo'})
        j.command(f'echo hello > {j.ofile}')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_single_task_write_output(self):
        b = self.batch()
        j = b.new_job()
        j.command(f'echo hello > {j.ofile}')
        b.write_output(j.ofile, f'{self.cloud_output_dir}/test_single_task_output.txt')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_single_task_resource_group(self):
        b = self.batch()
        j = b.new_job()
        j.declare_resource_group(output={'foo': '{root}.foo'})
        j.command(f'echo "hello" > {j.output.foo}')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_single_task_write_resource_group(self):
        b = self.batch()
        j = b.new_job()
        j.declare_resource_group(output={'foo': '{root}.foo'})
        j.command(f'echo "hello" > {j.output.foo}')
        b.write_output(j.output, f'{self.cloud_output_dir}/test_single_task_write_resource_group')
        b.write_output(j.output.foo, f'{self.cloud_output_dir}/test_single_task_write_resource_group_file.txt')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_multiple_dependent_tasks(self):
        output_file = f'{self.cloud_output_dir}/test_multiple_dependent_tasks.txt'
        b = self.batch()
        j = b.new_job()
        j.command(f'echo "0" >> {j.ofile}')

        for i in range(1, 3):
            j2 = b.new_job()
            j2.command(f'echo "{i}" > {j2.tmp1}')
            j2.command(f'cat {j.ofile} {j2.tmp1} > {j2.ofile}')
            j = j2

        b.write_output(j.ofile, output_file)
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_specify_cpu(self):
        b = self.batch()
        j = b.new_job()
        j.cpu('0.5')
        j.command(f'echo "hello" > {j.ofile}')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_specify_memory(self):
        b = self.batch()
        j = b.new_job()
        j.memory('100M')
        j.command(f'echo "hello" > {j.ofile}')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_scatter_gather(self):
        b = self.batch()

        for i in range(3):
            j = b.new_job(name=f'foo{i}')
            j.command(f'echo "{i}" > {j.ofile}')

        merger = b.new_job()
        merger.command('cat {files} > {ofile}'.format(files=' '.join([j.ofile for j in sorted(b.select_jobs('foo'),
                                                                                              key=lambda x: x.name,
                                                                                              reverse=True)]),
                                                      ofile=merger.ofile))

        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_file_name_space(self):
        b = self.batch()
        input = b.read_input(f'{self.cloud_input_dir}/hello (foo) spaces.txt')
        j = b.new_job()
        j.command(f'cat {input} > {j.ofile}')
        b.write_output(j.ofile, f'{self.cloud_output_dir}/hello (foo) spaces.txt')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_dry_run(self):
        b = self.batch()
        j = b.new_job()
        j.command(f'echo hello > {j.ofile}')
        b.write_output(j.ofile, f'{self.cloud_output_dir}/test_single_job_output.txt')
        b.run(dry_run=True)

    def test_verbose(self):
        b = self.batch()
        input = b.read_input(f'{self.cloud_input_dir}/hello.txt')
        j = b.new_job()
        j.command(f'cat {input}')
        b.write_output(input, f'{self.cloud_output_dir}/hello.txt')
        res = b.run(verbose=True)
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    @fails_in_azure
    def test_gcsfuse(self):
        assert self.bucket_name
        path = f'/{self.bucket_name}{self.cloud_output_path}'

        b = self.batch()
        head = b.new_job()
        head.command(f'mkdir -p {path}; echo head > {path}/gcsfuse_test_1')
        head.gcsfuse(self.bucket_name, f'/{self.bucket_name}', read_only=False)

        tail = b.new_job()
        tail.command(f'cat {path}/gcsfuse_test_1')
        tail.gcsfuse(self.bucket_name, f'/{self.bucket_name}', read_only=True)
        tail.depends_on(head)

        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    @fails_in_azure
    def test_gcsfuse_read_only(self):
        assert self.bucket_name
        path = f'/{self.bucket_name}{self.cloud_output_path}'

        b = self.batch()
        j = b.new_job()
        j.command(f'mkdir -p {path}; echo head > {path}/gcsfuse_test_1')
        j.gcsfuse(self.bucket_name, f'/{self.bucket_name}', read_only=True)

        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'failure', str((res_status, res.debug_info()))

    @fails_in_azure
    def test_gcsfuse_implicit_dirs(self):
        assert self.bucket_name
        path = f'/{self.bucket_name}{self.cloud_output_path}'

        b = self.batch()
        head = b.new_job()
        head.command(f'mkdir -p {path}/gcsfuse/; echo head > {path}/gcsfuse/data')
        head.gcsfuse(self.bucket_name, f'/{self.bucket_name}', read_only=False)

        tail = b.new_job()
        tail.command(f'cat {path}/gcsfuse/data')
        tail.gcsfuse(self.bucket_name, f'/{self.bucket_name}', read_only=True)
        tail.depends_on(head)

        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    @fails_in_azure
    def test_gcsfuse_empty_string_bucket_fails(self):
        assert self.bucket_name
        b = self.batch()
        j = b.new_job()
        with self.assertRaises(BatchException):
            j.gcsfuse('', '/empty_bucket')
        with self.assertRaises(BatchException):
            j.gcsfuse(self.bucket_name, '')

    @skip_in_azure
    def test_requester_pays(self):
        b = self.batch(requester_pays_project='hail-vdc')
        input = b.read_input('gs://hail-services-requester-pays/hello')
        j = b.new_job()
        j.command(f'cat {input}')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_benchmark_lookalike_workflow(self):
        b = self.batch()

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

        combine = b.new_job(f'combine_output').cpu(0.25)
        for tasks in grouped(arg_max(), jobs):
            combine.command(f'cat {" ".join(shq(j.ofile) for j in jobs)} >> {combine.ofile}')
        b.write_output(combine.ofile, f'{self.cloud_output_dir}/pipeline_benchmark_test.txt')
        # too slow
        # assert b.run().status()['state'] == 'success'

    def test_envvar(self):
        b = self.batch()
        j = b.new_job()
        j.env('SOME_VARIABLE', '123abcdef')
        j.command('[ $SOME_VARIABLE = "123abcdef" ]')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_single_job_with_shell(self):
        msg = 'hello world'
        b = self.batch()
        j = b.new_job(shell='/bin/sh')
        j.command(f'echo "{msg}"')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_single_job_with_nonsense_shell(self):
        b = self.batch()
        j = b.new_job(shell='/bin/ajdsfoijasidojf')
        j.command(f'echo "hello"')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'failure', str((res_status, res.debug_info()))

    def test_single_job_with_intermediate_failure(self):
        b = self.batch()
        j = b.new_job()
        j.command(f'echoddd "hello"')
        j2 = b.new_job()
        j2.command(f'echo "world"')

        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'failure', str((res_status, res.debug_info()))

    def test_input_directory(self):
        b = self.batch()
        input1 = b.read_input(self.cloud_input_dir)
        input2 = b.read_input(self.cloud_input_dir.rstrip('/') + '/')
        j = b.new_job()
        j.command(f'ls {input1}/hello.txt')
        j.command(f'ls {input2}/hello.txt')
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))

    def test_python_job(self):
        b = self.batch(default_python_image=PYTHON_DILL_IMAGE)
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
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))
        assert res.get_job_log(4)['main'] == "3\n5\n30\n{\"x\": 3, \"y\": 5}\n", str(res.debug_info())

    def test_python_job_w_resource_group_unpack_individually(self):
        b = self.batch(default_python_image=PYTHON_DILL_IMAGE)
        head = b.new_job()
        head.declare_resource_group(count={'r5': '{root}.r5',
                                           'r3': '{root}.r3'})

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
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))
        assert res.get_job_log(4)['main'] == "3\n5\n30\n{\"x\": 3, \"y\": 5}\n", str(res.debug_info())

    def test_python_job_w_resource_group_unpack_jointly(self):
        b = self.batch(default_python_image=PYTHON_DILL_IMAGE)
        head = b.new_job()
        head.declare_resource_group(count={'r5': '{root}.r5',
                                           'r3': '{root}.r3'})

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
        res_status = res.status()
        assert res_status['state'] == 'success', str((res_status, res.debug_info()))
        job_log_3 = res.get_job_log(3)
        assert job_log_3['main'] == "15\n", str((job_log_3, res.debug_info()))

    def test_python_job_w_non_zero_ec(self):
        b = self.batch(default_python_image=PYTHON_DILL_IMAGE)
        j = b.new_python_job()

        def error():
            raise Exception("this should fail")

        j.call(error)
        res = b.run()
        res_status = res.status()
        assert res_status['state'] == 'failure', str((res_status, res.debug_info()))

    def test_fail_fast(self):
        b = self.batch(cancel_after_n_failures=1)

        j1 = b.new_job()
        j1.command('false')

        j2 = b.new_job()
        j2.command('sleep 300')

        res = b.run()
        job_status = res.get_job(2).status()
        assert job_status['state'] == 'Cancelled', str((job_status, res.debug_info()))

    def test_service_backend_remote_tempdir_with_trailing_slash(self):
        backend = ServiceBackend(remote_tmpdir=f'{self.remote_tmpdir}/temporary-files/')
        b = Batch(backend=backend)
        j1 = b.new_job()
        j1.command(f'echo hello > {j1.ofile}')
        j2 = b.new_job()
        j2.command(f'cat {j1.ofile}')
        b.run()

    def test_service_backend_remote_tempdir_with_no_trailing_slash(self):
        backend = ServiceBackend(remote_tmpdir=f'{self.remote_tmpdir}/temporary-files')
        b = Batch(backend=backend)
        j1 = b.new_job()
        j1.command(f'echo hello > {j1.ofile}')
        j2 = b.new_job()
        j2.command(f'cat {j1.ofile}')
        b.run()

    def test_large_command(self):
        backend = ServiceBackend(remote_tmpdir=f'{self.remote_tmpdir}/temporary-files')
        b = Batch(backend=backend)
        j1 = b.new_job()
        long_str = secrets.token_urlsafe(15 * 1024)
        j1.command(f'echo "{long_str}"')
        b.run()
