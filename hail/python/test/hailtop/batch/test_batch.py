import unittest
import os
import subprocess as sp
import tempfile
from shlex import quote as shq
import uuid
import google.oauth2.service_account
import google.cloud.storage

from hailtop.batch import Batch, ServiceBackend, LocalBackend
from hailtop.batch.utils import arg_max
from hailtop.utils import grouped
from hailtop.config import get_user_config


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


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.backend = ServiceBackend()

        self.bucket_name = get_user_config().get('batch', 'bucket')

        self.gcs_input_dir = f'gs://{self.bucket_name}/batch-tests/resources'

        token = uuid.uuid4()
        self.gcs_output_path = f'/batch-tests/{token}'
        self.gcs_output_dir = f'gs://{self.bucket_name}{self.gcs_output_path}'

        in_cluster_key_file = '/test-gsa-key/key.json'
        if os.path.exists(in_cluster_key_file):
            credentials = google.oauth2.service_account.Credentials.from_service_account_file(
                in_cluster_key_file)
        else:
            credentials = None
        gcs_client = google.cloud.storage.Client(project='hail-vdc', credentials=credentials)
        bucket = gcs_client.bucket(self.bucket_name)
        if not bucket.blob('batch-tests/resources/hello.txt').exists():
            bucket.blob('batch-tests/resources/hello.txt').upload_from_string(
                'hello world')
        if not bucket.blob('batch-tests/resources/hello spaces.txt').exists():
            bucket.blob('batch-tests/resources/hello spaces.txt').upload_from_string(
                'hello')
        if not bucket.blob('batch-tests/resources/hello (foo) spaces.txt').exists():
            bucket.blob('batch-tests/resources/hello (foo) spaces.txt').upload_from_string(
                'hello')

    def tearDown(self):
        self.backend.close()

    def batch(self, requester_pays_project=None):
        return Batch(backend=self.backend,
                     default_image='google/cloud-sdk:237.0.0-alpine',
                     attributes={'foo': 'a', 'bar': 'b'},
                     requester_pays_project=requester_pays_project)

    def test_single_task_no_io(self):
        b = self.batch()
        j = b.new_job()
        j.command('echo hello')
        assert b.run().status()['state'] == 'success'

    def test_single_task_input(self):
        b = self.batch()
        input = b.read_input(f'{self.gcs_input_dir}/hello.txt')
        j = b.new_job()
        j.command(f'cat {input}')
        assert b.run().status()['state'] == 'success'

    def test_single_task_input_resource_group(self):
        b = self.batch()
        input = b.read_input_group(foo=f'{self.gcs_input_dir}/hello.txt')
        j = b.new_job()
        j.storage('0.25Gi')
        j.command(f'cat {input.foo}')
        j.command(f'cat {input}.foo')
        assert b.run().status()['state'] == 'success'

    def test_single_task_output(self):
        b = self.batch()
        j = b.new_job(attributes={'a': 'bar', 'b': 'foo'})
        j.command(f'echo hello > {j.ofile}')
        assert b.run().status()['state'] == 'success'

    def test_single_task_write_output(self):
        b = self.batch()
        j = b.new_job()
        j.command(f'echo hello > {j.ofile}')
        b.write_output(j.ofile, f'{self.gcs_output_dir}/test_single_task_output.txt')
        assert b.run().status()['state'] == 'success'

    def test_single_task_resource_group(self):
        b = self.batch()
        j = b.new_job()
        j.declare_resource_group(output={'foo': '{root}.foo'})
        j.command(f'echo "hello" > {j.output.foo}')
        assert b.run().status()['state'] == 'success'

    def test_single_task_write_resource_group(self):
        b = self.batch()
        j = b.new_job()
        j.declare_resource_group(output={'foo': '{root}.foo'})
        j.command(f'echo "hello" > {j.output.foo}')
        b.write_output(j.output, f'{self.gcs_output_dir}/test_single_task_write_resource_group')
        b.write_output(j.output.foo, f'{self.gcs_output_dir}/test_single_task_write_resource_group_file.txt')
        assert b.run().status()['state'] == 'success'

    def test_multiple_dependent_tasks(self):
        output_file = f'{self.gcs_output_dir}/test_multiple_dependent_tasks.txt'
        b = self.batch()
        j = b.new_job()
        j.command(f'echo "0" >> {j.ofile}')

        for i in range(1, 3):
            j2 = b.new_job()
            j2.command(f'echo "{i}" > {j2.tmp1}')
            j2.command(f'cat {j.ofile} {j2.tmp1} > {j2.ofile}')
            j = j2

        b.write_output(j.ofile, output_file)
        assert b.run().status()['state'] == 'success'

    def test_specify_cpu(self):
        b = self.batch()
        j = b.new_job()
        j.cpu('0.5')
        j.command(f'echo "hello" > {j.ofile}')
        assert b.run().status()['state'] == 'success'

    def test_specify_memory(self):
        b = self.batch()
        j = b.new_job()
        j.memory('100M')
        j.command(f'echo "hello" > {j.ofile}')
        assert b.run().status()['state'] == 'success'

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

        assert b.run().status()['state'] == 'success'

    def test_file_name_space(self):
        b = self.batch()
        input = b.read_input(f'{self.gcs_input_dir}/hello (foo) spaces.txt')
        j = b.new_job()
        j.command(f'cat {input} > {j.ofile}')
        b.write_output(j.ofile, f'{self.gcs_output_dir}/hello (foo) spaces.txt')
        assert b.run().status()['state'] == 'success'

    def test_dry_run(self):
        b = self.batch()
        j = b.new_job()
        j.command(f'echo hello > {j.ofile}')
        b.write_output(j.ofile, f'{self.gcs_output_dir}/test_single_job_output.txt')
        b.run(dry_run=True)

    def test_verbose(self):
        b = self.batch()
        input = b.read_input(f'{self.gcs_input_dir}/hello.txt')
        j = b.new_job()
        j.command(f'cat {input}')
        b.write_output(input, f'{self.gcs_output_dir}/hello.txt')
        assert b.run(verbose=True).status()['state'] == 'success'

    def test_gcsfuse(self):
        path = f'/{self.bucket_name}{self.gcs_output_path}'

        b = self.batch()
        head = b.new_job()
        head.command(f'mkdir -p {path}; echo head > {path}/gcsfuse_test_1')
        head.gcsfuse(self.bucket_name, f'/{self.bucket_name}', read_only=False)

        tail = b.new_job()
        tail.command(f'cat {path}/gcsfuse_test_1')
        tail.gcsfuse(self.bucket_name, f'/{self.bucket_name}', read_only=True)
        tail.depends_on(head)

        assert b.run().status()['state'] == 'success'

    def test_gcsfuse_read_only(self):
        path = f'/{self.bucket_name}{self.gcs_output_path}'

        b = self.batch()
        j = b.new_job()
        j.command(f'mkdir -p {path}; echo head > {path}/gcsfuse_test_1')
        j.gcsfuse(self.bucket_name, f'/{self.bucket_name}', read_only=True)

        assert b.run().status()['state'] == 'failure'

    def test_requester_pays(self):
        b = self.batch(requester_pays_project='hail-vdc')
        input = b.read_input('gs://hail-services-requester-pays/hello')
        j = b.new_job()
        j.command(f'cat {input}')
        assert b.run().status()['state'] == 'success'

    def test_benchmark_lookalike_workflow(self):
        b = self.batch()

        setup_jobs = []
        for i in range(10):
            j = b.new_job(f'setup_{i}').cpu(0.1)
            j.command(f'echo "foo" > {j.ofile}')
            setup_jobs.append(j)

        jobs = []
        for i in range(500):
            j = b.new_job(f'create_file_{i}').cpu(0.1)
            j.command(f'echo {setup_jobs[i % len(setup_jobs)].ofile} > {j.ofile}')
            j.command(f'echo "bar" >> {j.ofile}')
            jobs.append(j)

        combine = b.new_job(f'combine_output').cpu(0.1)
        for tasks in grouped(arg_max(), jobs):
            combine.command(f'cat {" ".join(shq(j.ofile) for j in jobs)} >> {combine.ofile}')
        b.write_output(combine.ofile, f'{self.gcs_output_dir}/pipeline_benchmark_test.txt')
        # too slow
        # assert b.run().status()['state'] == 'success'

    def test_envvar(self):
        b = self.batch()
        j = b.new_job()
        j.env('SOME_VARIABLE', '123abcdef')
        j.command('[ $SOME_VARIABLE = "123abcdef" ]')
        assert b.run().status()['state'] == 'success'
