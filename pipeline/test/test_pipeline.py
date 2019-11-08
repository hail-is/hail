import unittest
import os
import subprocess as sp
import tempfile

from hailtop.pipeline import Pipeline, BatchBackend, LocalBackend, PipelineException

gcs_input_dir = os.environ.get('SCRATCH') + '/input'
gcs_output_dir = os.environ.get('SCRATCH') + '/output'


class LocalTests(unittest.TestCase):
    def pipeline(self):
        return Pipeline(backend=LocalBackend())

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

            p = self.pipeline()
            input = p.read_input(input_file.name)
            p.write_output(input, output_file.name)
            p.run()

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

            p = self.pipeline()
            input = p.read_input_group(in1=input_file1.name,
                                       in2=input_file2.name)

            p.write_output(input.in1, output_file1.name)
            p.write_output(input.in2, output_file2.name)
            p.run()

            self.assert_same_file(input_file1.name, output_file1.name)
            self.assert_same_file(input_file2.name, output_file2.name)

    def test_write_resource_group(self):
        with tempfile.NamedTemporaryFile('w') as input_file1, \
                tempfile.NamedTemporaryFile('w') as input_file2, \
                tempfile.TemporaryDirectory() as output_dir:

            p = self.pipeline()
            input = p.read_input_group(in1=input_file1.name,
                                       in2=input_file2.name)

            p.write_output(input, output_dir + '/foo')
            p.run()

            self.assert_same_file(input_file1.name, output_dir + '/foo.in1')
            self.assert_same_file(input_file2.name, output_dir + '/foo.in2')

    def test_single_task(self):
        with tempfile.NamedTemporaryFile('w') as output_file:
            msg = 'hello world'

            p = self.pipeline()
            t = p.new_task()
            t.command(f'echo "{msg}" > {t.ofile}')
            p.write_output(t.ofile, output_file.name)
            p.run()

            assert self.read(output_file.name) == msg

    def test_single_task_w_input(self):
        with tempfile.NamedTemporaryFile('w') as input_file, \
                tempfile.NamedTemporaryFile('w') as output_file:
            msg = 'abc'
            input_file.write(msg)
            input_file.flush()

            p = self.pipeline()
            input = p.read_input(input_file.name)
            t = p.new_task()
            t.command(f'cat {input} > {t.ofile}')
            p.write_output(t.ofile, output_file.name)
            p.run()

            assert self.read(output_file.name) == msg

    def test_single_task_w_input_group(self):
        with tempfile.NamedTemporaryFile('w') as input_file1, \
                tempfile.NamedTemporaryFile('w') as input_file2, \
                tempfile.NamedTemporaryFile('w') as output_file:
            msg1 = 'abc'
            msg2 = '123'

            input_file1.write(msg1)
            input_file2.write(msg2)
            input_file1.flush()
            input_file2.flush()

            p = self.pipeline()
            input = p.read_input_group(in1=input_file1.name,
                                       in2=input_file2.name)
            t = p.new_task()
            t.command(f'cat {input.in1} {input.in2} > {t.ofile}')
            p.write_output(t.ofile, output_file.name)
            p.run()

            assert self.read(output_file.name) == msg1 + msg2

    def test_single_task_bad_command(self):
        p = self.pipeline()
        t = p.new_task()
        t.command("foo")  # this should fail!
        with self.assertRaises(sp.CalledProcessError):
            p.run()

    def test_declare_resource_group(self):
        with tempfile.NamedTemporaryFile('w') as output_file:
            msg = 'hello world'
            p = self.pipeline()
            t = p.new_task()
            t.declare_resource_group(ofile={'log': "{root}.txt"})
            t.command(f'echo "{msg}" > {t.ofile.log}')
            p.write_output(t.ofile.log, output_file.name)
            p.run()

            assert self.read(output_file.name) == msg

    def test_resource_group_get_all_inputs(self):
        p = self.pipeline()
        input = p.read_input_group(fasta="foo",
                                   idx="bar")
        t = p.new_task()
        t.command(f"cat {input.fasta}")
        assert input.fasta in t._inputs
        assert input.idx in t._inputs

    def test_resource_group_get_all_mentioned(self):
        p = self.pipeline()
        t = p.new_task()
        t.declare_resource_group(foo={'bed': '{root}.bed', 'bim': '{root}.bim'})
        t.command(f"cat {t.foo.bed}")
        assert t.foo.bed in t._mentioned
        assert t.foo.bim not in t._mentioned

    def test_resource_group_get_all_mentioned_dependent_tasks(self):
        p = self.pipeline()
        t = p.new_task()
        t.declare_resource_group(foo={'bed': '{root}.bed', 'bim': '{root}.bim'})
        t.command(f"cat")
        t2 = p.new_task()
        t2.command(f"cat {t.foo}")

    def test_resource_group_get_all_outputs(self):
        p = self.pipeline()
        t1 = p.new_task()
        t1.declare_resource_group(foo={'bed': '{root}.bed', 'bim': '{root}.bim'})
        t1.command(f"cat {t1.foo.bed}")
        t2 = p.new_task()
        t2.command(f"cat {t1.foo.bed}")

        for r in [t1.foo.bed, t1.foo.bim]:
            assert r in t1._internal_outputs
            assert r in t2._inputs

        assert t1.foo.bed in t1._mentioned
        assert t1.foo.bim not in t1._mentioned

        assert t1.foo.bed in t2._mentioned
        assert t1.foo.bim not in t2._mentioned

        assert t1.foo not in t1._mentioned

    def test_multiple_isolated_tasks(self):
        p = self.pipeline()

        output_files = []
        try:
            output_files = [tempfile.NamedTemporaryFile('w') for i in range(5)]

            for i, ofile in enumerate(output_files):
                msg = f'hello world {i}'
                t = p.new_task()
                t.command(f'echo "{msg}" > {t.ofile}')
                p.write_output(t.ofile, ofile.name)
            p.run()

            for i, ofile in enumerate(output_files):
                msg = f'hello world {i}'
                assert self.read(ofile.name) == msg
        finally:
            [ofile.close() for ofile in output_files]

    def test_multiple_dependent_tasks(self):
        with tempfile.NamedTemporaryFile('w') as output_file:
            p = self.pipeline()
            t = p.new_task()
            t.command(f'echo "0" >> {t.ofile}')

            for i in range(1, 3):
                t2 = p.new_task()
                t2.command(f'echo "{i}" > {t2.tmp1}')
                t2.command(f'cat {t.ofile} {t2.tmp1} > {t2.ofile}')
                t = t2

            p.write_output(t.ofile, output_file.name)
            p.run()

            assert self.read(output_file.name) == "0\n1\n2"

    def test_select_tasks(self):
        p = self.pipeline()
        for i in range(3):
            t = p.new_task(name=f'foo{i}')
        self.assertTrue(len(p.select_tasks('foo')) == 3)

    def test_scatter_gather(self):
        with tempfile.NamedTemporaryFile('w') as output_file:
            p = self.pipeline()

            for i in range(3):
                t = p.new_task(name=f'foo{i}')
                t.command(f'echo "{i}" > {t.ofile}')

            merger = p.new_task()
            merger.command('cat {files} > {ofile}'.format(files=' '.join([t.ofile for t in sorted(p.select_tasks('foo'),
                                                                                                  key=lambda x: x.name,
                                                                                                  reverse=True)]),
                                                          ofile=merger.ofile))

            p.write_output(merger.ofile, output_file.name)
            p.run()

            assert self.read(output_file.name) == '2\n1\n0'

    def test_add_extension_task_resource_file(self):
        p = self.pipeline()
        t = p.new_task()
        t.command(f'echo "hello" > {t.ofile}')
        t.ofile.add_extension('.txt.bgz')
        assert t.ofile._value.endswith('.txt.bgz')

    def test_add_extension_input_resource_file(self):
        input_file1 = '/tmp/data/example1.txt.bgz.foo'
        p = self.pipeline()
        in1 = p.read_input(input_file1, extension='.txt.bgz.foo')
        with self.assertRaises(Exception):
            in1.add_extension('.baz')
        assert in1._value.endswith('.txt.bgz.foo')

    def test_file_name_space(self):
        with tempfile.NamedTemporaryFile('w', prefix="some file name with (foo) spaces") as input_file, \
                tempfile.NamedTemporaryFile('w', prefix="another file name with (foo) spaces") as output_file:

            input_file.write('abc')
            input_file.flush()

            p = self.pipeline()
            input = p.read_input(input_file.name)
            t = p.new_task()
            t.command(f'cat {input} > {t.ofile}')
            p.write_output(t.ofile, output_file.name)
            p.run()

            self.assert_same_file(input_file.name, output_file.name)

    def test_resource_group_mentioned(self):
        p = self.pipeline()
        t = p.new_task()
        t.declare_resource_group(foo={'bed': '{root}.bed'})
        t.command(f'echo "hello" > {t.foo}')

        t2 = p.new_task()
        t2.command(f'echo "hello" >> {t.foo.bed}')
        p.run()


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.backend = BatchBackend()

    def tearDown(self):
        self.backend.close()

    def pipeline(self):
        return Pipeline(backend=self.backend,
                        default_image='google/cloud-sdk:237.0.0-alpine',
                        attributes={'foo': 'a', 'bar': 'b'})

    def test_single_task_no_io(self):
        p = self.pipeline()
        t = p.new_task()
        t.command('echo hello')
        p.run()

    def test_single_task_input(self):
        p = self.pipeline()
        input = p.read_input(f'{gcs_input_dir}/hello.txt')
        t = p.new_task()
        t.command(f'cat {input}')
        p.run()

    def test_single_task_input_resource_group(self):
        p = self.pipeline()
        input = p.read_input_group(foo=f'{gcs_input_dir}/hello.txt')
        t = p.new_task()
        t.storage('0.25Gi')
        t.command(f'cat {input.foo}')
        p.run()

    def test_single_task_output(self):
        p = self.pipeline()
        t = p.new_task(attributes={'a': 'bar', 'b': 'foo'})
        t.command(f'echo hello > {t.ofile}')
        p.run()

    def test_single_task_write_output(self):
        p = self.pipeline()
        t = p.new_task()
        t.command(f'echo hello > {t.ofile}')
        p.write_output(t.ofile, f'{gcs_output_dir}/test_single_task_output.txt')
        p.run()

    def test_single_task_resource_group(self):
        p = self.pipeline()
        t = p.new_task()
        t.declare_resource_group(output={'foo': '{root}.foo'})
        t.command(f'echo "hello" > {t.output.foo}')
        p.run()

    def test_single_task_write_resource_group(self):
        p = self.pipeline()
        t = p.new_task()
        t.declare_resource_group(output={'foo': '{root}.foo'})
        t.command(f'echo "hello" > {t.output.foo}')
        p.write_output(t.output, f'{gcs_output_dir}/test_single_task_write_resource_group')
        p.write_output(t.output.foo, f'{gcs_output_dir}/test_single_task_write_resource_group_file.txt')
        p.run()

    def test_multiple_dependent_tasks(self):
        output_file = f'{gcs_output_dir}/test_multiple_dependent_tasks.txt'
        p = self.pipeline()
        t = p.new_task()
        t.command(f'echo "0" >> {t.ofile}')

        for i in range(1, 3):
            t2 = p.new_task()
            t2.command(f'echo "{i}" > {t2.tmp1}')
            t2.command(f'cat {t.ofile} {t2.tmp1} > {t2.ofile}')
            t = t2

        p.write_output(t.ofile, output_file)
        p.run()

    def test_specify_cpu(self):
        p = self.pipeline()
        t = p.new_task()
        t.cpu('0.5')
        t.command(f'echo "hello" > {t.ofile}')
        p.run()

    def test_specify_memory(self):
        p = self.pipeline()
        t = p.new_task()
        t.memory('100M')
        t.command(f'echo "hello" > {t.ofile}')
        p.run()

    def test_scatter_gather(self):
        p = self.pipeline()

        for i in range(3):
            t = p.new_task(name=f'foo{i}')
            t.command(f'echo "{i}" > {t.ofile}')

        merger = p.new_task()
        merger.command('cat {files} > {ofile}'.format(files=' '.join([t.ofile for t in sorted(p.select_tasks('foo'),
                                                                                              key=lambda x: x.name,
                                                                                              reverse=True)]),
                                                      ofile=merger.ofile))

        p.run()

    def test_file_name_space(self):
        p = self.pipeline()
        input = p.read_input(f'{gcs_input_dir}/hello (foo) spaces.txt')
        t = p.new_task()
        t.command(f'cat {input} > {t.ofile}')
        p.write_output(t.ofile, f'{gcs_output_dir}/hello (foo) spaces.txt')
        p.run()

    def test_dry_run(self):
        p = self.pipeline()
        t = p.new_task()
        t.command(f'echo hello > {t.ofile}')
        p.write_output(t.ofile, f'{gcs_output_dir}/test_single_task_output.txt')
        p.run(dry_run=True)

    def test_verbose(self):
        p = self.pipeline()
        input = p.read_input(f'{gcs_input_dir}/hello.txt')
        t = p.new_task()
        t.command(f'cat {input}')
        p.write_output(input, f'{gcs_output_dir}/hello.txt')
        p.run(verbose=True)

    def test_failed_job_error_msg(self):
        with self.assertRaises(PipelineException):
            p = self.pipeline()
            t = p.new_task()
            t.command('false')
            p.run()
