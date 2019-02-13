import unittest
import os
import subprocess as sp

from pipeline import Pipeline, BatchBackend


gcs_input_dir = 'gs://hail-pipeline-test/data/'
gcs_output_dir = 'gs://hail-pipeline-test/output/'


class LocalTests(unittest.TestCase):
    def read(self, file):
        with open(file, 'r') as f:
            result = f.read().rstrip()
        return result

    def write(self, file, msg):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w') as f:
            f.write(msg)
            f.close()

    def rm(self, *files):
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
            else:
                raise ValueError(f"file '{file}' does not exist.")

    def assert_same_file(self, file1, file2):
        self.assertEqual(self.read(file1).rstrip(), self.read(file2).rstrip())

    def test_read_input_and_write_output(self):
        input_file = '/tmp/data/example1.txt'
        output_file = '/tmp/example1.txt'
        self.write(input_file, 'abc')

        p = Pipeline()
        input = p.read_input(input_file)
        p.write_output(input, output_file)
        p.run()

        self.assert_same_file(input_file, output_file)
        self.rm(input_file, output_file)

    def test_read_input_group(self):
        input_file1 = '/tmp/data/example1.txt'
        input_file2 = '/tmp/data/example2.txt'
        output_file1 = '/tmp/example1.txt'
        output_file2 = '/tmp/example2.txt'
        self.write(input_file1, 'abc')
        self.write(input_file2, '123')

        p = Pipeline()
        input = p.read_input_group(in1=input_file1,
                                   in2=input_file2)

        p.write_output(input.in1, output_file1)
        p.write_output(input.in2, output_file2)
        p.run()

        self.assert_same_file(input_file1, output_file1)
        self.assert_same_file(input_file2, output_file2)
        self.rm(input_file1, input_file2, output_file1, output_file2)

    def test_single_task(self):
        output_file = '/tmp/test_single_task.txt'
        msg = 'hello world'

        p = Pipeline()
        t = p.new_task()
        t.command(f'echo "{msg}" > {t.ofile}')
        p.write_output(t.ofile, output_file)
        p.run()

        self.assertEqual(self.read(output_file), msg)
        self.rm(output_file)

    def test_single_task_w_input(self):
        msg = 'abc'
        input_file = '/tmp/data/example1.txt'
        output_file = '/tmp/test_single_task_w_input.txt'
        self.write(input_file, 'abc')

        p = Pipeline()
        input = p.read_input(input_file)
        t = p.new_task()
        t.command(f'cat {input} > {t.ofile}')
        p.write_output(t.ofile, output_file)
        p.run()

        self.assertEqual(self.read(output_file), msg)
        self.rm(input_file, output_file)

    def test_single_task_w_input_group(self):
        msg1 = 'abc'
        msg2 = '123'
        input_file1 = '/tmp/data/example1.txt'
        input_file2 = '/tmp/data/example2.txt'
        output_file = '/tmp/test_single_task_w_input_group.txt'
        self.write(input_file1, 'abc')
        self.write(input_file2, '123')

        p = Pipeline()
        input = p.read_input_group(in1=input_file1, in2=input_file2)
        t = p.new_task()
        t.command(f'cat {input.in1} {input.in2} > {t.ofile}')
        p.write_output(t.ofile, output_file)
        p.run()

        self.assertEqual(self.read(output_file), msg1 + msg2)
        self.rm(input_file1, input_file2, output_file)

    def test_single_task_bad_command(self):
        p = Pipeline()
        t = p.new_task()
        t.command("foo") # this should fail!
        with self.assertRaises(sp.CalledProcessError):
            p.run()

    def test_declare_resource_group(self):
        output_file = '/tmp/test_declare_resource_group.txt'
        msg = 'hello world'

        p = Pipeline()
        t = p.new_task()
        t.declare_resource_group(ofile={'log': "{root}.txt"})
        t.command(f'echo "{msg}" > {t.ofile.log}')
        p.write_output(t.ofile.log, output_file)
        p.run()

        self.assertEqual(self.read(output_file), msg)
        self.rm(output_file)

    def test_multiple_isolated_tasks(self):
        p = Pipeline()

        for i in range(5):
            output_file = f'/tmp/test_multiple_isolated_tasks_{i}.txt'
            msg = f'hello world {i}'
            t = p.new_task()
            t.command(f'echo "{msg}" > {t.ofile}')
            p.write_output(t.ofile, output_file)

        p.run()

        for i in range(5):
            output_file = f'/tmp/test_multiple_isolated_tasks_{i}.txt'
            msg = f'hello world {i}'
            self.assertEqual(self.read(output_file), msg)
            self.rm(output_file)

    def test_multiple_dependent_tasks(self):
        output_file = '/tmp/test_multiple_dependent_tasks.txt'
        p = Pipeline()
        t = p.new_task()
        t.command(f'echo "0" >> {t.ofile}')

        for i in range(1, 3):
            t2 = p.new_task()
            t2.command(f'echo "{i}" > {t2.tmp1}')
            t2.command(f'cat {t.ofile} {t2.tmp1} > {t2.ofile}')
            t = t2

        p.write_output(t.ofile, output_file)
        p.run()

        self.assertEqual(self.read(output_file), "0\n1\n2")
        self.rm(output_file)

    def test_select_tasks(self):
        p = Pipeline()

        for i in range(3):
            t = p.new_task().label(f'foo{i}')

        self.assertTrue(len(p.select_tasks('foo')) == 3)

    def test_scatter_gather(self):
        output_file = '/tmp/test_scatter_gather.txt'
        p = Pipeline()

        for i in range(3):
            t = p.new_task().label(f'foo{i}')
            t.command(f'echo "{i}" > {t.ofile}')

        merger = p.new_task()
        merger.command('cat {files} > {ofile}'.format(files=' '.join([t.ofile for t in sorted(p.select_tasks('foo'),
                                                                                              key=lambda x: x._label,
                                                                                              reverse=True)]),
                                                      ofile=merger.ofile))

        p.write_output(merger.ofile, output_file)
        p.run()

        self.assertEqual(self.read(output_file), '2\n1\n0')
        self.rm(output_file)

    # FIXME: Run docker inside docker
    # def test_single_task_docker(self):
    #     output_file = '/tmp/test_single_task_docker.txt'
    #     msg = 'hello world'
    #
    #     p = Pipeline()
    #     t = p.new_task()
    #     t.docker('ubuntu')
    #     t.command(f'echo "{msg}" > {t.ofile}')
    #     p.write_output(t.ofile, output_file)
    #     p.run()
    #
    #     self.assertEqual(self.read(output_file), msg)
    #     self.rm(output_file)


class BatchTests(unittest.TestCase):
    def pipeline(self):
        url = os.environ.get('BATCH_URL')
        if url is None:
            raise Exception("BATCH_URL not defined in the environment.")
        return Pipeline(backend=BatchBackend(url),
                        default_image='google/cloud-sdk:alpine')

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
        t.command(f'cat {input.foo}')
        p.run()

    def test_single_task_output(self):
        p = self.pipeline()
        t = p.new_task()
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
