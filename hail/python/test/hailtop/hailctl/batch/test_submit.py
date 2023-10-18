import os
import pytest
import tempfile

from typer.testing import CliRunner

from hailtop.hailctl.batch import cli


@pytest.fixture
def runner():
    yield CliRunner(mix_stderr=False)


def write_script(dir: str, filename: str):
    with open(f'{dir}/test_job.py', 'w') as f:
        f.write(f'''
import hailtop.batch as hb
b = hb.Batch()
j = b.new_job()
j.command('cat {filename}')
b.run(wait=False)
backend.close()
''')


def write_hello(filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('hello\n')


def test_file_in_current_dir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        write_hello(f'{dir}/hello.txt')
        write_script(dir, '/hello.txt')
        res = runner.invoke(cli.app, ['submit', '--mounts', 'hello.txt:/', 'test_job.py'])
        assert res.exit_code == 0


def test_file_mount_in_child_dir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        write_hello(f'{dir}/hello.txt')
        write_script(dir, '/child/hello.txt')
        res = runner.invoke(cli.app, ['submit', '--mounts', 'hello.txt:/child/', 'test_job.py'])
        assert res.exit_code == 0


def test_file_mount_in_child_dir_to_root_dir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        write_hello(f'{dir}/child/hello.txt')
        write_script(dir, '/hello.txt')
        res = runner.invoke(cli.app, ['submit', '--mounts', 'child/hello.txt:/', 'test_job.py'])
        assert res.exit_code == 0


def test_mount_multiple_files(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        write_hello(f'{dir}/child/hello1.txt')
        write_hello(f'{dir}/child/hello2.txt')
        write_script(dir, '/hello1.txt')
        res = runner.invoke(cli.app, ['submit', '--mounts', 'child/hello1.txt:/', '--mounts', 'child/hello2.txt:/', 'test_job.py'])
        assert res.exit_code == 0


def test_dir_mount_in_child_dir_to_child_dir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        write_hello(f'{dir}/child/hello1.txt')
        write_hello(f'{dir}/child/hello2.txt')
        write_script(dir, '/child/hello1.txt')
        res = runner.invoke(cli.app, ['submit', '--mounts', 'child/:/child/', 'test_job.py'])
        assert res.exit_code == 0


def test_file_outside_curdir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.mkdir(f'{dir}/working_dir')
        os.chdir(f'{dir}/working_dir')
        write_hello(f'{dir}/hello.txt')
        write_script(dir, '/hello.txt')
        res = runner.invoke(cli.app, ['submit', '--mounts', f'{dir}/hello.txt:/', '../test_job.py'])
        assert res.exit_code == 0


def test_dir_outside_curdir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.mkdir(f'{dir}/working_dir')
        os.chdir(f'{dir}/working_dir')
        write_hello(f'{dir}/hello1.txt')
        write_hello(f'{dir}/hello2.txt')
        write_script(dir, '/hello1.txt')
        res = runner.invoke(cli.app, ['submit', '--mounts', f'{dir}/:/', '../test_job.py'])
        assert res.exit_code == 0


def test_dir_outside_curdir_fails_with_files(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.mkdir(f'{dir}/working_dir')
        os.chdir(f'{dir}/working_dir')
        write_hello(f'{dir}/hello1.txt')
        write_hello(f'{dir}/hello2.txt')
        write_script(dir, '/hello1.txt')
        res = runner.invoke(cli.app, ['submit', '--files', dir, '../test_job.py'])
        assert res.exit_code == 1
