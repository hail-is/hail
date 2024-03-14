import os
import tempfile

import pytest
from typer.testing import CliRunner

from hailtop.hailctl.batch import cli
from hailtop.hailctl.batch.submit import real_absolute_expanded_path


@pytest.fixture
def runner():
    yield CliRunner(mix_stderr=False)


def write_script(dir: str, filename: str):
    with open(f'{dir}/test_job.py', 'w') as f:
        file, _ = real_absolute_expanded_path(filename)
        f.write(f"""
print(open("{file}").read())
""")


def write_hello(filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('hello\n')


@pytest.mark.timeout(5 * 60)  # image pulling is very slow
def test_file_with_no_dest(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        write_hello(f'{dir}/hello.txt')
        write_script(dir, f'{dir}/hello.txt')
        res = runner.invoke(
            cli.app, ['submit', '--wait', '--files', 'hello.txt', 'test_job.py'], catch_exceptions=False
        )
        assert res.exit_code == 0, repr((res.output, res.stdout, res.stderr, res.exception))


@pytest.mark.timeout(5 * 60)  # image pulling is very slow
def test_file_in_current_dir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        write_hello(f'{dir}/hello.txt')
        write_script(dir, f'/hello.txt')
        res = runner.invoke(
            cli.app, ['submit', '--wait', '--files', 'hello.txt:/', 'test_job.py'], catch_exceptions=False
        )
        assert res.exit_code == 0, repr((res.output, res.stdout, res.stderr, res.exception))


@pytest.mark.timeout(5 * 60)  # image pulling is very slow
def test_file_mount_in_child_dir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        write_hello(f'{dir}/hello.txt')
        write_script(dir, '/child/hello.txt')
        res = runner.invoke(
            cli.app, ['submit', '--wait', '--files', 'hello.txt:/child/', 'test_job.py'], catch_exceptions=False
        )
        assert res.exit_code == 0, repr((res.output, res.stdout, res.stderr, res.exception))


@pytest.mark.timeout(5 * 60)  # image pulling is very slow
def test_file_mount_in_child_dir_to_root_dir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        write_hello(f'{dir}/child/hello.txt')
        write_script(dir, '/hello.txt')
        res = runner.invoke(
            cli.app, ['submit', '--wait', '--files', 'child/hello.txt:/', 'test_job.py'], catch_exceptions=False
        )
        assert res.exit_code == 0, repr((res.output, res.stdout, res.stderr, res.exception))


@pytest.mark.timeout(5 * 60)  # image pulling is very slow
def test_mount_multiple_files(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        write_hello(f'{dir}/child/hello1.txt')
        write_hello(f'{dir}/child/hello2.txt')
        write_script(dir, '/hello1.txt')
        res = runner.invoke(
            cli.app,
            [
                'submit',
                '--wait',
                '--files',
                'child/hello1.txt:/',
                '--files',
                'child/hello2.txt:/',
                'test_job.py',
            ],
            catch_exceptions=False,
        )
        assert res.exit_code == 0, repr((res.output, res.stdout, res.stderr, res.exception))


@pytest.mark.timeout(5 * 60)  # image pulling is very slow
def test_dir_mount_in_child_dir_to_child_dir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        write_hello(f'{dir}/child/hello1.txt')
        write_hello(f'{dir}/child/hello2.txt')
        write_script(dir, '/child/hello1.txt')
        res = runner.invoke(
            cli.app, ['submit', '--wait', '--files', 'child/:/child/', 'test_job.py'], catch_exceptions=False
        )
        assert res.exit_code == 0, repr((res.output, res.stdout, res.stderr, res.exception))


@pytest.mark.timeout(5 * 60)  # image pulling is very slow
def test_file_outside_curdir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.mkdir(f'{dir}/working_dir')
        os.chdir(f'{dir}/working_dir')
        write_hello(f'{dir}/hello.txt')
        write_script(dir, '/hello.txt')
        res = runner.invoke(
            cli.app, ['submit', '--wait', '--files', f'{dir}/hello.txt:/', '../test_job.py'], catch_exceptions=False
        )
        assert res.exit_code == 0, repr((res.output, res.stdout, res.stderr, res.exception))


@pytest.mark.timeout(5 * 60)  # image pulling is very slow
def test_dir_outside_curdir(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        os.mkdir(f'{dir}/working_dir')
        os.chdir(f'{dir}/working_dir')
        write_hello(f'{dir}/hello1.txt')
        write_hello(f'{dir}/hello2.txt')

        write_script(dir, f'/foo/hello1.txt')

        res = runner.invoke(
            cli.app, ['submit', '--wait', '--files', f'{dir}/:/foo/', '../test_job.py'], catch_exceptions=False
        )
        assert res.exit_code == 0, repr((res.output, res.stdout, res.stderr, res.exception))


@pytest.mark.timeout(5 * 60)  # image pulling is very slow
def test_mounting_dir_to_root_dir_fails(runner: CliRunner):
    with tempfile.TemporaryDirectory() as dir:
        write_hello(f'{dir}/hello1.txt')
        write_hello(f'{dir}/hello2.txt')

        write_script(dir, '/hello1.txt')

        with pytest.raises(ValueError, match='cannot mount a directory to "/"'):
            runner.invoke(
                cli.app, ['submit', '--wait', '--files', f'{dir}/:/', '../test_job.py'], catch_exceptions=False
            )
