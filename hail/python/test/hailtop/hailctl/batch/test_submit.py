import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Union

import pytest
from typer.testing import CliRunner, Result

from hailtop import __pip_version__
from hailtop.batch import Batch
from hailtop.hailctl.batch import cli
from hailtop.utils import secret_alnum_string


@pytest.fixture(scope='function', autouse=True)
def expect_timeouts_as_image_pulling_is_very_slow(request):
    five_minutes = 5 * 60
    timeout = pytest.mark.timeout(five_minutes)
    request.node.add_marker(timeout)


@pytest.fixture
def submit(request):
    runner = CliRunner(mix_stderr=False)

    def invoker(script: Union[str, os.PathLike], *args: str, **kwargs):
        command = ['submit', str(script), *args]

        # For ease of identifying the test in the batch ui
        if '--name' not in command:
            command += ['--name', request.node.nodeid]

        return runner.invoke(
            cli.app,
            command,
            catch_exceptions=kwargs.get('catch_exceptions', False),
            **kwargs,
        )

    return invoker


@contextmanager
def tmp_cwd(path: Path):
    cwd = os.getcwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


@pytest.fixture(name='tmp_cwd')
def tmp_cwd_fixture(tmp_path):
    with tmp_cwd(tmp_path) as cd:
        yield cd


def assert_exit_code(res: Result, exit_code: int):
    assert res.exit_code == exit_code, repr((res.output, res.stdout, res.stderr, res.exception))


def parse_batch_from_output(res: Result) -> Batch:
    batch_id = re.findall(r'\d+', res.output)[0]
    return Batch.from_batch_id(int(batch_id))


def echo0(dir: Path) -> Path:
    script = dir / f'script{secret_alnum_string(6)}'
    script.write_text('echo 0')
    return script


def write_pyscript(dir: Union[str, Path], file_to_echo: str) -> Path:
    script = Path(dir) / f'test_job{secret_alnum_string(6)}.py'
    script.write_text(f'print(open("{file_to_echo}").read())')
    return script


def write_hello(filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text('hello\n')


def test_name(submit, tmp_path, request):
    batch_name = request.node.nodeid + secret_alnum_string()
    res = submit(echo0(tmp_path), '--name', batch_name)
    assert_exit_code(res, 0)

    b = parse_batch_from_output(res)
    assert b.run().attributes['name'] == batch_name


def test_image(submit, tmp_path):
    image = 'busybox:latest'
    res = submit(echo0(tmp_path), '--image', image)
    assert_exit_code(res, 0)

    b = parse_batch_from_output(res)
    j = b.run().get_job(1)
    assert j.status()['spec']['process']['image'] == image


def test_default_image(submit, tmp_path):
    res = submit(echo0(tmp_path))
    assert_exit_code(res, 0)

    b = parse_batch_from_output(res)
    j = b.run().get_job(1)
    assert j.status()['spec']['process']['image'] == f'hailgenetics/hail:{__pip_version__}'


def test_image_environment_variable(submit, tmp_path):
    res = submit(echo0(tmp_path), env={'HAIL_GENETICS_HAIL_IMAGE': 'busybox:latest'})
    assert_exit_code(res, 0)

    b = parse_batch_from_output(res)
    j = b.run().get_job(1)
    assert j.status()['spec']['process']['image'] == 'busybox:latest'


def test_shebang(submit, tmp_path):
    script = tmp_path / 'script'
    script.write_text(
        """#!/usr/bin/env cat
        hello,
        world!
        """,
    )
    res = submit(script, '--wait')
    assert_exit_code(res, 0)


@pytest.mark.parametrize('files', ['.', '*', '*.txt'])
def test_copy_cwd_no_dst(submit, tmp_cwd, files):
    write_hello(tmp_cwd / 'hello.txt')
    write_hello(tmp_cwd / 'hello2.txt')

    script = tmp_cwd / 'script'
    script.write_text(
        """
        echo hello.txt
        echo hello2.txt
        """,
    )

    res = submit(script, '--wait', '--files', files)
    assert_exit_code(res, 0)


def test_files_copy_rename(submit, tmp_cwd):
    write_hello(tmp_cwd / 'hello.txt')
    pyscript = write_pyscript(tmp_cwd, '/child')
    res = submit(pyscript, '--wait', '--files', 'hello.txt:/child')
    assert_exit_code(res, 0)


def test_files_mount_copy_file_in_child_dir_to_root(submit, tmp_cwd):
    write_hello(tmp_cwd / 'child/hello.txt')
    pyscript = write_pyscript(tmp_cwd, '/hello.txt')
    res = submit(pyscript, '--wait', '--files', 'child/hello.txt:/')
    assert_exit_code(res, 0)


def test_files_mount_multiple_files_options(submit, tmp_cwd):
    write_hello(tmp_cwd / 'hello1.txt')
    write_hello(tmp_cwd / 'hello2.txt')

    script = tmp_cwd / 'script'
    script.write_text(
        """
        echo /bodger
        echo /badger
        """,
    )

    res = submit(script, '--wait', '--files', 'hello1.txt:/bodger', '--files', 'hello2.txt:/badger')
    assert_exit_code(res, 0)


def test_files_outside_current_dir(submit, tmp_path):
    with tmp_cwd(tmp_path / 'working') as cwd:
        write_hello(tmp_path / 'data' / 'hello.txt')
        pyscript = write_pyscript(cwd, '/hello.txt')
        res = submit(pyscript, '--wait', '--files', f'{tmp_path}/data/hello.txt:/')
        assert_exit_code(res, 0)


def test_files_relative_dst(submit, tmp_path):
    with tmp_cwd(tmp_path / 'working') as cwd:
        write_hello(tmp_path / 'hello.txt')
        pyscript = write_pyscript(cwd, '../hello.txt')
        res = submit(pyscript, '--wait', '--files', '../hello.txt:../')
        assert_exit_code(res, 0)


def test_files_dir_outside_curdir(submit, tmp_path):
    with tmp_cwd(tmp_path / 'working'):
        write_hello(tmp_path / 'hello1.txt')
        write_hello(tmp_path / 'hello2.txt')
        pyscript = write_pyscript(tmp_path, '/foo/hello1.txt')
        res = submit(pyscript, '--wait', '--files', f'{tmp_path}:/foo')
        assert_exit_code(res, 0)


@pytest.mark.parametrize('glob', ['*', '*.txt'])
def test_files_copy_contents_into_dir(submit, tmp_path, glob):
    write_hello(tmp_path / 'hello1.txt')
    pyscript = write_pyscript(tmp_path, '/hello1.txt')
    res = submit(pyscript, '--wait', '--files', f'{tmp_path}/{glob}:/')
    assert_exit_code(res, 0)


def test_files_environment_variables(submit, tmp_path):
    write_hello(tmp_path / 'hello.txt')
    pyscript = write_pyscript(tmp_path, 'hello.txt')

    varname = secret_alnum_string(6)
    os.environ[varname] = str(tmp_path)
    res = submit(pyscript, '--wait', '--files', f'${varname}')
    assert_exit_code(res, 0)


def test_unsupported_glob_patterns(submit, tmp_path):
    pyscript = write_pyscript(tmp_path, 'hello.txt')
    res = submit(pyscript, '--wait', '--files', '../**/*.txt')
    assert_exit_code(res, 1)
