import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Union

import orjson
import pytest
from typer.testing import CliRunner, Result

from hailtop import __pip_version__
from hailtop.batch import Batch
from hailtop.hailctl.batch import cli
from hailtop.hailctl.batch.submit import parse_files_to_src_dest
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


def parse_batch_from_text_output(res: Result) -> Batch:
    batch_id = re.findall(r'\d+', res.output)[0]
    return Batch.from_batch_id(int(batch_id))


def puts(filename: Path, content: str):
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(content)


def echo0(dir: Path) -> Path:
    script = dir / f'script{secret_alnum_string(6)}'
    puts(script, 'echo 0')
    return script


def write_pyscript(dir: Union[str, Path], file_to_echo: Union[str, Path]) -> Path:
    script = Path(dir) / f'test_job_{secret_alnum_string(6)}.py'
    puts(script, f'print(open("{file_to_echo}").read())')
    return script


def write_hello(filename: Path):
    puts(filename, 'hello\n')


def test_name(submit, tmp_path, request):
    batch_name = request.node.nodeid + secret_alnum_string()
    res = submit(echo0(tmp_path), '--name', batch_name)
    assert_exit_code(res, 0)

    b = parse_batch_from_text_output(res)
    assert b.run().attributes['name'] == batch_name


def test_image(submit, tmp_path):
    image = 'busybox:latest'
    res = submit(echo0(tmp_path), '--image', image)
    assert_exit_code(res, 0)

    b = parse_batch_from_text_output(res)
    j = b.run().get_job(1)
    assert j.status()['spec']['process']['image'] == image


def test_default_image(submit, tmp_path):
    res = submit(echo0(tmp_path))
    assert_exit_code(res, 0)

    b = parse_batch_from_text_output(res)
    j = b.run().get_job(1)
    assert j.status()['spec']['process']['image'] == f'hailgenetics/hail:{__pip_version__}'


def test_image_environment_variable(submit, tmp_path):
    res = submit(echo0(tmp_path), env={'HAIL_GENETICS_HAIL_IMAGE': 'busybox:latest'})
    assert_exit_code(res, 0)

    b = parse_batch_from_text_output(res)
    j = b.run().get_job(1)
    assert j.status()['spec']['process']['image'] == 'busybox:latest'


def test_script_shebang(submit, tmp_path):
    script_text = """\
#!/usr/bin/env cat
hello,
world!
"""

    script = tmp_path / 'script'
    script.write_text(script_text)
    res = submit(script, '--wait', '-o', 'json')
    assert_exit_code(res, 0)

    output = orjson.loads(res.output)
    assert output['log'] == script_text


@pytest.mark.parametrize('files', ['', ':', ':dst'])
def test_files_invalid_format(submit, files):
    with pytest.raises(ValueError, match='Invalid file specification'):
        submit(__file__, '--wait', '--files', files)


def test_files_copy_rename(submit, tmp_cwd):
    write_hello(tmp_cwd / 'hello.txt')
    pyscript = write_pyscript(tmp_cwd, '/child')
    res = submit(pyscript, '--wait', '--files', 'hello.txt:/child')
    assert_exit_code(res, 0)


@pytest.mark.parametrize('files', ['.', '.:.', '*', '*.txt'])
def test_files_copy_cwd(submit, tmp_cwd, files):
    write_hello(tmp_cwd / 'hello.txt')
    write_hello(tmp_cwd / 'hello2.txt')

    script = tmp_cwd / 'script'
    script.write_text(
        """\
cat hello.txt
cat hello2.txt
""",
    )

    res = submit(script, '--wait', '--files', files)
    assert_exit_code(res, 0)


@pytest.mark.parametrize(
    'files, remote',
    [
        ('a', '.'),
        ('a:/', '/'),
        ('a:a', 'a'),
        ('a/b:a', 'a'),
        ('a/../b:b', 'b'),
        ('a:a/b', 'a/b'),
        ('a:a/../b', 'b'),
    ],
)
def test_files_copy_folder(submit, tmp_cwd, files, remote):
    src, dst = parse_files_to_src_dest(files)
    write_hello(src / 'hello.txt')
    pyscript = write_pyscript(tmp_cwd, Path(remote) / 'hello.txt')
    res = submit(pyscript, '--wait', '--files', files)
    assert_exit_code(res, 0)


def test_files_nested_folders(submit, tmp_path):
    puts(tmp_path / 'python' / 'main' / '__init__.py', '')
    puts(tmp_path / 'python' / 'main' / 'a' / '__init__.py', 'message: str = "hello"')
    puts(tmp_path / 'python' / 'main' / 'b' / '__init__.py', 'message: str = "world"')

    script = tmp_path / 'script'
    script.write_text(
        """\
#!/usr/bin/env python3
from main import a, b
print(f'{a.message}, {b.message}')
""",
    )

    res = submit(script, '--wait', '--files', str(tmp_path / 'python'))
    assert_exit_code(res, 0)


def test_files_mount_multiple_files_options(submit, tmp_cwd):
    write_hello(tmp_cwd / 'hello1.txt')
    write_hello(tmp_cwd / 'hello2.txt')

    script = tmp_cwd / 'script'
    script.write_text(
        """
        cat a/hello.txt
        cat b/hello.txt
        """,
    )

    res = submit(script, '--wait', '--files', 'hello1.txt:a/hello.txt', '--files', 'hello2.txt:b/hello.txt')
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


def test_files_environment_variables(submit, tmp_path):
    write_hello(tmp_path / 'hello.txt')
    pyscript = write_pyscript(tmp_path, 'hello.txt')

    varname = secret_alnum_string(6)
    os.environ[varname] = str(tmp_path)
    res = submit(pyscript, '--wait', '--files', f'${varname}')
    assert_exit_code(res, 0)


def test_files_unsupported_glob_patterns(submit):
    with pytest.raises(ValueError, match='Recursive SRC glob patterns are not supported'):
        submit(__file__, '--wait', '--files', '../**/*.txt')


@pytest.mark.parametrize('files', ['/', '/*'])
def test_files_unsupported_transfer_root(submit, files):
    with pytest.raises(ValueError, match='Cannot transfer whole drive or root filesystem to remote worker'):
        submit(__file__, '--wait', '--files', files)


def test_files_clobber_file_with_folder(submit, tmp_path):
    write_hello(tmp_path / 'hello.txt')
    pyscript = write_pyscript(tmp_path, 'hello.txt')
    res = submit(pyscript, '--files', f'{tmp_path}/hello.txt', '--files', f'{tmp_path}:hello.txt')
    assert_exit_code(res, 1)
