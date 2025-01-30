import os
from contextlib import contextmanager
from pathlib import Path
from typing import List, Union

import orjson
import pytest
from typer.testing import CliRunner, Result

import hailtop.batch_client.client as bc
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.batch_client.client import BatchClient
from hailtop.config import get_remote_tmpdir
from hailtop.hailctl.batch import cli
from hailtop.utils import secret_alnum_string


@pytest.fixture(scope='function', autouse=True)
def expect_timeouts_as_image_pulling_is_very_slow(request):
    five_minutes = 5 * 60
    timeout = pytest.mark.timeout(five_minutes)
    request.node.add_marker(timeout)


@pytest.fixture(scope='session')
def client():
    client = BatchClient('test')
    yield client
    client.close()


@pytest.fixture
def submit(request):
    runner = CliRunner()

    def invoker(args: List[str], opts: List[str], **kwargs):
        args = [str(arg) for arg in args]
        command = ['submit', *opts, *args]

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


def get_batch_from_text_output(res: Result, client: BatchClient) -> bc.Batch:
    batch_id = orjson.loads(res.output)['id']
    return client.get_batch(batch_id)


def puts(filename: Path, content: str):
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(content)


def echo0(dir: Path) -> Path:
    script = dir / f'script{secret_alnum_string(6)}.sh'
    puts(script, f'ls -R {os.path.dirname(str(dir))}; echo 0')
    return script


def write_pyscript(dir: Union[str, Path], file_to_echo: Union[str, Path]) -> Path:
    script = Path(dir) / f'test_job_{secret_alnum_string(6)}.py'
    puts(
        script,
        f"""
import os
print(os.listdir("{os.path.dirname(str(file_to_echo))}"))
print(open("{file_to_echo}").read())
""",
    )
    return script


def write_hello(filename: Path):
    puts(filename, 'hello\n')


def test_name(submit, tmp_path, request, client):
    batch_name = request.node.nodeid + secret_alnum_string()
    echo_script = echo0(tmp_path)
    res = submit(
        [echo_script.name],
        ['--name', batch_name, '-v', f'{echo_script}:/', '--wait', '-o', 'json', '--quiet'],
    )
    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    assert b.attributes['name'] == batch_name, str(b.debug_info())


def test_workdir(submit, tmp_path, request, client):
    batch_name = request.node.nodeid + secret_alnum_string()
    echo_script = echo0(tmp_path)
    res = submit(
        ['sh', echo_script.name],
        [
            '--name',
            batch_name,
            '--workdir',
            '/workdir/',
            '-v',
            f'{echo_script}:/workdir/',
            '--wait',
            '-o',
            'json',
            '--quiet',
        ],
    )
    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    assert b.status()['state'] == 'success', str(b.debug_info())


def test_image(submit, tmp_path, client):
    image = 'busybox:latest'
    echo_script = echo0(tmp_path)
    res = submit(
        ['sh', echo_script.name], ['--image', image, '-v', f'{echo_script}:/', '--wait', '-o', 'json', '--quiet']
    )
    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    j = b.get_job(1)
    assert j.status()['spec']['process']['image'] == image, str(j.status())


def test_image_environment_variable(submit, tmp_path, client):
    echo_script = echo0(tmp_path)
    res = submit(
        ['sh', echo_script.name],
        ['-v', f'{echo_script}:/', '--wait', '--quiet', '-o', 'json'],
        env={'HAIL_GENETICS_HAIL_IMAGE': 'busybox:latest'},
    )
    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    j = b.get_job(1)
    assert j.status()['spec']['process']['image'] == 'busybox:latest', str(j.status())


@pytest.mark.xfail(reason='Executable files are not supported.')
def test_script_shebang(submit, tmp_path, client):
    script_text = """\
#!/bin/bash
echo "Hello, world!"
"""

    script = tmp_path / 'script.sh'
    script.write_text(script_text)
    res = submit(
        [f'./{script.name}'],
        ['--wait', '--quiet', '-o', 'json', '-v', f'{script}:/', '--wait', '-o', 'json'],
    )
    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    log_output = b.get_job_log(1)['main']
    assert 'Hello, world!' in log_output


@pytest.mark.parametrize("src_dest", ['', ':', ':dst'])
def test_files_invalid_format(submit, src_dest):
    res = submit([__file__], ['--wait', '--quiet', '-v', src_dest])
    assert_exit_code(res, 2)
    assert 'Invalid format for file mount' in res.output


def test_files_invalid_source_dest(submit, tmp_path):
    with pytest.raises(ValueError, match='not a directory'):
        submit([__file__], ['--wait', '--quiet', '-v', '/garbage_path_does_not_exist/:/'])

    with pytest.raises(ValueError, match='copy and renaming a directory is not supported'):
        this_dir = os.path.curdir.rstrip('/') + '/'
        submit([__file__], ['--wait', '--quiet', '-v', f'{this_dir}:/a'])


def test_files_copy_rename(submit, tmp_cwd, client):
    write_hello(tmp_cwd / 'hello.txt')
    pyscript = write_pyscript(tmp_cwd, '/renamed_hello.txt')
    res = submit(
        ['python3', pyscript.name],
        ['--wait', '--quiet', '-v', 'hello.txt:/renamed_hello.txt', '-v', f'{pyscript.name}:/', '--wait', '-o', 'json'],
    )
    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    assert b.status()['state'] == 'success', str(b.debug_info())


@pytest.mark.parametrize(
    'src, dest, hello_dir',
    [
        ('a', '/', '/a/'),
        ('a/', '/', '/'),
        ('a/b', '/a/b/', '/a/b/b/'),
        ('a/../b', '/indirect/', '/indirect/b/'),
        ('a/b/', '/', '/'),
        ('a/', '/b/', '/b/'),
    ],
)
def test_files_copy_folder(submit, tmp_cwd, src, dest, hello_dir, client):
    write_hello(Path(src) / 'hello.txt')
    pyscript = write_pyscript(tmp_cwd, Path(hello_dir) / 'hello.txt')
    res = submit(
        ['python3', pyscript.name],
        ['--quiet', '--wait', '-o', 'json', '-v', f'{src}:{dest}', '-v', f'{pyscript}:/'],
    )
    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    assert b.status()['state'] == 'success', str(b.debug_info())


def test_files_nested_folders(submit, tmp_path, client):
    puts(tmp_path / 'python' / 'main' / '__init__.py', '')
    puts(tmp_path / 'python' / 'main' / 'a' / '__init__.py', 'message: str = "hello"')
    puts(tmp_path / 'python' / 'main' / 'b' / '__init__.py', 'message: str = "world"')

    script = tmp_path / 'script.py'
    script.write_text(
        """\
#!/usr/bin/env python3
from main import a, b
print(f'{a.message}, {b.message}')
""",
    )

    res = submit(
        ['python3', script.name],
        [
            '--workdir',
            '/python/',
            '--wait',
            '--quiet',
            '-o',
            'json',
            '-v',
            f"{tmp_path / 'python'!s}:/",
            '-v',
            f"{script}:/python/",
        ],
    )
    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    assert b.status()['state'] == 'success', str(b.debug_info())


def test_files_mount_multiple_files_options(submit, tmp_cwd, client):
    write_hello(tmp_cwd / 'hello1.txt')
    write_hello(tmp_cwd / 'hello2.txt')

    script = tmp_cwd / 'script.sh'
    script.write_text(
        """
        cat /a/hello.txt
        cat /b/hello.txt
        """,
    )

    res = submit(
        ['sh', script.name],
        [
            '--wait',
            '-o',
            'json',
            '--quiet',
            '-v',
            'hello1.txt:/a/hello.txt',
            '-v',
            'hello2.txt:/b/hello.txt',
            '-v',
            f'{script}:/',
        ],
    )

    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    assert b.status()['state'] == 'success', str(b.debug_info())


def test_files_outside_current_dir(submit, tmp_path, client):
    with tmp_cwd(tmp_path / 'working') as cwd:
        write_hello(tmp_path / 'data' / 'hello.txt')
        pyscript = write_pyscript(cwd, '/hello.txt')
        res = submit(
            ['python3', pyscript.name],
            [
                '--wait',
                '-o',
                'json',
                '--quiet',
                '-v',
                f'{tmp_path}/data/hello.txt:/',
                '-v',
                f'{pyscript}:/',
            ],
        )
        assert_exit_code(res, 0)

        b = get_batch_from_text_output(res, client)
        assert b.status()['state'] == 'success', str(b.debug_info())


def test_files_dir_outside_curdir(submit, tmp_path, client):
    with tmp_cwd(tmp_path / 'working') as workdir_path:
        write_hello(workdir_path / 'hello1.txt')
        write_hello(workdir_path / 'hello2.txt')
        pyscript = write_pyscript(tmp_path, '/foo/working/hello1.txt')
        res = submit(
            ['python3', f'/foo/{pyscript.name}'], ['--wait', '--quiet', '-o', 'json', '-v', f'{tmp_path}:/foo']
        )
        assert_exit_code(res, 0)

        b = get_batch_from_text_output(res, client)
        assert b.status()['state'] == 'success', str(b.debug_info())


def test_submit_with_args(submit, tmp_path, client):
    script = tmp_path / 'script'
    script.write_text(
        """\
#!/usr/bin/env python3
import sys
args = sys.argv[1:]
assert args == ['1', '2', 'a', 'b', '--foo', 'bar=5']
""",
    )

    res = submit(
        ['python3', script.name, '1', '2', 'a', 'b', '--foo', 'bar=5'],
        ['--wait', '-o', 'json', '--quiet', '-v', f"{script}:/"],
    )
    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    assert b.status()['state'] == 'success', str(b.debug_info())


def test_submit_with_proper_job_settings(submit, tmp_path, client):
    remote_tmpdir = get_remote_tmpdir('test_submit.py::tmpdir')

    fs = RouterAsyncFS()
    url = fs.parse_url(remote_tmpdir)
    bucket = '/'.join(url.bucket_parts)

    echo_script = echo0(tmp_path)
    res = submit(
        ['sh', echo_script.name],
        [
            '--cpu',
            '0.25',
            '--memory',
            'highmem',
            '--storage',
            '15Gi',
            '--region',
            "us-east1",
            '--region',
            "us-central1",
            '--attrs',
            'foo=bar',
            '--env',
            'FOO=bar',
            '--remote-tmpdir',
            remote_tmpdir,
            '--cloudfuse',
            f'{bucket}:/foo:true',
            '-v',
            f'{echo_script}:/',
            '--wait',
            '-o',
            'json',
            '--quiet',
        ],
    )
    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    j = b.get_job(1)

    assert j.status()['spec']['resources']['req_cpu'] == '0.25', str(j.status())
    assert j.status()['spec']['resources']['req_memory'] == 'highmem', str(j.status())
    assert j.status()['spec']['resources']['req_storage'] == '15Gi', str(j.status())
    assert set(j.status()['spec']['regions']) == set(['us-east1', 'us-central1']), str(j.status())
    assert 'gcsfuse' in j.status()['spec'], str(j.status())
    assert 'FOO' in [env['name'] for env in j.status()['spec']['env']], str(j.status())
    assert j.status()['spec']['process']['image'] == os.environ['HAIL_GENETICS_HAIL_IMAGE'], str(j.status())


def test_hail_config_in_right_place(submit, tmp_path, request, client):
    batch_name = request.node.nodeid + secret_alnum_string()
    script = tmp_path / 'script.py'
    script.write_text(
        """\
#!/usr/bin/env python3
import os
assert "XDG_CONFIG_HOME" in os.environ
files = os.listdir(os.environ["XDG_CONFIG_HOME"])
assert os.path.isfile(os.environ["XDG_CONFIG_HOME"] + "/hail/config.ini"), str(files)
""",
    )
    res = submit(
        ['python3', script.name], ['--name', batch_name, '-v', f'{script}:/', '--wait', '-o', 'json', '--quiet']
    )
    assert_exit_code(res, 0)

    b = get_batch_from_text_output(res, client)
    assert b.status()['state'] == 'success', str(b.debug_info())
