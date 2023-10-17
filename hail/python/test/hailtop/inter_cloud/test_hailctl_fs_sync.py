from typing import Tuple, Dict, AsyncIterator, List
import pytest
import os.path
import tempfile
import secrets
import asyncio
import pytest
from hailtop.utils import url_scheme, check_exec_output
from hailtop.aiotools import Transfer, FileAndDirectoryError, Copier, AsyncFS, FileListEntry


from .generate_copy_test_specs import run_test_spec, create_test_file, create_test_dir
from .test_copy import cloud_scheme, router_filesystem, fresh_dir


@pytest.mark.asyncio
async def test_cli_file_and_dir(router_filesystem, cloud_scheme):
    sema, fs, bases = router_filesystem

    test_dir = await fresh_dir(fs, bases, cloud_scheme)
    plandir = await fresh_dir(fs, bases, cloud_scheme)
    plandir2 = await fresh_dir(fs, bases, cloud_scheme)

    await fs.write(f"{test_dir}file1", b"hello world\n")

    await check_exec_output(
        'hailctl',
        'fs',
        'sync',
        '--make-plan',
        plandir,
        '--copy-to',
        f'{test_dir}file1',
        f'{test_dir}file2',
        '--copy-into',
        f'{test_dir}file1',
        f'{test_dir}dir1',
    )

    await check_exec_output(
        'hailctl',
        'fs',
        'sync',
        '--use-plan',
        plandir,
    )

    expected_files = [f"{test_dir}file1", f"{test_dir}file2", f"{test_dir}dir1/file1"]
    for url in expected_files:
        assert await fs.read(url) == b"hello world\n"

    await check_exec_output(
        'hailctl',
        'fs',
        'sync',
        '--make-plan',
        plandir2,
        '--copy-to',
        f'{test_dir}file1',
        f'{test_dir}file2',
        '--copy-into',
        f'{test_dir}file1',
        f'{test_dir}dir1',
    )
    assert await fs.read(plandir2 + 'differs') == b''
    assert await fs.read(plandir2 + 'dstonly') == b''
    assert await fs.read(plandir2 + 'srconly') == b''


@pytest.mark.asyncio
async def test_cli_subdir(router_filesystem, cloud_scheme):
    sema, fs, bases = router_filesystem

    test_dir = await fresh_dir(fs, bases, cloud_scheme)
    plandir = await fresh_dir(fs, bases, cloud_scheme)
    plandir2 = await fresh_dir(fs, bases, cloud_scheme)

    await fs.makedirs(f"{test_dir}dir")
    await fs.makedirs(f"{test_dir}dir/subdir")
    await fs.write(f"{test_dir}dir/subdir/file1", b"hello world\n")

    await check_exec_output(
        'hailctl', 'fs', 'sync', '--make-plan', plandir, '--copy-to', f'{test_dir}dir', f'{test_dir}dir2'
    )

    await check_exec_output(
        'hailctl',
        'fs',
        'sync',
        '--use-plan',
        plandir,
    )

    assert await fs.read(f"{test_dir}dir2/subdir/file1") == b"hello world\n"

    await check_exec_output(
        'hailctl', 'fs', 'sync', '--make-plan', plandir2, '--copy-to', f'{test_dir}dir', f'{test_dir}dir2'
    )

    assert await fs.read(plandir2 + 'plan') == b''
    assert await fs.read(plandir2 + 'summary') == b'0\t0\n'
    assert await fs.read(plandir2 + 'differs') == b''
    assert await fs.read(plandir2 + 'dstonly') == b''
    assert await fs.read(plandir2 + 'srconly') == b''
    assert await fs.read(plandir2 + 'matches') == f'{test_dir}dir/subdir/file1\t{test_dir}dir2/subdir/file1\n'.encode()


@pytest.mark.asyncio
async def test_cli_already_synced(router_filesystem, cloud_scheme):
    sema, fs, bases = router_filesystem

    test_dir = await fresh_dir(fs, bases, cloud_scheme)
    plandir = await fresh_dir(fs, bases, cloud_scheme)

    await fs.makedirs(f"{test_dir}dir")
    await fs.write(f"{test_dir}dir/foo", b"hello world\n")
    await fs.write(f"{test_dir}bar", b"hello world\n")

    await check_exec_output(
        'hailctl', 'fs', 'sync', '--make-plan', plandir, '--copy-to', f'{test_dir}dir/foo', f'{test_dir}bar'
    )

    assert await fs.read(plandir + 'plan') == b''
    assert await fs.read(plandir + 'summary') == b'0\t0\n'
    assert await fs.read(plandir + 'differs') == b''
    assert await fs.read(plandir + 'dstonly') == b''
    assert await fs.read(plandir + 'srconly') == b''
    assert await fs.read(plandir + 'matches') == f'{test_dir}dir/foo\t{test_dir}bar\n'.encode()
