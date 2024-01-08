import os
import secrets
from typing import Generator

import pytest

import hail as hl
from hail.context import _get_local_tmpdir
from hail.utils import hadoop_copy, hadoop_ls, hadoop_open
from hail.utils.java import FatalError
from hailtop import fs
from hailtop.utils import secret_alnum_string

from ..helpers import qobtest


def touch(fs, filename: str):
    with fs.open(filename, 'w') as fobj:
        fobj.write('hello world')


@pytest.fixture(params=['remote', 'local'])
def tmpdir(request) -> Generator[str, None, None]:
    if request.param == 'local':
        tmpdir = _get_local_tmpdir(None)
        tmpdir = tmpdir[len('file://') :]
    else:
        tmpdir = os.environ['HAIL_TEST_STORAGE_URI']
    tmpdir = os.path.join(tmpdir, secret_alnum_string(5))

    fs.mkdir(tmpdir)
    yield tmpdir
    fs.rmtree(tmpdir)


@qobtest
def test_hadoop_methods_1(tmpdir: str):
    data = ['foo', 'bar', 'baz']
    data.extend(map(str, range(100)))

    with hadoop_open(f'{tmpdir}/test_out.txt', 'w') as f:
        for d in data:
            f.write(d)
            f.write('\n')

    with hadoop_open(f'{tmpdir}/test_out.txt') as f:
        data2 = [line.strip() for line in f]

    assert data == data2


@qobtest
def test_hadoop_methods_2(tmpdir: str):
    data = ['foo', 'bar', 'baz']
    data.extend(map(str, range(100)))

    with hadoop_open(f'{tmpdir}/test_out.txt.gz', 'w') as f:
        for d in data:
            f.write(d)
            f.write('\n')

    with hadoop_open(f'{tmpdir}/test_out.txt.gz') as f:
        data3 = [line.strip() for line in f]

    assert data == data3


@qobtest
def test_hadoop_methods_3(tmpdir: str):
    data = ['foo', 'bar', 'baz']
    data.extend(map(str, range(100)))

    with hadoop_open(f'{tmpdir}/test_out.txt.gz', 'w') as f:
        for d in data:
            f.write(d)
            f.write('\n')

    hadoop_copy(f'{tmpdir}/test_out.txt.gz', f'{tmpdir}/test_out.copy.txt.gz')

    with hadoop_open(f'{tmpdir}/test_out.copy.txt.gz') as f:
        data4 = [line.strip() for line in f]

    assert data == data4

    print(f'contents of my tmpdir, {tmpdir}:')
    print(repr(hadoop_ls(tmpdir)))


@qobtest
def test_read_overwrite(tmpdir):
    with fs.open(os.path.join(tmpdir, 'randomBytes'), 'wb') as f:
        f.write(secrets.token_bytes(2048))

    with fs.open(os.path.join(tmpdir, 'randomBytes'), 'rb', buffer_size=100) as f:
        with hadoop_open(f'{tmpdir}/randomBytesOut', 'wb', buffer_size=2**18) as out:
            b = f.read()
            out.write(b)

    with hadoop_open(f'{tmpdir}/randomBytesOut', 'rb', buffer_size=2**18) as f:
        b2 = f.read()

    assert b == b2


@qobtest
def test_hadoop_exists(tmpdir: str):
    with hadoop_open(f'{tmpdir}/test_exists.txt', 'w') as f:
        f.write("HELLO WORLD")

    r_exists = f'{tmpdir}/test_exists.txt'
    r_not_exists = f'{tmpdir}/not_exists.txt'
    assert hl.hadoop_exists(r_exists)
    assert not hl.hadoop_exists(r_not_exists)


@qobtest
def test_hadoop_is_file(tmpdir: str):
    a_file = f'{tmpdir}/test_hadoop_is_file.txt'
    with hadoop_open(a_file, 'w') as f:
        f.write("HELLO WORLD")

    assert hl.hadoop_is_file(a_file)
    assert not hl.hadoop_is_file(f'{tmpdir}/')
    assert not hl.hadoop_is_file(f'{tmpdir}/invalid-path')


@qobtest
def test_hadoop_stat(tmpdir: str):
    data = ['foo', 'bar', 'baz']
    data.extend(map(str, range(100)))
    with hadoop_open(f'{tmpdir}/test_hadoop_stat.txt.gz', 'w') as f:
        for d in data:
            f.write(d)
            f.write('\n')

    stat1 = hl.hadoop_stat(f'{tmpdir}')
    assert stat1['is_dir']

    hadoop_copy(f'{tmpdir}/test_hadoop_stat.txt.gz', f'{tmpdir}/test_hadoop_stat.copy.txt.gz')

    stat2 = hl.hadoop_stat(f'{tmpdir}/test_hadoop_stat.copy.txt.gz')
    # The gzip format permits metadata which makes the compressed file's size unpredictable. In
    # practice, Hadoop creates a 175 byte file and gzip.GzipFile creates a 202 byte file. The 27
    # extra bytes appear to include at least the filename (20 bytes) and a modification timestamp.
    assert stat2['size_bytes'] == 175 or stat2['size_bytes'] == 202
    assert not stat2['is_dir']
    assert 'path' in stat2


@qobtest
def test_subdirs(tmpdir: str):
    dir = f'{tmpdir}foo/'
    subdir1 = f'{dir}foo/'
    subdir1subdir1 = f'{subdir1}foo/'
    subdir1subdir2 = f'{subdir1}bar/'
    subdir1subdir3 = f'{subdir1}baz/'
    subdir1subdir4_empty = f'{subdir1}qux/'
    subdir2 = f'{dir}bar/'
    subdir3 = f'{dir}baz/'
    subdir4_empty = f'{dir}qux/'

    fs.mkdir(dir)
    touch(fs, f'{dir}a')
    touch(fs, f'{dir}b')

    fs.mkdir(subdir1)
    fs.mkdir(subdir1subdir1)
    fs.mkdir(subdir1subdir2)
    fs.mkdir(subdir1subdir3)
    fs.mkdir(subdir1subdir4_empty)
    fs.mkdir(subdir2)
    fs.mkdir(subdir3)
    fs.mkdir(subdir4_empty)

    for subdir in [dir, subdir1, subdir2, subdir3, subdir1subdir1, subdir1subdir2, subdir1subdir3]:
        for i in range(30):
            touch(fs, f'{subdir}a{i:02}')

    assert fs.is_dir(dir)
    assert fs.is_dir(subdir1)
    assert fs.is_dir(subdir1subdir1)
    assert fs.is_dir(subdir1subdir2)
    assert fs.is_dir(subdir1subdir3)
    # subdir1subdir4_empty: in cloud fses, empty dirs do not exist and thus are not dirs
    assert fs.is_dir(subdir2)
    assert fs.is_dir(subdir3)
    # subdir4_empty: in cloud fses, empty dirs do not exist and thus are not dirs

    fs.rmtree(subdir1subdir2)

    assert fs.is_dir(dir)
    assert fs.is_file(f'{dir}a')
    assert fs.is_file(f'{dir}b')

    assert fs.is_dir(subdir1)
    assert fs.is_file(f'{subdir1}a00')

    assert fs.is_dir(subdir1subdir1)
    assert fs.is_file(f'{subdir1subdir1}a00')

    assert not fs.is_dir(subdir1subdir2)
    assert not fs.is_file(f'{subdir1subdir2}a00')

    assert fs.is_dir(subdir1subdir3)
    assert fs.is_file(f'{subdir1subdir3}a00')

    assert fs.is_dir(subdir2)
    assert fs.is_file(f'{subdir2}a00')
    assert fs.is_dir(subdir3)
    assert fs.is_file(f'{subdir3}a00')

    fs.rmtree(dir)

    assert not fs.is_dir(dir)


@qobtest
def test_rmtree_empty_is_ok(tmpdir: str):
    dir = f'{tmpdir}foo/'
    fs.mkdir(dir)
    fs.rmtree(dir)


@qobtest
def test_rmtree_empty_subdir_is_ok(tmpdir: str):
    dir = f'{tmpdir}foo/'
    fs.mkdir(f'{tmpdir}foo/')
    fs.mkdir(f'{tmpdir}foo/bar')
    fs.rmtree(dir)


@qobtest
def test_remove_and_rmtree(tmpdir: str):
    dir = f'{tmpdir}foo/'
    subdir1 = f'{dir}foo/'
    subdir1subdir1 = f'{subdir1}foo/'
    subdir1subdir2 = f'{subdir1}bar/'
    subdir1subdir3 = f'{subdir1}baz/'

    fs.mkdir(dir)
    touch(fs, f'{dir}a')
    touch(fs, f'{dir}b')

    fs.mkdir(subdir1)
    touch(fs, f'{subdir1}a')
    fs.mkdir(subdir1subdir1)
    touch(fs, f'{subdir1subdir1}a')
    fs.mkdir(subdir1subdir2)
    touch(fs, f'{subdir1subdir2}a')
    fs.mkdir(subdir1subdir3)
    touch(fs, f'{subdir1subdir3}a')

    try:
        fs.remove(subdir1subdir2)
    except (FileNotFoundError, IsADirectoryError):
        pass
    except FatalError as err:
        java_nio_error_message = 'DirectoryNotEmptyException: Cannot delete a non-empty directory'
        hadoop_error_message = f'Directory {subdir1subdir2.rstrip("/")} is not empty'
        assert java_nio_error_message in err.args[0] or hadoop_error_message in err.args[0]
    else:
        assert False

    fs.remove(f'{subdir1subdir2}a')

    assert fs.exists(dir)
    assert fs.exists(f'{dir}a')
    assert fs.exists(f'{dir}b')
    assert fs.exists(subdir1)
    assert fs.exists(f'{subdir1}a')
    assert fs.exists(subdir1subdir1)
    assert fs.exists(f'{subdir1subdir1}a')
    # subdir1subdir2: will exist in cloud, but not local, so do not test for it
    assert not fs.exists(f'{subdir1subdir2}a')
    assert fs.exists(subdir1subdir3)
    assert fs.exists(f'{subdir1subdir3}a')

    fs.rmtree(subdir1subdir1)

    assert fs.exists(dir)
    assert fs.exists(f'{dir}a')
    assert fs.exists(f'{dir}b')
    assert fs.exists(subdir1)
    assert fs.exists(f'{subdir1}a')
    assert not fs.exists(subdir1subdir1)
    assert not fs.exists(f'{subdir1subdir1}a')
    # subdir1subdir2: will exist in cloud, but not local, so do not test for it
    assert not fs.exists(f'{subdir1subdir2}a')
    assert fs.exists(subdir1subdir3)
    assert fs.exists(f'{subdir1subdir3}a')

    fs.rmtree(subdir1)

    assert fs.exists(dir)
    assert fs.exists(f'{dir}a')
    assert fs.exists(f'{dir}b')
    assert not fs.exists(subdir1)
    assert not fs.exists(f'{subdir1}a')
    assert not fs.exists(subdir1subdir1)
    assert not fs.exists(f'{subdir1subdir1}a')
    assert not fs.exists(subdir1subdir2)
    assert not fs.exists(f'{subdir1subdir2}a')
    assert not fs.exists(subdir1subdir3)
    assert not fs.exists(f'{subdir1subdir3}a')
