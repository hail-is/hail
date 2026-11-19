import os
import signal
import subprocess
import sys
import textwrap

import pytest

import hailtop.fs as hfs
from hailtop.fs import fs_utils
from hailtop.utils import secret_alnum_string
from hailtop.utils.gcs_requester_pays import GCSRequesterPaysFSCache

pytestmark = pytest.mark.timeout(20)


class FakeFS:
    def __init__(self, **kwargs):
        self.closed = False

    def exists(self, path: str) -> bool:
        return True

    def close(self):
        self.closed = True


class FailingCloseFS(FakeFS):
    def close(self):
        raise ValueError('boom')


@pytest.fixture
def fake_fses(monkeypatch) -> GCSRequesterPaysFSCache:
    fses = GCSRequesterPaysFSCache(fs_constructor=FakeFS)
    monkeypatch.setattr(fs_utils, '_fses', fses)
    return fses


def test_close_fses_closes_every_cached_fs(fake_fses):
    hfs.exists('gs://bucket/a')
    hfs.exists('gs://bucket/a', requester_pays_config='my-project')
    cached = [fake_fses[None], fake_fses['my-project']]

    fs_utils._close_fses()

    assert all(fs.closed for fs in cached)
    assert fake_fses.pop_all() == []


def test_close_fses_logs_instead_of_raising(monkeypatch, caplog):
    fses = GCSRequesterPaysFSCache(fs_constructor=FailingCloseFS)
    monkeypatch.setattr(fs_utils, '_fses', fses)
    hfs.exists('gs://bucket/a')

    fs_utils._close_fses()

    assert 'failed to close a hailtop.fs filesystem at exit' in caplog.text
    assert fses.pop_all() == []


def test_a_forked_child_drops_the_parents_fses_without_closing_them(fake_fses):
    hfs.exists('gs://bucket/a')
    parents_fs = fake_fses[None]

    pid = os.fork()
    if pid == 0:
        signal.alarm(5)
        try:
            os._exit(0 if fake_fses.pop_all() == [] and not parents_fs.closed else 1)
        finally:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert fake_fses[None] is parents_fs


def test_fses_are_closed_when_the_interpreter_exits():
    script = textwrap.dedent("""
        import hailtop.fs as hfs
        from hailtop.fs import fs_utils

        class PrintingFS:
            def __init__(self, **kwargs):
                pass

            def exists(self, path):
                return True

            def close(self):
                print('closed', flush=True)

        fs_utils._fses._fs_constructor = PrintingFS
        hfs.exists('gs://bucket/a')
        hfs.exists('gs://bucket/a', requester_pays_config='my-project')
    """)
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['closed', 'closed']


@pytest.mark.timeout(60)
def test_using_cloud_storage_leaves_no_unclosed_sessions_at_exit():
    # aiohttp reports unclosed sessions while the interpreter shuts down, after any test in this
    # process could see them, so the storage is used from a subprocess.
    path = os.path.join(os.environ['HAIL_TEST_STORAGE_URI'], secret_alnum_string(5), 'file')
    script = textwrap.dedent(f"""
        import hailtop.fs as hfs

        with hfs.open({path!r}, 'w') as f:
            f.write('hello')
        assert hfs.exists({path!r})
        with hfs.open({path!r}, 'rb') as f:
            assert f.read() == b'hello'
        hfs.remove({path!r})
    """)
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=45, check=False)
    assert result.returncode == 0, result.stderr
    assert 'Unclosed' not in result.stderr, result.stderr
