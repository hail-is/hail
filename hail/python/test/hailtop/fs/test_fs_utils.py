import subprocess
import sys
import textwrap

import pytest

import hailtop.fs as hfs
from hailtop.fs import fs_utils
from hailtop.utils.gcs_requester_pays import GCSRequesterPaysFSCache

pytestmark = pytest.mark.timeout(60)


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


def test_close_fses_closes_every_cached_fs(monkeypatch):
    fses = GCSRequesterPaysFSCache(fs_constructor=FakeFS)
    monkeypatch.setattr(fs_utils, '_fses', fses)
    hfs.exists('gs://bucket/a')
    hfs.exists('gs://bucket/a', requester_pays_config='my-project')
    cached = list(fses._dict.values())
    assert len(cached) == 2

    fs_utils._close_fses()

    assert all(fs.closed for fs in cached)
    assert fses._dict == {}


def test_close_fses_logs_instead_of_raising(monkeypatch, caplog):
    fses = GCSRequesterPaysFSCache(fs_constructor=FailingCloseFS)
    monkeypatch.setattr(fs_utils, '_fses', fses)
    hfs.exists('gs://bucket/a')

    fs_utils._close_fses()

    assert 'failed to close a hailtop.fs filesystem at exit' in caplog.text
    assert fses._dict == {}


def test_forget_fses_after_fork_drops_without_closing(monkeypatch):
    fses = GCSRequesterPaysFSCache(fs_constructor=FakeFS)
    monkeypatch.setattr(fs_utils, '_fses', fses)
    hfs.exists('gs://bucket/a')
    (fs,) = fses._dict.values()

    fs_utils._forget_fses_after_fork()

    assert not fs.closed
    assert fses._dict == {}


def test_fses_are_closed_when_the_interpreter_exits():
    script = textwrap.dedent(
        """
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
        """
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == ['closed', 'closed']
