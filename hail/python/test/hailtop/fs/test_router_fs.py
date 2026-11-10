import concurrent.futures

import pytest

from hailtop.fs.router_fs import RouterFS

pytestmark = pytest.mark.timeout(20)


def test_stat_from_a_thread_without_an_event_loop(tmp_path):
    path = tmp_path / 'file'
    path.write_bytes(b'abc')
    fs = RouterFS()
    try:
        # Hail's sync API must not need an event loop on the calling thread. A new thread has none.
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            stat = pool.submit(fs.stat, str(path)).result()
    finally:
        fs.close()
    assert not stat.is_dir()
    assert stat.size == 3
