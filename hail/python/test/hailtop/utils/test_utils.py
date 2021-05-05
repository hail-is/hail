from hailtop.utils import (partition, url_basename, url_join, url_scheme,
                           parse_docker_image_reference)


def test_partition_zero_empty():
    assert list(partition(0, [])) == []


def test_partition_even_small():
    assert list(partition(3, range(3))) == [range(0, 1), range(1, 2), range(2, 3)]


def test_partition_even_big():
    assert list(partition(3, range(9))) == [range(0, 3), range(3, 6), range(6, 9)]


def test_partition_uneven_big():
    assert list(partition(2, range(9))) == [range(0, 5), range(5, 9)]


def test_partition_toofew():
    assert list(partition(6, range(3))) == [range(0, 1), range(1, 2), range(2, 3),
                                            range(3, 3), range(3, 3), range(3, 3)]


def test_url_basename():
    assert url_basename('/path/to/file') == 'file'
    assert url_basename('https://hail.is/path/to/file') == 'file'


def test_url_join():
    assert url_join('/path/to', 'file') == '/path/to/file'
    assert url_join('/path/to/', 'file') == '/path/to/file'
    assert url_join('/path/to/', '/absolute/file') == '/absolute/file'
    assert url_join('https://hail.is/path/to', 'file') == 'https://hail.is/path/to/file'
    assert url_join('https://hail.is/path/to/', 'file') == 'https://hail.is/path/to/file'
    assert url_join('https://hail.is/path/to/', '/absolute/file') == 'https://hail.is/absolute/file'


def test_url_scheme():
    assert url_scheme('https://hail.is/path/to') == 'https'
    assert url_scheme('/path/to') == ''

def test_parse_docker_image_reference():
    x = parse_docker_image_reference('animage')
    assert x.domain is None
    assert x.path == 'animage'
    assert x.tag is None
    assert x.digest is None
    assert x.name() == 'animage'
    assert str(x) == 'animage'

    x = parse_docker_image_reference('hailgenetics/animage')
    assert x.domain == 'hailgenetics'
    assert x.path == 'animage'
    assert x.tag is None
    assert x.digest is None
    assert x.name() == 'hailgenetics/animage'
    assert str(x) == 'hailgenetics/animage'

    x = parse_docker_image_reference('localhost:5000/animage')
    assert x.domain == 'localhost:5000'
    assert x.path == 'animage'
    assert x.tag is None
    assert x.digest is None
    assert x.name() == 'localhost:5000/animage'
    assert str(x) == 'localhost:5000/animage'

    x = parse_docker_image_reference('localhost:5000/a/b/name')
    assert x.domain == 'localhost:5000'
    assert x.path == 'a/b/name'
    assert x.tag is None
    assert x.digest is None
    assert x.name() == 'localhost:5000/a/b/name'
    assert str(x) == 'localhost:5000/a/b/name'

    x = parse_docker_image_reference('localhost:5000/a/b/name:tag')
    assert x.domain == 'localhost:5000'
    assert x.path == 'a/b/name'
    assert x.tag == 'tag'
    assert x.digest is None
    assert x.name() == 'localhost:5000/a/b/name'
    assert str(x) == 'localhost:5000/a/b/name:tag'

    x = parse_docker_image_reference('localhost:5000/a/b/name:tag@sha256:abc123')
    assert x.domain == 'localhost:5000'
    assert x.path == 'a/b/name'
    assert x.tag == 'tag'
    assert x.digest == 'sha256:abc123'
    assert x.name() == 'localhost:5000/a/b/name'
    assert str(x) == 'localhost:5000/a/b/name:tag@sha256:abc123'

    x = parse_docker_image_reference('localhost:5000/a/b/name@sha256:abc123')
    assert x.domain == 'localhost:5000'
    assert x.path == 'a/b/name'
    assert x.tag is None
    assert x.digest == 'sha256:abc123'
    assert x.name() == 'localhost:5000/a/b/name'
    assert str(x) == 'localhost:5000/a/b/name@sha256:abc123'

    x = parse_docker_image_reference('name@sha256:abc123')
    assert x.domain is None
    assert x.path == 'name'
    assert x.tag is None
    assert x.digest == 'sha256:abc123'
    assert x.name() == 'name'
    assert str(x) == 'name@sha256:abc123'

    x = parse_docker_image_reference('gcr.io/hail-vdc/batch-worker:123fds312')
    assert x.domain == 'gcr.io'
    assert x.path == 'hail-vdc/batch-worker'
    assert x.tag == '123fds312'
    assert x.digest is None
    assert x.name() == 'gcr.io/hail-vdc/batch-worker'
    assert str(x) == 'gcr.io/hail-vdc/batch-worker:123fds312'

    x = parse_docker_image_reference('us-docker.pkg.dev/my-project/my-repo/test-image')
    assert x.domain == 'us-docker.pkg.dev'
    assert x.path == 'my-project/my-repo/test-image'
    assert x.tag is None
    assert x.digest is None
    assert x.name() == 'us-docker.pkg.dev/my-project/my-repo/test-image'
    assert str(x) == 'us-docker.pkg.dev/my-project/my-repo/test-image'
