from hailtop.utils import partition, url_basename, url_join, url_scheme


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
