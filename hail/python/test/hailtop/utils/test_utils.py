from hailtop.utils import (
    partition,
    url_basename,
    url_join,
    url_scheme,
    url_and_params,
    parse_docker_image_reference,
    grouped,
)
from hailtop.utils.utils import digits_needed, unzip, filter_none, flatten


def test_partition_zero_empty():
    assert list(partition(0, [])) == []


def test_partition_even_small():
    assert list(partition(3, range(3))) == [range(0, 1), range(1, 2), range(2, 3)]


def test_partition_even_big():
    assert list(partition(3, range(9))) == [range(0, 3), range(3, 6), range(6, 9)]


def test_partition_uneven_big():
    assert list(partition(2, range(9))) == [range(0, 5), range(5, 9)]


def test_partition_toofew():
    assert list(partition(6, range(3))) == [
        range(0, 1),
        range(1, 2),
        range(2, 3),
        range(3, 3),
        range(3, 3),
        range(3, 3),
    ]


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


def test_url_and_params():
    assert url_and_params('https://example.com/') == ('https://example.com/', {})
    assert url_and_params('https://example.com/foo?') == ('https://example.com/foo', {})
    assert url_and_params('https://example.com/foo?a=b&c=d') == ('https://example.com/foo', {'a': 'b', 'c': 'd'})


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


def test_grouped_size_0_groups_9_elements():
    try:
        list(grouped(0, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
    except ValueError:
        pass
    else:
        assert False


def test_grouped_size_1_groups_9_elements():
    actual = list(grouped(1, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
    expected = [[1], [2], [3], [4], [5], [6], [7], [8], [9]]
    assert actual == expected


def test_grouped_size_5_groups_9_elements():
    actual = list(grouped(5, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
    expected = [[1, 2, 3, 4, 5], [6, 7, 8, 9]]
    assert actual == expected


def test_grouped_size_3_groups_0_elements():
    actual = list(grouped(3, []))
    expected = []
    assert actual == expected


def test_grouped_size_2_groups_1_elements():
    actual = list(grouped(2, [1]))
    expected = [[1]]
    assert actual == expected


def test_grouped_size_1_groups_0_elements():
    actual = list(grouped(1, [0]))
    expected = [[0]]
    assert actual == expected


def test_grouped_size_1_groups_5_elements():
    actual = list(grouped(1, ['abc', 'def', 'ghi', 'jkl', 'mno']))
    expected = [['abc'], ['def'], ['ghi'], ['jkl'], ['mno']]
    assert actual == expected


def test_grouped_size_2_groups_5_elements():
    actual = list(grouped(2, ['abc', 'def', 'ghi', 'jkl', 'mno']))
    expected = [['abc', 'def'], ['ghi', 'jkl'], ['mno']]
    assert actual == expected


def test_grouped_size_3_groups_6_elements():
    actual = list(grouped(3, ['abc', 'def', 'ghi', 'jkl', 'mno', '']))
    expected = [['abc', 'def', 'ghi'], ['jkl', 'mno', '']]
    assert actual == expected


def test_grouped_size_3_groups_7_elements():
    actual = list(grouped(3, ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqr', 'stu']))
    expected = [['abc', 'def', 'ghi'], ['jkl', 'mno', 'pqr'], ['stu']]
    assert actual == expected


def test_unzip():
    assert unzip([]) == ([], [])
    assert unzip([(0, 'a')]) == ([0], ['a'])
    assert unzip([(123, '')]) == ([123], [''])
    assert unzip([(123, 'abc')]) == ([123], ['abc'])
    assert unzip([(123, 456), ('abc', 'def')]) == ([123, 'abc'], [456, 'def'])
    assert unzip([(123, 'abc'), (456, 'def'), (789, 'ghi')]) == ([123, 456, 789], ['abc', 'def', 'ghi'])


def test_digits_needed():
    assert digits_needed(0) == 1
    assert digits_needed(1) == 1
    assert digits_needed(12) == 2
    assert digits_needed(333) == 3
    assert digits_needed(100) == 3
    assert digits_needed(3000) == 4
    assert digits_needed(50000) == 5


def test_filter_none():
    assert filter_none([]) == []
    assert filter_none([None, []]) == [[]]
    assert filter_none([0, []]) == [0, []]
    assert filter_none([1, 2, [None]]) == [1, 2, [None]]
    assert filter_none(
        [
            1,
            3.5,
            2,
            4,
        ]
    ) == [1, 3.5, 2, 4]
    assert filter_none([1, 2, 3.0, None, 5]) == [1, 2, 3.0, 5]
    assert filter_none(['a', 'b', 'c', None]) == ['a', 'b', 'c']
    assert filter_none([None, [None, [None, [None]]]]) == [[None, [None, [None]]]]


def test_flatten():
    assert flatten([]) == []
    assert flatten([[]]) == []
    assert flatten([[], []]) == []
    assert flatten([[], [3]]) == [3]
    assert flatten([[1, 2, 3], [3], [4, 5]]) == [1, 2, 3, 3, 4, 5]
    assert flatten([['a', 'b', 'c'], ['d', 'e']]) == ['a', 'b', 'c', 'd', 'e']
    assert flatten([[['a'], ['b']], [[1, 2, 3], [4, 5]]]) == [['a'], ['b'], [1, 2, 3], [4, 5]]
    assert flatten([['apples'], ['bannanas'], ['oranges']]) == ['apples', 'bannanas', 'oranges']
    assert flatten([['apple', 'bannana'], ['a', 'b', 'c'], [1, 2, 3, 4]]) == [
        'apple',
        'bannana',
        'a',
        'b',
        'c',
        1,
        2,
        3,
        4,
    ]
    assert flatten([['apples'], [''], ['bannanas'], [''], ['oranges'], ['']]) == [
        'apples',
        '',
        'bannanas',
        '',
        'oranges',
        '',
    ]
