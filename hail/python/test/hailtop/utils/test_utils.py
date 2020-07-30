from hailtop.utils import partition


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
