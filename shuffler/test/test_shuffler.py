import hail as hl

from hailtop.config import get_deploy_config

shuffle_service_url = get_deploy_config().base_url('shuffler')
print(f'using shuffle service url {shuffle_service_url}')


def test_reverse_range_table():
    optimizer_iterations = hl._get_flags('optimizer_iterations')['optimizer_iterations']
    hl._set_flags(shuffle_service_url=shuffle_service_url,
                  optimizer_iterations="0")
    t = hl.utils.range_table(30, n_partitions=8)
    t = t.order_by(-t.idx)

    expected = [hl.Struct(idx=x) for x in range(30)]
    expected = sorted(expected, key=lambda x: -x.idx)
    assert t.collect() == expected
    hl._set_flags(shuffle_service_url=None,
                  optimizer_iterations=optimizer_iterations)


def test_range_table_as_strings():
    optimizer_iterations = hl._get_flags('optimizer_iterations')['optimizer_iterations']
    hl._set_flags(shuffle_service_url=shuffle_service_url,
                  optimizer_iterations="0")
    t = hl.utils.range_table(30, n_partitions=8)
    t = t.annotate(x=hl.str(t.idx))
    t = t.order_by(t.x)

    expected = [hl.Struct(idx=x, x=str(x)) for x in range(30)]
    expected = sorted(expected, key=lambda x: x.x)
    assert t.collect() == expected
    hl._set_flags(shuffle_service_url=None,
                  optimizer_iterations=optimizer_iterations)


def test_range_table_with_garbage():
    optimizer_iterations = hl._get_flags('optimizer_iterations')['optimizer_iterations']
    hl._set_flags(shuffle_service_url=shuffle_service_url,
                  optimizer_iterations="0")
    t = hl.utils.range_table(30, n_partitions=8)
    t = t.annotate(
        x = 'foo' + hl.str(t.idx),
        y = 3 * t.idx,
        z = hl.dict({'x': 3.14, 'y': 6.28, 'idx': hl.float(t.idx)})
    )
    t = t.order_by(-t.idx)

    expected = [hl.Struct(idx=x,
                          x=f'foo{x}',
                          y=3 * x,
                          z={'x': 3.14, 'y': 6.28, 'idx': float(x)})
                for x in range(30)]
    expected = sorted(expected, key=lambda x: -x.idx)
    assert t.collect() == expected
    hl._set_flags(shuffle_service_url=None,
                  optimizer_iterations=optimizer_iterations)


def test_large_table():
    optimizer_iterations = hl._get_flags('optimizer_iterations')['optimizer_iterations']
    hl._set_flags(shuffle_service_url=shuffle_service_url,
                  optimizer_iterations="0")
    t = hl.utils.range_table(1_000_000, n_partitions=100)
    t = t.order_by(-t.idx)

    expected = [hl.Struct(idx=x) for x in range(1_000_000)]
    expected = sorted(expected, key=lambda x: -x.idx)
    assert t.collect() == expected
    hl._set_flags(shuffle_service_url=None,
                  optimizer_iterations=optimizer_iterations)


def test_large_table_key_by():
    optimizer_iterations = hl._get_flags('optimizer_iterations')['optimizer_iterations']
    hl._set_flags(shuffle_service_url=shuffle_service_url,
                  optimizer_iterations="0")
    t = hl.utils.range_table(1_000_000, n_partitions=100)
    t = t.key_by(rev_idx=-t.idx)

    expected = [hl.Struct(idx=x, rev_idx=-x) for x in range(1_000_000)]
    expected = sorted(expected, key=lambda x: x.rev_idx)
    assert t.collect() == expected
    hl._set_flags(shuffle_service_url=None,
                  optimizer_iterations=optimizer_iterations)


def test_large_table_key_by_workaround():
    optimizer_iterations = hl._get_flags('optimizer_iterations')['optimizer_iterations']
    hl._set_flags(shuffle_service_url=shuffle_service_url,
                  optimizer_iterations="0")
    t = hl.utils.range_table(1_000_000, n_partitions=100)
    t = t.key_by(rev_idx=-t.idx)

    expected = [hl.Struct(idx=x, rev_idx=-x) for x in range(1_000_000)]
    expected = sorted(expected, key=lambda x: x.rev_idx)
    t._force_count()
    hl._set_flags(shuffle_service_url=None,
                  optimizer_iterations=optimizer_iterations)
