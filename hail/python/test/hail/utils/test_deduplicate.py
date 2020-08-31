from hail.utils.deduplicate import deduplicate


def test_deduplicate_simple():
    mappings, new_ids = deduplicate([str(i) for i in range(5)])
    assert mappings == []
    assert new_ids == ['0', '1', '2', '3', '4']

    mappings, new_ids = deduplicate('0' for i in range(5))
    assert mappings == [('0', '0_1'), ('0', '0_2'), ('0', '0_3'), ('0', '0_4')]
    assert new_ids == ['0', '0_1', '0_2', '0_3', '0_4']

    mappings, new_ids = deduplicate(['0', '0_1', '0', '0_2', '0'])
    assert mappings == [('0', '0_2'), ('0_2', '0_2_1'), ('0', '0_3')]
    assert new_ids == ['0', '0_1', '0_2', '0_2_1', '0_3']


def test_deduplicate_max_attempts():
    try:
        deduplicate(['0', '0_1', '0'], max_attempts=1)
    except RecursionError as exc:
        assert 'cannot deduplicate' in exc.args[0]
    else:
        assert False


def test_deduplicate_already_used():
    mappings, new_ids = deduplicate(['0', '0_1', '0'],
                                    already_used={'0_1', '0_2'})
    assert mappings == [('0_1', '0_1_1'), ('0', '0_3')]
    assert new_ids == ['0', '0_1_1', '0_3']
