from typing import Dict
import hail as hl
from hailtop.frozendict import frozendict
from hailtop.hail_frozenlist import frozenlist

from ..helpers import run_in


@run_in('local')
def test_collect_as_set_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = ['hello'])
    result = t.aggregate_entries(hl.agg.collect_as_set(t.l))

    assert result == {frozenlist(['hello'])}


@run_in('local')
def test_counter_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = ['hello'])
    result = t.aggregate_entries(hl.agg.counter(t.l))

    assert list(result) == [frozenlist(['hello'])]

    assert list(result.values()) == [1]


@run_in('local')
def test_collect_as_set_tuple_of_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = (['hello'],))
    result = t.aggregate_entries(hl.agg.collect_as_set(t.l))

    assert result == {(frozenlist(['hello']),)}


@run_in('local')
def test_counter_tuple_of_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = (['hello'],))
    result = t.aggregate_entries(hl.agg.counter(t.l))

    assert list(result) == [(frozenlist(['hello']),)]

    assert list(result.values()) == [1]


@run_in('local')
def test_collect_as_set_struct_of_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = hl.struct(bad=['hello'], good=3))
    result = t.aggregate_entries(hl.agg.collect_as_set(t.l))

    assert result == {(hl.Struct(bad=frozenlist(['hello']), good=3))}


@run_in('local')
def test_counter_struct_of_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = hl.struct(bad=['hello'], good=3))
    result = t.aggregate_entries(hl.agg.counter(t.l))

    assert list(result) == [hl.Struct(bad=frozenlist(['hello']), good=3)]

    assert list(result.values()) == [1]


@run_in('local')
def test_collect_as_set_dict_value_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = hl.dict([(3, ['hello'])]))
    result = t.aggregate_entries(hl.agg.collect_as_set(t.l))

    assert result == {frozendict({3: frozenlist(['hello'])})}


@run_in('local')
def test_counter_dict_value_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = hl.dict([(3, ['hello'])]))
    result = t.aggregate_entries(hl.agg.counter(t.l))

    assert list(result) == [frozendict({3: frozenlist(['hello'])})]

    assert list(result.values()) == [1]


@run_in('local')
def test_collect_as_set_list_list_list_set_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = [[[hl.set([['hello']])]]])
    result = t.aggregate_entries(hl.agg.collect_as_set(t.l))

    assert result == {frozenlist([frozenlist([frozenlist([frozenset([frozenlist(['hello'])])])])])}


@run_in('local')
def test_counter_list_list_list_set_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = [[[hl.set([['hello']])]]])
    result = t.aggregate_entries(hl.agg.counter(t.l))

    assert list(result) == [frozenlist([frozenlist([frozenlist([frozenset([frozenlist(['hello'])])])])])]

    assert list(result.values()) == [1]


@run_in('local')
def test_collect_dict_value_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = hl.dict([(3, ['hello'])]))
    result = t.aggregate_entries(hl.agg.collect(t.l))

    # NB: We never return dict, only frozendict, so we assert that. However, dict *values* must be
    # hashable if and only if the frozendict must be hashable. In this case, the frozendict *need
    # not* be hashable because its inside a list. As a result, the value *should not* be frozen
    # (i.e. hashable). This preserves backwards compatibility for users who expect normal Python
    # lists when possible.
    assert result == [frozendict({3: ['hello']})]


@run_in('local')
def test_collect_dict_key_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = hl.dict([(['hello'], 3)]))
    result = t.aggregate_entries(hl.agg.collect(t.l))

    assert result == [frozendict({frozenlist(['hello']): 3})]


@run_in('local')
def test_collect_dict_key_and_value_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = hl.dict([(['hello'], ['goodbye'])]))
    result = t.aggregate_entries(hl.agg.collect(t.l))

    # NB: See note in test_collect_dict_value_list.
    assert result == [frozendict({frozenlist(['hello']): ['goodbye']})]


@run_in('local')
def test_collect_set_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = hl.set([['hello']]))
    result = t.aggregate_entries(hl.agg.collect(t.l))

    assert result == [frozenset({frozenlist(['hello'])})]


@run_in('local')
def test_collect_set_dict_list_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = hl.set([hl.dict([(['hello'], ['goodbye'])])]))
    result = t.aggregate_entries(hl.agg.collect(t.l))

    d: Dict[frozenlist[str], frozenlist[str]] = {frozenlist(['hello']): frozenlist(['goodbye'])}
    assert result == [frozenset({frozendict(d)})]


@run_in('local')
def test_collect_set_tuple_struct_struct_list():
    t = hl.utils.range_matrix_table(1, 1)
    t = t.annotate_entries(l = hl.set([(hl.struct(a=hl.struct(inside=['hello'], aside=4.0), b='abc'), 3)]))
    result = t.aggregate_entries(hl.agg.collect(t.l))

    assert result == [frozenset({(hl.Struct(a=hl.Struct(inside=frozenlist(['hello']), aside=4.0), b='abc'), 3)})]
