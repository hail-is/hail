import pytest

import hail as hl


def unless(test: bool, kvs):
    return {} if test else kvs


class TestData:
    globals = {'hello': 'world'}
    rows = {'foo': ['a', 'b']}
    cols = {'bar': ['c', 'd']}
    entries = {'baz': [[1, 2], [3, 4]]}

    @staticmethod
    def assert_matches_globals(mt: 'hl.MatrixTable', no_props=False):
        assert hl.eval(mt.globals) == hl.Struct(**unless(no_props, TestData.globals))

    @staticmethod
    def assert_no_globals(mt: 'hl.MatrixTable'):
        TestData.assert_matches_globals(mt, no_props=True)

    @staticmethod
    def assert_matches_rows(mt: 'hl.MatrixTable', no_props=False):
        assert mt.rows().collect() == [
            hl.Struct(row_idx=0, **unless(no_props, {'foo': 'a'})),
            hl.Struct(row_idx=1, **unless(no_props, {'foo': 'b'})),
        ]

    @staticmethod
    def assert_matches_cols(mt: 'hl.MatrixTable', no_props=False):
        assert mt.cols().collect() == [
            hl.Struct(col_idx=0, **unless(no_props, {'bar': 'c'})),
            hl.Struct(col_idx=1, **unless(no_props, {'bar': 'd'})),
        ]

    @staticmethod
    def assert_matches_entries(mt: 'hl.MatrixTable', no_props=False):
        assert mt.select_rows().select_cols().entries().collect() == [
            hl.Struct(row_idx=0, col_idx=0, **unless(no_props, {'baz': 1})),
            hl.Struct(row_idx=0, col_idx=1, **unless(no_props, {'baz': 2})),
            hl.Struct(row_idx=1, col_idx=0, **unless(no_props, {'baz': 3})),
            hl.Struct(row_idx=1, col_idx=1, **unless(no_props, {'baz': 4})),
        ]


def test_from_parts():
    mt = hl.MatrixTable.from_parts(
        globals=TestData.globals, rows=TestData.rows, cols=TestData.cols, entries=TestData.entries
    )
    TestData.assert_matches_globals(mt)
    TestData.assert_matches_rows(mt)
    TestData.assert_matches_cols(mt)
    TestData.assert_matches_entries(mt)


def test_optional_globals():
    mt = hl.MatrixTable.from_parts(rows=TestData.rows, cols=TestData.cols, entries=TestData.entries)
    TestData.assert_no_globals(mt)
    TestData.assert_matches_rows(mt)
    TestData.assert_matches_cols(mt)
    TestData.assert_matches_entries(mt)


def test_optional_rows():
    mt = hl.MatrixTable.from_parts(globals=TestData.globals, cols=TestData.cols, entries=TestData.entries)
    TestData.assert_matches_globals(mt)
    TestData.assert_matches_rows(mt, no_props=True)
    TestData.assert_matches_cols(mt)
    TestData.assert_matches_entries(mt)


def test_optional_cols():
    mt = hl.MatrixTable.from_parts(globals=TestData.globals, rows=TestData.rows, entries=TestData.entries)
    TestData.assert_matches_globals(mt)
    TestData.assert_matches_rows(mt)
    TestData.assert_matches_cols(mt, no_props=True)
    TestData.assert_matches_entries(mt)


def test_optional_globals_and_cols():
    mt = hl.MatrixTable.from_parts(rows=TestData.rows, entries=TestData.entries)
    TestData.assert_no_globals(mt)
    TestData.assert_matches_rows(mt)
    TestData.assert_matches_cols(mt, no_props=True)
    TestData.assert_matches_entries(mt)


def test_optional_globals_and_rows_and_cols():
    mt = hl.MatrixTable.from_parts(entries=TestData.entries)
    TestData.assert_no_globals(mt)
    TestData.assert_matches_rows(mt, no_props=True)
    TestData.assert_matches_cols(mt, no_props=True)
    TestData.assert_matches_entries(mt)


def test_optional_entries():
    mt = hl.MatrixTable.from_parts(rows=TestData.rows, cols=TestData.cols)
    TestData.assert_no_globals(mt)
    TestData.assert_matches_rows(mt)
    TestData.assert_matches_cols(mt)
    TestData.assert_matches_entries(mt, no_props=True)


def test_rectangular_matrices():
    mt = hl.MatrixTable.from_parts(entries={'foo': [[1], [2]]})
    assert mt.select_rows().select_cols().entries().collect() == [
        hl.Struct(row_idx=0, col_idx=0, foo=1),
        hl.Struct(row_idx=1, col_idx=0, foo=2),
    ]


def test_raises_when_no_rows_and_entries():
    with pytest.raises(AssertionError):
        hl.MatrixTable.from_parts(cols=TestData.cols)


def test_raises_when_no_cols_and_entries():
    with pytest.raises(AssertionError):
        hl.MatrixTable.from_parts(rows=TestData.rows)


def test_raises_when_mismatched_row_property_dimensions():
    with pytest.raises(ValueError):
        hl.MatrixTable.from_parts(rows={'foo': [1], 'bar': [1, 2]}, entries=TestData.entries)


def test_raises_when_mismatched_col_property_dimensions():
    with pytest.raises(ValueError):
        hl.MatrixTable.from_parts(cols={'foo': [1], 'bar': [1, 2]}, entries=TestData.entries)


def test_raises_when_mismatched_entry_property_dimensions():
    with pytest.raises(ValueError):
        hl.MatrixTable.from_parts(entries={'foo': [[1]], 'bar': [[1, 2]]})


def test_raises_when_mismatched_rows_cols_entry_dimensions():
    with pytest.raises(ValueError):
        hl.MatrixTable.from_parts(rows={'foo': [1]}, cols={'bar': [1]}, entries={'baz': [[1, 2]]})
