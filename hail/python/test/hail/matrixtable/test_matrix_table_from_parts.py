import pytest
import hail as hl

class Inits():
    globals={'hello': 'world'}
    rows={'foo': ['a', 'b']}
    cols={'bar': ['c', 'd']}
    entries={'baz': [[1, 2], [3, 4]]}

def unless(test: bool, kvs):
    return {} if test else kvs

def assert_matches_globals(mt: 'hl.MatrixTable', no_props=False):
    assert hl.eval(mt.globals) == hl.Struct(**unless(no_props, Inits.globals))

def assert_matches_rows(mt: 'hl.MatrixTable', no_props=False):
    assert mt.rows().order_by('row_idx').collect() == \
        [ hl.Struct(row_idx=0, **unless(no_props, {'foo':'a'}))
        , hl.Struct(row_idx=1, **unless(no_props, {'foo':'b'}))
        ]

def assert_matches_cols(mt: 'hl.MatrixTable', no_props=False):
    assert mt.cols().order_by('col_idx').collect() == \
        [ hl.Struct(col_idx=0, **unless(no_props, {'bar': 'c'}))
        , hl.Struct(col_idx=1, **unless(no_props, {'bar': 'd'}))
        ]

def assert_matches_entries( mt: 'hl.MatrixTable'
                          , no_row_props=False
                          , no_col_props=False
                          , no_entry_props=False
                          ):
    assert mt.entries().order_by('row_idx', 'col_idx').collect() == \
        [ hl.Struct( **unless(no_row_props, {'foo':'a'})
                   , row_idx=0
                   , col_idx=0
                   , **unless(no_col_props, {'bar':'c'})
                   , **unless(no_entry_props, {'baz':1})
                   )
        , hl.Struct( **unless(no_row_props, {'foo':'a'})
                   , row_idx=0
                   , col_idx=1
                   , **unless(no_col_props, {'bar':'d'})
                   , **unless(no_entry_props, {'baz':2})
                   )
        , hl.Struct( **unless(no_row_props, {'foo':'b'})
                   , row_idx=1
                   , col_idx=0
                   , **unless(no_col_props, {'bar':'c'})
                   , **unless(no_entry_props, {'baz':3})
                   )
        , hl.Struct( **unless(no_row_props, {'foo':'b'})
                   , row_idx=1
                   , col_idx=1
                   , **unless(no_col_props, {'bar':'d'})
                   , **unless(no_entry_props, {'baz':4})
                   )
        ]

def test_from_parts():
    mt = hl.MatrixTable.from_parts( globals=Inits.globals
                                  , rows=Inits.rows
                                  , cols=Inits.cols
                                  , entries=Inits.entries
                                  )
    assert_matches_globals(mt)
    assert_matches_rows(mt)
    assert_matches_cols(mt)
    assert_matches_entries(mt)

def test_optional_globals():
    mt = hl.MatrixTable.from_parts( rows=Inits.rows
                                  , cols=Inits.cols
                                  , entries=Inits.entries
                                  )
    assert_matches_globals(mt, no_props=True)
    assert_matches_rows(mt)
    assert_matches_cols(mt)
    assert_matches_entries(mt)

def test_optional_rows():
    mt = hl.MatrixTable.from_parts( globals=Inits.globals
                                  , cols=Inits.cols
                                  , entries=Inits.entries
                                  )
    assert_matches_globals(mt)
    assert_matches_rows(mt, no_props=True)
    assert_matches_cols(mt)
    assert_matches_entries(mt, no_row_props=True)

def test_optional_cols():
    mt = hl.MatrixTable.from_parts( globals=Inits.globals
                                  , rows=Inits.rows
                                  , entries=Inits.entries
                                  )
    assert_matches_globals(mt)
    assert_matches_rows(mt)
    assert_matches_cols(mt, no_props=True)
    assert_matches_entries(mt, no_col_props=True)

def test_optional_globals_and_cols():
    mt = hl.MatrixTable.from_parts( rows=Inits.rows
                                  , entries=Inits.entries
                                  )
    assert_matches_globals(mt, no_props=True)
    assert_matches_rows(mt)
    assert_matches_cols(mt, no_props=True)
    assert_matches_entries(mt, no_col_props=True)

def test_optional_globals_and_rows_and_cols():
    mt = hl.MatrixTable.from_parts(entries=Inits.entries)
    assert_matches_globals(mt, no_props=True)
    assert_matches_rows(mt, no_props=True)
    assert_matches_cols(mt, no_props=True)
    assert_matches_entries(mt, no_row_props=True, no_col_props=True)

def test_optional_entries():
    mt = hl.MatrixTable.from_parts(rows=Inits.rows, cols=Inits.cols)
    assert_matches_globals(mt, no_props=True)
    assert_matches_rows(mt)
    assert_matches_cols(mt)
    assert_matches_entries(mt, no_entry_props=True)

def assert_raises_when_no_rows_and_entries():
    with pytest.raises(AssertionError):
        hl.MatrixTable.from_parts(cols=Inits.cols)

def assert_raises_when_no_cols_and_entries():
    with pytest.raises(AssertionError):
        hl.MatrixTable.from_parts(rows=Inits.rows)
