import hail as hl

from ..helpers import run_in


# run in all locations; however we should do this local only
# once we're sure table_show / table head are covered else where


@run_in('all')
def test_show_1():
    mt = hl.balding_nichols_model(3, 10, 10)
    mt.GT.show()
    mt.locus.show()


@run_in('all')
def test_show_2():
    mt = hl.balding_nichols_model(3, 10, 10)
    mt.af.show()
    mt.pop.show()


@run_in('all')
def test_show_3():
    mt = hl.balding_nichols_model(3, 10, 10)
    mt.sample_idx.show()
    mt.bn.show()


@run_in('all')
def test_show_4():
    mt = hl.balding_nichols_model(3, 10, 10)
    mt.bn.fst.show()
    mt.GT.n_alt_alleles().show()


@run_in('all')
def test_show_5():
    mt = hl.balding_nichols_model(3, 10, 10)
    (mt.GT.n_alt_alleles() * mt.GT.n_alt_alleles()).show()
    (mt.af * mt.GT.n_alt_alleles()).show()


@run_in('all')
def test_show_rows_table():
    t = hl.balding_nichols_model(3, 10, 10).rows()
    t.af.show()
    (t.af * 3).show()


@run_in('all')
def test_show_negative():
    hl.utils.range_table(5).show(-1)


@run_in('all')
def test_show_mt_duplicate_col_key():
    shown_cols = 2

    mt = hl.utils.range_matrix_table(5, 5)
    mt = mt.key_cols_by(c = 0)
    showobj = mt.show(n_cols=shown_cols, handler=lambda x: x)

    assert len(showobj.table_show.table.row) == len(mt.row) + shown_cols


@run_in('all')
def test_show_mt_fewer_cols():
    shown_cols = 7

    mt = hl.utils.range_matrix_table(5, 5)
    mt = mt.key_cols_by(c = 0)
    showobj = mt.show(n_cols=shown_cols, handler=lambda x: x)

    assert len(showobj.table_show.table.row) == len(mt.row) + mt.count_cols()
