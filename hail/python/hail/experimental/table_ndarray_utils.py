from typing import Optional, List, Union, Tuple

import hail as hl
from hail.expr import (check_entry_indexed, matrix_table_source)
from hail.utils.java import Env


def key_intervals(ht: hl.Table, max_rows_per_interval: int, n_rows: int) -> List[hl.utils.Interval]:
    index_field = Env.get_uid()
    ht = ht.add_index(index_field)
    ht = ht.filter(hl.any(ht[index_field] % max_rows_per_interval == 0,
                          ht[index_field] == n_rows - 1))
    interval_bounds = ht.select().collect()
    n_intervals = len(interval_bounds) - 1
    intervals = []
    for index, (left, right) in enumerate(zip(interval_bounds, interval_bounds[1:])):
        is_last = index == n_intervals - 1
        intervals.append(hl.utils.Interval(
            start=left,
            end=right,
            includes_start=True,
            includes_end=is_last))
    return intervals


def reasonable_block_size(typ: hl.HailType, n_rows: int, n_cols: int):
    if n_cols == 0:
        raise ValueError(f'BlockMatrix cannot contain zero columns. Dimensions: {(n_rows, n_cols)}.')

    ndarray_mem_limit = 4096 * 4096
    if typ == hl.tint64:
        row_mem_size = n_cols * 8
    elif typ == hl.tint32:
        row_mem_size = n_cols * 4
    else:
        raise ValueError(f'Unsupported BlockMatrix entry type: {typ}')
    return (ndarray_mem_limit + row_mem_size - 1) // row_mem_size


class TallSkinnyMatrix:
    def __init__(self, block_table: hl.Table, block_expr: hl.Expression, source_mt: hl.MatrixTable, n_rows: int, n_cols: int):
        self.col_key = col_key
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.block_table = block_table
        self.block_expr = block_expr
        self.source_mt = source_mt

    def row_key_table(self) -> hl.Table:
        return self.source_mt.rows()

    def col_key_table(self) -> hl.Table:
        return self.source_mt.cols()

    def entryless_matrix_table(self) -> hl.MatrixTable:
        return self.source_mt.select_entries()


def mt_to_tsm(entry_expr,
              rows_per_block: int,
              dimensions: Optional[Tuple[int, int]],
              ) -> TallSkinnyMatrix:
    check_entry_indexed('mt_to_table_of_ndarray/entry_expr', entry_expr)
    mt = matrix_table_source('mt_to_table_of_ndarray/entry_expr', entry_expr)

    n_rows, n_cols = dimensions or mt.count()

    mt, field = mt.expr_to_field(entry_expr)
    mt = mt.select_globals()
    mt = mt.select_cols()
    mt = mt.select_rows()
    mt = mt.select_entries(field)
    first_checkpoint = hl.utils.new_temp_file("mt_to_table_of_ndarray", "mt")
    source_mt = mt.checkpoint(first_checkpoint)

    new_partitions = key_intervals(mt.rows(), rows_per_ndarray, n_rows)
    mt = hl.read_matrix_table_table(first_checkpoint, _intervals=new_partitions, _assert_type=mt._type)
    mt = mt.select_rows(row_vector = hl.agg.collect(mt[field]))
    ht = mt.rows()
    ht = ht._group_within_partitions("groups", rows_per_ndarray)
    A = ht.select(ndarray=hl.nd.array(ht.groups.row_vector))
    A = A.checkpoint(hl.utils.new_temp_file("mt_to_table_of_ndarray", "A"))
    return TallSkinnyMatrix(A, A.ndarray, source_mt, n_rows, n_cols)
