from typing import Optional, List, Tuple

import hail as hl
from hail.expr import (check_entry_indexed, matrix_table_source)
from hail.utils.java import Env


def key_intervals(ht: hl.Table, max_rows_per_interval: int, n_rows: int) -> List[hl.utils.Interval]:
    assert max_rows_per_interval is not None
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
    if typ == hl.tfloat64:
        row_mem_size = n_cols * 8
    elif typ == hl.tfloat32:
        row_mem_size = n_cols * 4
    else:
        raise ValueError(f'Unsupported BlockMatrix entry type: {typ}')
    return (ndarray_mem_limit + row_mem_size - 1) // row_mem_size


class TallSkinnyMatrix:
    def __init__(self,
                 block_matrix_table: hl.MatrixTable,
                 col_key: List[str],
                 n_rows: int,
                 n_cols: int,
                 rows_per_block: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        # We must use an MT with exactly one col instead of a Table because
        #
        #     mt.annotate_cols(x=hl.agg.collect(mt.row))
        #
        # is allowed but
        #
        #     ht.annotate_globals(x=hl.agg.collect(mt.row))
        #
        # is not.
        self.mt = block_matrix_table
        assert 'block' in block_matrix_table.row
        self.col_key = col_key
        # NB: the last row may have fewer rows
        self.rows_per_block = rows_per_block

    def __getitem__(self, x):
        return self.mt.__getitem__(x)

    def __getattr__(self, x):
        return self.mt.__getattr__(x)

    def _copy(self, mt):
        return TallSkinnyMatrix(mt,
                                self.col_key,
                                self.n_rows,
                                self.n_cols,
                                self.rows_per_block)

    def annotate_globals(self, **kwargs):
        return self._copy(self.mt.annotate_cols(**kwargs))

    def select_globals(self, **kwargs):
        return self._copy(self.mt.annotate_cols(**kwargs))

    def annotate(self, **kwargs):
        return self._copy(self.mt.annotate_entries(**kwargs))

    def select(self, **kwargs):
        return self._copy(self.mt.annotate_entries(**kwargs))

    def drop(self, *args):
        return self._copy(self.mt.drop(*args))

    def checkpoint(self, path):
        return self._copy(self.checkpoint(path))

    def to_table(self):
        mt = self.mt
        ht = mt.localize_entries('fake_entries', 'fake_cols')

        the_col = ht.fake_cols[0]
        ht = ht.select_globals(
            *mt.globals,
            **{k: the_col[k] for k in the_col}
        )

        the_entry = ht.fake_entries[0]
        ht = ht.select(
            *mt.row_value,
            **{k: the_entry[k] for k in the_entry}
        )
        return ht

    @property
    def globals(self):
        return self.to_table().globals


def mt_to_tsm(entry_expr,
              rows_per_block: Optional[int],
              dimensions: Optional[Tuple[int, int]],
              ) -> TallSkinnyMatrix:
    check_entry_indexed('mt_to_table_of_ndarray/entry_expr', entry_expr)
    mt = matrix_table_source('mt_to_table_of_ndarray/entry_expr', entry_expr)

    n_rows, n_cols = dimensions or mt.count()
    rows_per_block = reasonable_block_size(entry_expr.dtype, n_rows, n_cols)
    field = Env.get_uid()
    mt = mt.select_entries(**{field: entry_expr})
    mt = mt.select_globals()
    mt = mt.select_cols()
    mt = mt.select_rows()
    first_checkpoint = hl.utils.new_temp_file('mt_to_table_of_ndarray', 'mt')
    source_mt = mt.checkpoint(first_checkpoint)

    new_partitions = key_intervals(mt.rows(), rows_per_block, n_rows)
    mt = hl.read_matrix_table(first_checkpoint, _intervals=new_partitions, _assert_type=mt._type, _load_refs=False)
    ht = mt.localize_entries('row_vector', 'col_keys')
    ht = ht.annotate(row_vector = ht.row_vector[field])
    ht = ht._group_within_partitions('groups', rows_per_block)
    ht = ht.annotate(row_keys = ht.groups.map(lambda g: g.select(*mt.row_key)))
    ht = ht.annotate(block = hl.nd.array(ht.groups.row_vector))
    ht = ht.select('row_keys', 'block')
    ht = ht.checkpoint(hl.utils.new_temp_file('mt_to_table_of_ndarray', 'A'))
    ht = ht.annotate_globals(fake_cols = [hl.struct()])
    ht = ht.annotate(fake_entries = [hl.struct()])
    mt = ht._unlocalize_entries('fake_entries', 'fake_cols', col_key=[])

    col_key_fields = list(source_mt.col_key)

    return TallSkinnyMatrix(mt, col_key_fields, n_rows, n_cols, rows_per_block)
