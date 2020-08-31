import json
import numpy as np
from typing import Optional, Tuple, Callable, Any

from hail.utils.java import Env
from hail.utils import range_table, new_temp_file
from hail.expr import Expression
from hail import nd
from hail.matrixtable import MatrixTable
from hail.table import Table

import hail as hl


def array(mt: MatrixTable,
          entry_field: str,
          *,
          n_partitions: Optional[int] = None,
          block_size: Optional[int] = None,
          sort_columns: bool = False) -> 'DNDArray':
    return DNDArray.from_matrix_table(mt,
                                      entry_field,
                                      n_partitions=n_partitions,
                                      block_size=block_size,
                                      sort_columns=sort_columns)


def read(fname: str) -> 'DNDArray':
    # read without good partitioning, just to get the globals
    a = DNDArray(hl.read_table(fname))
    t = hl.read_table(fname, _intervals=[
        hl.Interval(hl.Struct(r=i, c=j),
                    hl.Struct(r=i, c=j + 1))
        for i in range(a.n_block_rows)
        for j in range(a.n_block_cols)])
    return DNDArray(t)


class DNDArray:
    """An distributed n-dimensional array.

    Notes
    -----

    :class:`.DNDArray` makes extensive use of :meth:`.init`'s ``tmp_dir``
    parameter to write intermediate results. We advise you to regularly clean up
    that directory. If it is a bucket in Google Cloud Storage, you can use a
    retention policy to automatically clean it up
    """

    default_block_size = 4096
    fast_codec_spec = json.dumps({
        "name": "BlockingBufferSpec",
        "blockSize": 64 * 1024,
        "child": {
            "name": "LZ4FastBlockBufferSpec",
            "blockSize": 64 * 1024,
            "child": {
                "name": "StreamBlockBufferSpec"}}})

    @staticmethod
    def from_matrix_table(
            mt: MatrixTable,
            entry_field: str,
            *,
            n_partitions: Optional[int] = None,
            block_size: Optional[int] = None,
            sort_columns: bool = False
    ) -> 'DNDArray':
        if n_partitions is None:
            n_partitions = mt.n_partitions()
        if block_size is None:
            block_size = DNDArray.default_block_size
        if n_partitions == 0:
            assert mt.count_cols() == 0
            assert mt.count_rows() == 0
            t = range_table(0, 0)
            t = t.annotate(r=0, c=0, block=nd.array([]).reshape((0, 0)))
            t = t.select_globals(
                n_rows=0,
                n_cols=0,
                n_block_rows=0,
                n_block_cols=0,
                block_size=0)
            return DNDArray(t)

        assert 'r' not in mt.row
        assert 'c' not in mt.row
        assert 'block' not in mt.row

        n_rows, n_cols = mt.count()
        n_block_rows = (n_rows + block_size - 1) // block_size
        n_block_cols = (n_cols + block_size - 1) // block_size
        entries, cols, row_index, col_blocks = (Env.get_uid() for _ in range(4))

        if sort_columns:
            col_index = Env.get_uid()
            col_order = mt.add_col_index(col_index)
            col_order = col_order.key_cols_by().cols()
            col_order = col_order.select(key=col_order.row.select(*mt.col_key),
                                         index=col_order[col_index])
            col_order = col_order.collect(_localize=False)
            col_order = hl.sorted(col_order, key=lambda x: x.key)
            col_order = col_order['index'].collect()[0]
            mt = mt.choose_cols(col_order)
        else:
            col_keys = mt.col_key.collect(_localize=False)
            out_of_order = hl.range(hl.len(col_keys) - 1).map(
                lambda i: col_keys[i] > col_keys[i + 1])
            out_of_order = out_of_order.collect()[0]
            if any(out_of_order):
                raise ValueError(
                    'from_matrix_table: columns are not in sorted order. You may request a '
                    'sort with sort_columns=True.')

        mt = (mt
              .select_globals()
              .select_rows()
              .select_cols()
              .add_row_index(row_index)
              .localize_entries(entries, cols))
        # FIXME: remove when ndarray support structs
        mt = mt.annotate(**{entries: mt[entries][entry_field]})
        mt = mt.annotate(
            **{col_blocks: hl.range(n_block_cols).map(
                lambda c: hl.struct(
                    c=c,
                    entries=mt[entries][(c * block_size):((c + 1) * block_size)]))}
        )
        mt = mt.explode(col_blocks)
        mt = mt.select(row_index, **mt[col_blocks])
        mt = mt.annotate(r=hl.int(mt[row_index] // block_size))
        mt = mt.key_by(mt.r, mt.c)
        mt = mt.group_by(mt.r, mt.c).aggregate(
            entries=hl.sorted(
                hl.agg.collect(hl.struct(row_index=mt[row_index], entries=mt.entries)),
                key=lambda x: x.row_index
            ).map(lambda x: x.entries))
        mt = mt.select(block=hl.nd.array(mt.entries))
        mt = mt.select_globals(
            n_rows=n_rows,
            n_cols=n_cols,
            n_block_rows=n_block_rows,
            n_block_cols=n_block_cols,
            block_size=block_size)
        fname = new_temp_file()
        mt = mt.key_by(mt.r, mt.c)
        mt.write(fname, _codec_spec=DNDArray.fast_codec_spec)
        t = hl.read_table(fname, _intervals=[
            hl.Interval(hl.Struct(r=i, c=j),
                        hl.Struct(r=i, c=j + 1))
            for i in range(n_block_rows)
            for j in range(n_block_cols)])
        return DNDArray(t)

    def __init__(self, t: Table) -> 'DNDArray':
        assert 'r' in t.row
        assert 'c' in t.row
        assert 'block' in t.row

        self.m: Table = t

        dimensions = t.globals.collect()[0]
        self.n_rows: int = dimensions.n_rows
        self.n_cols: int = dimensions.n_cols
        self.n_block_rows: int = dimensions.n_block_rows
        self.n_block_cols: int = dimensions.n_block_cols
        self.block_size: int = dimensions.block_size

        assert self.n_block_rows == (self.n_rows + self.block_size - 1) // self.block_size
        assert self.n_block_cols == (self.n_cols + self.block_size - 1) // self.block_size

    def count_blocks(self) -> Tuple[int, int]:
        return (self.n_block_cols, self.n_block_rows)

    def transpose(self) -> 'DNDArray':
        return self.T

    @property
    def T(self) -> 'DNDArray':
        m = self.m
        m = m.annotate(block=m.block.T)
        dimensions = m.globals.collect()[0]
        m = m.select_globals(
            n_rows=dimensions.n_cols,
            n_cols=dimensions.n_rows,
            n_block_rows=dimensions.n_block_cols,
            n_block_cols=dimensions.n_block_rows,
            block_size=dimensions.block_size)
        m = m._key_by_assert_sorted('c', 'r')
        m = m.rename({'r': 'c', 'c': 'r'})
        return DNDArray(m)

    def _block_inner_product(self,
                             right: 'DNDArray',
                             block_product: Callable[[Expression, Expression], Expression],
                             block_aggregate: Callable[[Expression], Expression]
                             ) -> 'DNDArray':
        left = self
        assert left.block_size == right.block_size
        assert left.n_cols == right.n_rows
        assert left.n_block_cols == right.n_block_rows

        n_rows = left.n_rows
        n_cols = right.n_cols
        block_size = left.block_size

        n_block_rows = left.n_block_rows
        n_block_inner = left.n_block_cols
        n_block_cols = right.n_block_cols
        n_multiplies = n_block_rows * n_block_cols * n_block_inner

        o = hl.utils.range_table(n_multiplies, n_partitions=n_multiplies)
        o = o.key_by(
            r=o.idx // (n_block_cols * n_block_inner),
            c=(o.idx % (n_block_cols * n_block_inner)) // n_block_inner,
            k=o.idx % n_block_inner
        ).select()
        o = o._key_by_assert_sorted('r', 'c', 'k')
        o = o._key_by_assert_sorted('r', 'k', 'c')
        o = o.annotate(left=left.m[o.r, o.k].block)
        o = o._key_by_assert_sorted('k', 'c', 'r')
        o = o.annotate(right=right.m[o.k, o.c].block)

        o = o.annotate(product=block_product(o.left, o.right))
        o = o._key_by_assert_sorted('r', 'c', 'k')
        o = o._key_by_assert_sorted('r', 'c')

        import hail.methods.misc as misc
        misc.require_key(o, 'collect_by_key')
        import hail.ir as ir

        o = Table(ir.TableAggregateByKey(
            o._tir,
            hl.struct(block=block_aggregate(o.product))._ir))
        o = o.select('block')
        o = o.select_globals(
            n_rows=n_rows,
            n_cols=n_cols,
            n_block_rows=n_block_rows,
            n_block_cols=n_block_cols,
            block_size=block_size)
        return DNDArray(o)

    def __matmul__(self, right: 'DNDArray') -> 'DNDArray':
        return self._block_inner_product(right,
                                         lambda left, right: left @ right,
                                         hl.agg.ndarray_sum)

    def inner_product(self,
                      right: 'DNDArray',
                      multiply: Callable[[Expression, Expression], Expression],
                      add: Callable[[Expression, Expression], Expression],
                      zero: Expression,
                      add_as_an_aggregator: Callable[[Expression], Expression]
                      ) -> 'DNDArray':
        def block_product(left, right):
            n_rows, n_inner = left.shape
            _, n_cols = right.shape

            def compute_element(absolute):
                return hl.rbind(
                    absolute % n_rows,
                    absolute // n_rows,
                    lambda row, col: hl.range(hl.int(n_inner)).map(
                        lambda inner: multiply(left[row, inner], right[inner, col])
                    ).fold(add, zero))

            return hl.struct(
                shape=(left.shape[0], right.shape[1]),
                block=hl.range(hl.int(n_rows * n_cols)).map(compute_element))

        def block_aggregate(prod):
            shape = prod.shape
            block = prod.block
            return hl.nd.from_column_major(
                hl.agg.array_agg(add_as_an_aggregator, block),
                hl.agg.take(shape, 1)[0])

        return self._block_inner_product(right, block_product, block_aggregate)

    def _block_map(self, fun: Callable[[Expression], Expression]) -> 'DNDArray':
        o = self.m
        o = o.annotate(block=fun(o.block))
        return DNDArray(o)

    def _block_pairwise_map(self,
                            right: 'DNDArray',
                            fun: Callable[[Expression, Expression], Expression]
                            ) -> 'DNDArray':
        left = self
        assert left.block_size == right.block_size
        assert left.n_rows == right.n_rows
        assert left.n_cols == right.n_cols
        assert left.n_block_rows == right.n_block_rows
        assert left.n_block_cols == right.n_block_cols

        o = left.m
        o = o.annotate(
            block=fun(o.block, right.m[o.r, o.c].block))

        return DNDArray(o)

    def __add__(self, right: Any) -> 'DNDArray':
        assert not isinstance(right, Table)
        assert not isinstance(right, MatrixTable)

        if not isinstance(right, DNDArray):
            return self._block_map(lambda left: left + right)

        return self._block_pairwise_map(right, lambda left, right: left + right)

    def __radd__(self, left: Any) -> 'DNDArray':
        assert not isinstance(left, Table)
        assert not isinstance(left, MatrixTable)
        assert not isinstance(left, DNDArray)

        return self._block_map(lambda right: left + right)

    def __sub__(self, right: Any) -> 'DNDArray':
        assert not isinstance(right, Table)
        assert not isinstance(right, MatrixTable)

        if not isinstance(right, DNDArray):
            return self._block_map(lambda left: left - right)

        return self._block_pairwise_map(right, lambda left, right: left - right)

    def __rsub__(self, left: Any) -> 'DNDArray':
        assert not isinstance(left, Table)
        assert not isinstance(left, MatrixTable)
        assert not isinstance(left, DNDArray)

        return self._block_map(lambda right: left - right)

    def __mul__(self, right: Any) -> 'DNDArray':
        assert not isinstance(right, Table)
        assert not isinstance(right, MatrixTable)

        if not isinstance(right, DNDArray):
            return self._block_map(lambda left: left * right)

        return self._block_pairwise_map(right, lambda left, right: left * right)

    def __rmul__(self, left: Any) -> 'DNDArray':
        assert not isinstance(left, Table)
        assert not isinstance(left, MatrixTable)
        assert not isinstance(left, DNDArray)

        return self._block_map(lambda right: left * right)

    def write(self, *args, **kwargs) -> 'DNDArray':
        return self.m.write(*args, **kwargs)

    def checkpoint(self, fname, *, overwrite=False) -> 'DNDArray':
        self.write(fname, _codec_spec=DNDArray.fast_codec_spec, overwrite=overwrite)
        return read(fname)

    def _force_count_blocks(self) -> int:
        return self.m._force_count()

    def show(self, *args, **kwargs):
        return self.m.show(*args, **kwargs)

    def collect(self) -> np.array:
        blocks = self.m.collect()
        result = [[None
                   for _ in range(self.n_block_cols)]
                  for _ in range(self.n_block_rows)]
        for block in blocks:
            result[block.r][block.c] = block.block

        return np.concatenate(
            [np.concatenate(result[x], axis=1) for x in range(self.n_block_rows)],
            axis=0
        )
