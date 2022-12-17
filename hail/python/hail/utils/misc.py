from typing import Optional

import json

import hail
import hail as hl
from hail.typecheck import typecheck, nullable


@typecheck(n_rows=int, n_cols=int, n_partitions=nullable(int))
def range_matrix_table(n_rows, n_cols, n_partitions=None) -> 'hail.MatrixTable':
    """Construct a matrix table with row and column indices and no entry fields.

    Examples
    --------

    >>> range_ds = hl.utils.range_matrix_table(n_rows=100, n_cols=10)

    >>> range_ds.count_rows()
    100

    >>> range_ds.count_cols()
    10

    Notes
    -----
    The resulting matrix table contains the following fields:

     - `row_idx` (:py:data:`.tint32`) - Row index (row key).
     - `col_idx` (:py:data:`.tint32`) - Column index (column key).

    It contains no entry fields.

    This method is meant for testing and learning, and is not optimized for
    production performance.

    Parameters
    ----------
    n_rows : :obj:`int`
        Number of rows.
    n_cols : :obj:`int`
        Number of columns.
    n_partitions : int, optional
        Number of partitions (uses Spark default parallelism if None).

    Returns
    -------
    :class:`.MatrixTable`
    """
    check_nonnegative_and_in_range('range_matrix_table', 'n_rows', n_rows)
    check_nonnegative_and_in_range('range_matrix_table', 'n_cols', n_cols)
    if n_partitions is not None:
        check_positive_and_in_range('range_matrix_table', 'n_partitions', n_partitions)
    return hail.MatrixTable(hail.ir.MatrixRead(
        hail.ir.MatrixRangeReader(n_rows, n_cols, n_partitions),
        _assert_type=hl.tmatrix(
            hl.tstruct(),
            hl.tstruct(col_idx=hl.tint32),
            ['col_idx'],
            hl.tstruct(row_idx=hl.tint32),
            ['row_idx'],
            hl.tstruct()
        )
    ))


@typecheck(n=int, n_partitions=nullable(int))
def range_table(n: int,
                n_partitions: Optional[int] = None
                ) -> 'hail.Table':
    """Construct a table with the row index and no other fields.

    Examples
    --------

    >>> df = hl.utils.range_table(100)

    >>> df.count()
    100

    Notes
    -----
    The resulting table contains one field:

     - `idx` (:py:data:`.tint32`) - Row index (key).

    This method is meant for testing and learning, and is not optimized for
    production performance.

    Parameters
    ----------
    n : int
        Number of rows.
    n_partitions : int, optional
        Number of partitions (uses Spark default parallelism if None).

    Returns
    -------
    :class:`.Table`
    """
    check_nonnegative_and_in_range('range_table', 'n', n)
    if n_partitions is not None:
        check_positive_and_in_range('range_table', 'n_partitions', n_partitions)

    return hail.Table(hail.ir.TableRange(n, n_partitions))


def check_positive_and_in_range(caller, name, value):
    if value <= 0:
        raise ValueError(f"'{caller}': parameter '{name}' must be positive, found {value}")
    elif value > hail.tint32.max_value:
        raise ValueError(f"'{caller}': parameter '{name}' must be less than or equal to {hail.tint32.max_value}, "
                         f"found {value}")


def check_nonnegative_and_in_range(caller, name, value):
    if value < 0:
        raise ValueError(f"'{caller}': parameter '{name}' must be non-negative, found {value}")
    elif value > hail.tint32.max_value:
        raise ValueError(f"'{caller}': parameter '{name}' must be less than or equal to {hail.tint32.max_value}, "
                         f"found {value}")


def _dumps_partitions(partitions, row_key_type):
    parts_type = partitions.dtype
    if not (isinstance(parts_type, hl.tarray)
            and isinstance(parts_type.element_type, hl.tinterval)):
        raise ValueError(f'partitions type invalid: {parts_type} must be array of intervals')

    point_type = parts_type.element_type.point_type

    f1, t1 = next(iter(row_key_type.items()))
    if point_type == t1:
        partitions = hl.map(lambda x: hl.interval(
            start=hl.struct(**{f1: x.start}),
            end=hl.struct(**{f1: x.end}),
            includes_start=x.includes_start,
            includes_end=x.includes_end),
            partitions)
    else:
        if not isinstance(point_type, hl.tstruct):
            raise ValueError(f'partitions has wrong type: {point_type} must be struct or type of first row key field')
        if not point_type._is_prefix_of(row_key_type):
            raise ValueError(f'partitions type invalid: {point_type} must be prefix of {row_key_type}')

    s = json.dumps(partitions.dtype._convert_to_json(hl.eval(partitions)))
    return s, partitions.dtype


def divide_null(num, denom):
    from hail.expr.expressions.base_expression import unify_types_limited
    from hail.expr import missing, if_else
    typ = unify_types_limited(num.dtype, denom.dtype)
    assert typ is not None
    return if_else(denom != 0, num / denom, missing(typ))
