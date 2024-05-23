import hail as hl
from hail.expr.expressions import expr_any, analyze
from hail.expr.types import hail_type
from hail.expr.table_type import ttable
from hail.typecheck import typecheck, nullable


@typecheck(expr=expr_any, path=str, overwrite=bool)
def write_expression(expr, path, overwrite=False):
    """Write an Expression.

    In the same vein as Python's pickle, write out an expression
    that does not have a source (such as one that comes from
    Table.aggregate with _localize=False).

    Example
    -------
    >>> ht = hl.utils.range_table(100).annotate(x=hl.rand_norm())
    >>> mean_norm = ht.aggregate(hl.agg.mean(ht.x), _localize=False)
    >>> mean_norm
    >>> hl.eval(mean_norm)
    >>> hl.experimental.write_expression(mean_norm, 'output/expression.he')

    Parameters
    ----------

    expr : :class:`~.Expression`
        Expression to write.
    path : :class:`str`
        Path to which to write expression.
        Suggested extension: .he (hail expression).
    overwrite : :obj:`bool`
        If ``True``, overwrite an existing file at the destination.

    Returns
    -------
    None
    """
    source = expr._indices.source
    if source is not None:
        analyze('write_expression.expr', expr, source._global_indices)
        source = source.select_globals(__expr=expr)
        expr = source.index_globals().__expr
    hl.utils.range_table(1).filter(False).key_by().drop('idx').annotate_globals(expr=expr).write(
        path, overwrite=overwrite
    )


@typecheck(path=str, _assert_type=nullable(hail_type))
def read_expression(path, _assert_type=None):
    """Read an :class:`~.Expression` written with :func:`.experimental.write_expression`.

    Example
    -------
    >>> hl.experimental.write_expression(hl.array([1, 2]), 'output/test_expression.he')
    >>> expression = hl.experimental.read_expression('output/test_expression.he')
    >>> hl.eval(expression)

    Parameters
    ----------

    path : :class:`str`
        File to read.

    Returns
    -------
    :class:`~.Expression`
    """
    _assert_table_type = None
    _load_refs = True
    if _assert_type:
        _assert_table_type = ttable(hl.tstruct(expr=_assert_type), row_type=hl.tstruct(), row_key=[])
        _load_refs = False
    return hl.read_table(path, _assert_type=_assert_table_type, _load_refs=_load_refs).index_globals().expr
