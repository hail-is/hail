import hail as hl
from hail.expr.expressions import expr_any, analyze
from hail.typecheck import typecheck


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
    hl.utils.range_table(1).filter(False).annotate_globals(expr=expr).write(path, overwrite=overwrite)


@typecheck(path=str)
def read_expression(path):
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
    return hl.read_table(path).index_globals().expr
