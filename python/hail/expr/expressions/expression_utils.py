from hail.utils import warn, error
from .indices import *
from ..expressions import Expression, Aggregable, ExpressionException, expr_any
from typing import *


@typecheck(caller=str,
           expr=Expression,
           expected_indices=Indices,
           aggregation_axes=setof(str))
def analyze(caller: str, expr: Expression, expected_indices: Indices, aggregation_axes: Set = set()):
    indices = expr._indices
    source = indices.source
    axes = indices.axes
    aggregations = expr._aggregations

    warnings = []
    errors = []

    expected_source = expected_indices.source
    expected_axes = expected_indices.axes

    if source is not None and source is not expected_source:
        errors.append(
            ExpressionException('{} expects an expression from source {}, found expression derived from {}'.format(
                caller, expected_source, source
            )))

    # check for stray indices by subtracting expected axes from observed
    unexpected_axes = axes - expected_axes

    if unexpected_axes:
        # one or more out-of-scope fields
        refs = get_refs(expr)
        bad_refs = []
        for name, inds in refs.items():
            bad_axes = inds.axes.intersection(unexpected_axes)
            if bad_axes:
                bad_refs.append((name, inds))

        assert len(bad_refs) > 0
        errors.append(ExpressionException(
            "scope violation: '{caller}' expects an expression indexed by {expected}"
            "\n    Found indices {axes}, with unexpected indices {stray}. Invalid fields:{fields}{agg}".format(
                caller=caller,
                expected=list(expected_axes),
                axes=list(indices.axes),
                stray=list(unexpected_axes),
                fields=''.join("\n        '{}' (indices {})".format(name, list(inds.axes)) for name, inds in bad_refs),
                agg='' if (unexpected_axes - aggregation_axes) else
                "\n    '{}' supports aggregation over axes {}, "
                "so these fields may appear inside an aggregator function.".format(caller, list(aggregation_axes))
            )))

    if aggregations:
        if aggregation_axes:

            # the expected axes of aggregated expressions are the expected axes + axes aggregated over
            expected_agg_axes = expected_axes.union(aggregation_axes)

            for agg in aggregations:
                assert isinstance(agg, Aggregation)
                refs = get_refs(*agg.exprs)
                agg_indices = agg.indices
                agg_axes = agg_indices.axes
                if agg_indices.source is not None and agg_indices.source is not expected_source:
                    errors.append(
                        ExpressionException(
                            'Expected an expression from source {}, found expression derived from {}'
                            '\n    Invalid fields: [{}]'.format(
                                expected_source, source, ', '.join("'{}'".format(name) for name in refs)
                            )))

                # check for stray indices
                unexpected_agg_axes = agg_axes - expected_agg_axes
                if unexpected_agg_axes:
                    # one or more out-of-scope fields
                    bad_refs = []
                    for name, inds in refs.items():
                        bad_axes = inds.axes.intersection(unexpected_agg_axes)
                        if bad_axes:
                            bad_refs.append((name, inds))

                    assert len(bad_refs) > 0

                    errors.append(ExpressionException(
                        "scope violation: '{caller}' supports aggregation over indices {expected}"
                        "\n    Found indices {axes}, with unexpected indices {stray}. Invalid fields:{fields}".format(
                            caller=caller,
                            expected=list(aggregation_axes),
                            axes=list(agg_axes),
                            stray=list(unexpected_agg_axes),
                            fields=''.join("\n        '{}' (indices {})".format(
                                name, list(inds.axes)) for name, inds in bad_refs)
                        )
                    ))
        else:
            errors.append(ExpressionException("'{}' does not support aggregation".format(caller)))

    for w in warnings:
        warn('{}'.format(w.msg))
    if errors:
        for e in errors:
            error('{}'.format(e.msg))
        raise errors[0]


@typecheck(expression=expr_any)
def eval_expr(expression):
    """Evaluate a Hail expression, returning the result.

    This method is extremely useful for learning about Hail expressions and understanding
    how to compose them.

    The expression must have no indices, but can refer to the globals
    of a :class:`.hail.Table` or :class:`.hail.MatrixTable`.

    Examples
    --------

    Evaluate a conditional:

    >>> x = 6
    >>> hl.eval_expr(hl.cond(x % 2 == 0, 'Even', 'Odd'))
    'Even'

    Parameters
    ----------
    expression : :class:`.Expression`
        Any expression, or a Python value that can be implicitly interpreted as an expression.

    Returns
    -------
    any
        Result of evaluating `expression`.
    """
    return eval_expr_typed(expression)[0]


@typecheck(expression=expr_any)
def eval_expr_typed(expression):
    """Evaluate a Hail expression, returning the result and the type of the result.

    This method is extremely useful for learning about Hail expressions and understanding
    how to compose them.

    The expression must have no indices, but can refer to the globals
    of a :class:`.hail.Table` or :class:`.hail.MatrixTable`.

    Examples
    --------
    Evaluate a conditional:

    >>> x = 6
    >>> hl.eval_expr_typed(hl.cond(x % 2 == 0, 'Even', 'Odd'))
    ('Odd', tstr)

    Parameters
    ----------
    expression : :class:`.Expression`
        Any expression, or a Python value that can be implicitly interpreted as an expression.

    Returns
    -------
    (any, :class:`.HailType`)
        Result of evaluating `expression`, and its type.

    """
    analyze('eval_expr_typed', expression, Indices(expression._indices.source))

    return expression.collect()[0], expression.dtype


def _get_refs(expr: Union[Expression, Aggregable], builder: Dict[str, Indices]) -> None:
    from ..expr_ast import Select, TopLevelReference

    for ast in expr._ast.search(
            lambda a: isinstance(a, Select)
                      and not a.name.startswith('__uid')
                      and isinstance(a.parent, TopLevelReference)):
        builder[ast.name] = ast.parent.indices


def extract_refs_by_indices(exprs, indices):
    """Returns a set of references in `exprs` with indices `indices`.

    Parameters
    ----------
    expr : Expression
    indices : Indices

    Returns
    -------
    Set[str]
    """
    s = set()
    for e in exprs:
        for name, inds in get_refs(e).items():
            if inds == indices:
                s.add(name)
    return s

def get_refs(*exprs: Union[Expression, Aggregable]) -> Dict[str, Indices]:
    builder = {}
    for e in exprs:
        _get_refs(e, builder)
    return builder


@typecheck(caller=str,
           expr=Expression)
def matrix_table_source(caller, expr):
    from hail import MatrixTable
    source = expr._indices.source
    if not isinstance(source, MatrixTable):
        raise ValueError(
            "{}: Expect an expression of 'MatrixTable', found {}".format(
                caller,
                "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'))
    return source


@typecheck(caller=str,
           expr=Expression)
def check_entry_indexed(caller, expr):
    if expr._indices.source is None:
        raise ExpressionException("{}: expression must be entry-indexed,"
                                  " found no indices (no source)")
    if expr._indices != expr._indices.source._entry_indices:
        raise ExpressionException("{}: expression must be entry-indexed,"
                                  " found indices {}".format(caller, list(expr._indices.axes)))

@typecheck(caller=str,
           expr=Expression)
def check_row_indexed(caller, expr):
    if expr._indices.source is None:
        raise ExpressionException("{}: expression must be row-indexed,"
                                  " found no indices (no source)")
    if expr._indices != expr._indices.source._row_indices:
        raise ExpressionException("{}: expression must be row-indexed,"
                                  " found indices {}".format(caller, list(expr._indices.axes)))
