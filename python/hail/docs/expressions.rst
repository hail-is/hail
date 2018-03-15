Expressions
===========

.. toctree::
    :maxdepth: 2

.. currentmodule:: hail.expr.expressions.base_expression
.. autosummary::
    :nosignatures:

    Expression

.. currentmodule:: hail.expr.expressions.typed_expressions
.. autosummary::
    :nosignatures:

    ArrayExpression
    ArrayNumericExpression
    BooleanExpression
    CallExpression
    CollectionExpression
    DictExpression
    IntervalExpression
    LocusExpression
    NumericExpression
    Int32Expression
    Int64Expression
    Float32Expression
    Float64Expression
    SetExpression
    StringExpression
    StructExpression


.. autoclass:: hail.expr.expressions.base_expression.Expression
    :special-members: __eq__, __ne__

.. autoclass:: hail.expr.expressions.typed_expressions.ArrayExpression
    :show-inheritance:
    :special-members: __getitem__

.. autoclass:: hail.expr.expressions.typed_expressions.ArrayNumericExpression
    :show-inheritance:
    :special-members: __add__, __sub__, __mul__, __div__, __floordiv__, __pow__, __mod__, __neg__

.. autoclass:: hail.expr.expressions.typed_expressions.BooleanExpression
    :show-inheritance:
    :special-members: __and__, __or__, __invert__

.. autoclass:: hail.expr.expressions.typed_expressions.CallExpression
    :show-inheritance:
    :special-members: __getitem__

.. autoclass:: hail.expr.expressions.typed_expressions.CollectionExpression
    :show-inheritance:

.. autoclass:: hail.expr.expressions.typed_expressions.DictExpression
    :show-inheritance:
    :special-members: __getitem__

.. autoclass:: hail.expr.expressions.typed_expressions.IntervalExpression
    :show-inheritance:

.. autoclass:: hail.expr.expressions.typed_expressions.LocusExpression
    :show-inheritance:

.. autoclass:: hail.expr.expressions.typed_expressions.NumericExpression
    :show-inheritance:
    :special-members: __lt__, __le__, __gt__, __ge__, __neg__, __add__, __sub__, __mul__, __div__, __floordiv__, __mod__, __pow__

.. autoclass:: hail.expr.expressions.typed_expressions.Int32Expression
    :show-inheritance:

.. autoclass:: hail.expr.expressions.typed_expressions.Int64Expression
    :show-inheritance:

.. autoclass:: hail.expr.expressions.typed_expressions.Float32Expression
    :show-inheritance:

.. autoclass:: hail.expr.expressions.typed_expressions.Float64Expression
    :show-inheritance:

.. autoclass:: hail.expr.expressions.typed_expressions.SetExpression
    :show-inheritance:

.. autoclass:: hail.expr.expressions.typed_expressions.StringExpression
    :show-inheritance:
    :special-members: __getitem__, __add__

.. autoclass:: hail.expr.expressions.typed_expressions.StructExpression
    :show-inheritance:
    :special-members: __getitem__
