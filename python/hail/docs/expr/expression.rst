Expressions
===========

.. toctree::
    :maxdepth: 2

.. currentmodule:: hail.expr.expression

.. autosummary::
    :nosignatures:

    Expression
    ArrayExpression
    ArrayNumericExpression
    AtomicExpression
    BooleanExpression
    CallExpression
    CollectionExpression
    DictExpression
    IntervalExpression
    LocusExpression
    NumericExpression
    SetExpression
    StringExpression
    StructExpression


.. autoclass:: Expression
    :special-members: __eq__, __ne__

.. autoclass:: ArrayExpression
    :show-inheritance:
    :special-members: __getitem__

.. autoclass:: ArrayNumericExpression
    :show-inheritance:
    :special-members: __add__, __sub__, __mul__, __div__, __floordiv__, __pow__, __mod__

.. autoclass:: AtomicExpression
    :show-inheritance:

.. autoclass:: BooleanExpression
    :show-inheritance:
    :special-members: __and__, __or__, __invert__

.. autoclass:: CallExpression
    :show-inheritance:
    :special-members: __getitem__

.. autoclass:: CollectionExpression
    :show-inheritance:

.. autoclass:: DictExpression
    :show-inheritance:
    :special-members: __getitem__

.. autoclass:: IntervalExpression
    :show-inheritance:

.. autoclass:: LocusExpression
    :show-inheritance:

.. autoclass:: NumericExpression
    :show-inheritance:
    :special-members: __lt__, __le__, __gt__, __ge__, __neg__, __add__, __sub__, __mul__, __div__, __floordiv__, __mod__, __pow__

.. autoclass:: SetExpression
    :show-inheritance:

.. autoclass:: StringExpression
    :show-inheritance:
    :special-members: __getitem__, __add__

.. autoclass:: StructExpression
    :show-inheritance:
    :special-members: __getitem__
