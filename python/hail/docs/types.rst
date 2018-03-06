Types
=====

Fields and expressions in Hail have types. Throughout the documentation, you
will find type descriptions like ``array<str>`` or :class:`.tlocus`. It is
generally more important to know how to use expressions of various types than to
know how to manipulate the types themselves, but some operations like
:func:`.null` require type arguments.

Constructing types can be done either by using the type objects and classes
(prefixed by "t") or by parsing from strings. As an example, we will construct
a :class:`.tstruct` with each option:

.. doctest::

    >>> t = hl.tstruct(a = hl.tint32, b = hl.tstr, c = hl.tarray(hl.tfloat64))
    >>> t
    dtype('struct{a: int32, b: str, c: array<float64>}')

    >>> t = hl.dtype('struct{a: int32, b: str, c: array<float64>}')
    >>> t
    dtype('struct{a: int32, b: str, c: array<float64>}')

.. toctree::
    :maxdepth: 2

.. currentmodule:: hail.expr.types

.. autosummary::
    :nosignatures:

    HailType
    dtype
    tint
    tint32
    tint64
    tfloat
    tfloat32
    tfloat64
    tstr
    tbool
    tarray
    tset
    tdict
    ttuple
    tlocus
    tinterval
    tcall
    tstruct

.. autoclass:: HailType
.. autofunction:: dtype
.. autodata:: hail.expr.types.tint
.. autodata:: hail.expr.types.tint32
.. autodata:: hail.expr.types.tint64
.. autodata:: hail.expr.types.tfloat
.. autodata:: hail.expr.types.tfloat32
.. autodata:: hail.expr.types.tfloat64
.. autodata:: hail.expr.types.tstr
.. autodata:: hail.expr.types.tbool
.. autoclass:: tarray
.. autoclass:: tset
.. autoclass:: tdict
.. autoclass:: tstruct
.. autoclass:: ttuple
.. autodata:: hail.expr.types.tcall
.. autoclass:: tlocus
.. autoclass:: tinterval
