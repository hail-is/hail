.. _hail_types:

Types
=====

.. toctree::
    :maxdepth: 2

.. currentmodule:: hail.expr.types

Fields and expressions in Hail have types. Throughout the documentation, you
will find type descriptions like ``array<str>`` or :class:`.tlocus`. It is
generally more important to know how to use expressions of various types than to
know how to manipulate the types themselves, but some operations like
:func:`.missing` require type arguments.

In Python, ``5`` is of type :obj:`int` while ``"hello"`` is of type :obj:`str`.
Python is a dynamically-typed language, meaning that a function like:

    >>> def add_x_and_y(x, y):
    ...     return x + y

...can be called on any two objects which can be added, like numbers, strings, or
:mod:`numpy` arrays.

Types are very important in Hail, because the fields of :class:`.Table` and
:class:`.MatrixTable` objects have data types.

Primitive types
~~~~~~~~~~~~~~~

Hail's primitive data types for boolean, numeric and string objects are:

.. autosummary::
    :nosignatures:

    tint
    tint32
    tint64
    tfloat
    tfloat32
    tfloat64
    tstr
    tbool

Container types
~~~~~~~~~~~~~~~

Hail's container types are:

 - :class:`.tarray` - Ordered collection of homogenous objects.
 - :class:`.tndarray` - Ordered n-dimensional arrays of homogenous objects.
 - :class:`.tset` - Unordered collection of distinct homogenous objects.
 - :class:`.tdict` - Key-value map. Keys and values are both homogenous.
 - :class:`.ttuple` - Tuple of heterogeneous values.
 - :class:`.tstruct` - Structure containing named fields, each with its own
   type.

.. autosummary::
    :nosignatures:

    tarray
    tndarray
    tset
    tdict
    ttuple
    tinterval
    tstruct

Genetics types
~~~~~~~~~~~~~~

Hail has two genetics-specific types:

.. autosummary::
    :nosignatures:

    tlocus
    tcall

When to work with types
~~~~~~~~~~~~~~~~~~~~~~~

In general, you won't need to mention types explicitly.

There are a few situations where you may want to specify types explicitly:

- To specify column types in :func:`~.import_table` if the `impute` flag does not
  infer the type you want.
- When converting a Python value to a Hail expression with :func:`.literal`,
  if you don't wish to rely on the inferred type.
- With functions like :func:`.missing` and :func:`.empty_array`.

Viewing an object's type
~~~~~~~~~~~~~~~~~~~~~~~~

Hail objects have a ``dtype`` field that will print their type.

    >>> hl.rand_norm().dtype
    dtype('float64')

Printing the representation of a Hail expression will also show the type:

    >>> hl.rand_norm()
    <Float64Expression of type float64>

We can see that ``hl.rand_norm()`` is of type :py:data:`.tfloat64`, but what does
Expression mean?
Each data type in Hail is represented by its own Expression class. Data of
type :py:data:`.tfloat64` is represented by an :class:`.Float64Expression`. Data
of type :class:`.tstruct` is represented by a :class:`.StructExpression`.

Collection Types
~~~~~~~~~~~~~~~~

Hail's collection types (arrays, ndarrays, sets, and dicts) have homogenous elements,
meaning that all values in the collection must be of the same type. Python allows mixed
collections: ``['1', 2, 3.0]`` is a valid Python list. However, Hail arrays
cannot contain both :py:data:`.tstr` and :py:data:`.tint32` values. Likewise,
the :obj:`dict` ``{'a': 1, 2: 'b'}`` is a valid Python
dictionary, but a Hail dictionary cannot contain keys of different types.
An example of a valid dictionary in Hail is ``{'a': 1, 'b': 2}``, where the keys are all
strings and the values are all integers. The type of this dictionary would be
``dict<str, int32>``.

Constructing types
~~~~~~~~~~~~~~~~~~

Constructing types can be done either by using the type objects and classes
(prefixed by "t") or by parsing from strings with :func:`.dtype`. As an example,
we will construct a :class:`.tstruct` with each option:

.. doctest::

    >>> t = hl.tstruct(a = hl.tint32, b = hl.tstr, c = hl.tarray(hl.tfloat64))
    >>> t
    dtype('struct{a: int32, b: str, c: array<float64>}')

    >>> t = hl.dtype('struct{a: int32, b: str, c: array<float64>}')
    >>> t
    dtype('struct{a: int32, b: str, c: array<float64>}')


Reference documentation
~~~~~~~~~~~~~~~~~~~~~~~

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
.. autoclass:: tndarray
.. autoclass:: tset
.. autoclass:: tdict
.. autoclass:: tstruct
.. autoclass:: ttuple
.. autodata:: hail.expr.types.tcall

.. autoclass:: tlocus

    .. autoattribute:: reference_genome

.. autoclass:: tinterval
