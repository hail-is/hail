-----
Types
-----

In Python, ``5`` is of type :obj:`int` while ``"hello"`` is of type :obj:`str`.
Python is a dynamically-typed language, meaning that a function like:

    >>> def add_x_and_y(x, y):
    ...     return x + y

can be called on any two objects which can be added, like numbers, strings, or
:mod:`numpy` arrays.

Types are very important in Hail, because the fields of :class:`.Table` and
:class:`.MatrixTable` objects have data types.

Hail has basic data types for numeric and string objects:

 - :py:data:`.tstr` - Text string.
 - :py:data:`.tbool` - Boolean (``True`` or ``False``) value.
 - :py:data:`.tint32` - 32-bit integer.
 - :py:data:`.tint64` - 64-bit integer.
 - :py:data:`.tfloat32` - 32-bit floating point number.
 - :py:data:`.tfloat64` - 64-bit floating point number.

Hail has genetics-specific types:

 - :py:data:`.tcall` - Genotype calls.
 - :class:`.tlocus` - Genomic locus, parameterized by reference genome.

Hail has container types:

 - :class:`.tarray` - Ordered collection of homogenous objects.
 - :class:`.tset` - Unordered collection of distinct homogenous objects.
 - :class:`.tdict` - Key-value map. Keys and values are both homogenous.
 - :class:`.ttuple` - Tuple of heterogeneous values.
 - :class:`.tstruct` - Structure containing named fields, each with its own
   type.

Homogenous collections are a change from standard Python collections.
While the list ``['1', 2, 3.0]`` is a perfectly valid Python list,
a Hail array could not contain both :py:data:`.tstr` and :py:data:`.tint32`
objects. Likewise, a the :obj:`dict` ``{'a': 1, 2: 'b'}`` is a valid Python
dictionary, but a Hail dictionary cannot contain keys of different types.
An example of a valid dictionary is ``{'a': 1, 'b': 2}``, where the keys are all
strings and the values are all integers. The type of this dictionary would be
``dict<str, int32>``.

The :class:`.tstruct` type is used to compose types together to form nested
structures. The :class:`.tstruct` is an ordered mapping from field name to field
type. Each field name must be unique.

