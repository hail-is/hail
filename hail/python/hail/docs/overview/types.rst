--------------
Types Overview
--------------

In Python, ``5`` is of type :obj:`int` while ``"hello"`` is of type :obj:`str`.
Python is a dynamically-typed language, meaning that a function like:

    >>> def add_x_and_y(x, y):
    ...     return x + y

can be called on any two objects which can be added, like numbers, strings, or
:mod:`numpy` arrays.

Types are very important in Hail, because the fields of :class:`.Table` and
:class:`.MatrixTable` objects have data types.

Hail's primitive data types for boolean, numeric and string objects are:

 - :py:data:`.tstr` - Text string.
 - :py:data:`.tbool` - Boolean (``True`` or ``False``) value.
 - :py:data:`.tint32` - 32-bit integer.
 - :py:data:`.tint64` - 64-bit integer.
 - :py:data:`.tfloat32` - 32-bit floating point number.
 - :py:data:`.tfloat64` - 64-bit floating point number.

Hail's container types are:

 - :class:`.tarray` - Ordered collection of homogenous objects.
 - :class:`.tndarray` - Ordered n-dimensional arrays of homogenous objects.
 - :class:`.tset` - Unordered collection of distinct homogenous objects.
 - :class:`.tdict` - Key-value map. Keys and values are both homogenous.
 - :class:`.ttuple` - Tuple of heterogeneous values.
 - :class:`.tstruct` - Structure containing named fields, each with its own
   type.

Hail also has a few genetics-specific types:

 - :py:data:`.tcall` - Genotype calls.
 - :class:`.tlocus` - Genomic locus, parameterized by reference genome.

When to work with types
~~~~~~~~~~~~~~~~~~~~~~~

In general, you won't need to mention types explicitly. Hail will
automatically impute the type of your data.

There are a few situations where you may want to specify types explicitly:

- To specify column types in :func:`import_table` if the imputed types
  do not match what you want.
- When converting a Python value to a Hail expression with :func:`.literal`,
  if you don't wish to rely on the imputed type.
- When using missing types via the :func:`.null` constructor.

Viewing an object's type
~~~~~~~~~~~~~~~~~~~~~~~~

Hail objects have a ``dtype`` field that will print their type.

    >>> hl.int32(3).dtype
    dtype('int32')

Entering just the object will also give you some type information.

    >>> hl.int32(3)
    <Int32Expression of type int32>

We can see that ``hl.int32(3)`` is of type :py:data:`.tint32`, but what does
Expression mean?
Each data type in Hail is represented by its own Expression class. Data of
type :py:data:`.tint32` is represented by an :class:`.Int32Expression`. Data
of type :class:`.tstruct` is represented by a :class:`.StructExpression`.

If you examine the type of a container object, such as a struct,
you'll notice that the struct expression's type also contains the subtypes
of the nested fields.

    >>> hl.struct(name='Hail', dob=2015)
    <StructExpression of type struct{name: str, dob: int32}>

    >>> hl.struct(name='Hail', dob=2015).dtype
    dtype('struct{name: str, dob: int32}')

Container Types
~~~~~~~~~~~~~~~

Hail's container types for arrays, sets, dicts, and tuples require homogenous collections,
meaning that all values in the collection must be of the same type. In contrast,
Python allows mixed collections, e.g. ``['1', 2, 3.0]`` is a valid Python list. A Hail array
could not contain both :py:data:`.tstr` and :py:data:`.tint32`
objects. Likewise, the :obj:`dict` ``{'a': 1, 2: 'b'}`` is a valid Python
dictionary, but a Hail dictionary cannot contain keys of different types.
An example of a valid dictionary in Hail is ``{'a': 1, 'b': 2}``, where the keys are all
strings and the values are all integers. The type of this dictionary would be
``dict<str, int32>``.

Structs
~~~~~~~

Hail's :class:`.tstruct` type is used to compose types together to form nested
structures. Structs can contain any combination of types. The :class:`.tstruct`
is an ordered mapping from field name to field type. Each field name must be unique.
So a struct ``hl.struct(name='Hail', dob=2015)`` has type ``dtype('struct{name: str, dob: int32}')``
and contains a mapping from ``name`` to a string field and from ``dob`` to integer fields.

Structs are very common in Hail. Consider:

>>> new_table = table1.annotate(table2_fields = table2[table1.key])

This snippet adds a field to ``table1`` called ``table2_fields``. In the new table,
``table2_fields`` will be a struct containing all the nested fields from ``table2``.
