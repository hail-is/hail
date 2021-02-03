--------------------
Expressions Overview
--------------------

What is an Expression?
======================

Hail's expressions are lazy representations of data.

Each data type in Hail has its own :class:`.Expression` class. For example, an
:class:`.Int32Expression` represents a 32-bit integer, and a
:class:`.BooleanExpression` represents a boolean value of True or False.

>>> hl.int32(5)
<Int32Expression of type int32>

>>> hl.bool(True)
<BooleanExpression of type bool>

Expressions can be combined with operations to form new expressions. Much like
you would add two integers in Python, you can also add two
:class:`.Int32Expression` objects in Hail.

>>> hl.int32(5) + hl.int32(6)
<Int32Expression of type int32>

The result of adding two :class:`.Int32Expression` objects is another
:class:`.Int32Expression` object.

We say Hail's expressions are **lazy**, because they are not evaluated until the
result of the expression is needed. Let's explore what this means by comparing a
Python expression to a Hail expression.

In Python, an expression such as ``5+6`` will be immediately evaluated. If you
enter this expression into Python, you'll see the result, ``11``, right away.

>>> x = 5
>>> y = 6
>>> z = x + y
>>> z
11

The equivalent code written with Hail's expressions would look like:

>>> x = hl.int32(5)
>>> y = hl.int32(6)
>>> z = x + y
>>> z
<Int32Expression of type int32>

Notice that when we enter ``z``, we don't see the result, ``11``, like we
did with Python. Hail is not running Python code on your data. Instead, Hail is
keeping track of the computations applied to your data, compiling these
computations into native code, and running them in parallel.

The result of the expression is computed only when it is needed. So ``z`` is
an expression representing the computation of ``x + y``, but not the actual
value.

To peek at the value of this computation, there are two options:
:func:`~hail.expr.eval`, which returns a Python value, and :meth:`.Expression.show`,
which prints a human-readable representation of an expression.

    >>> hl.eval(z)
    11
    >>> z.show()
    +--------+
    | <expr> |
    +--------+
    |  int32 |
    +--------+
    |     11 |
    +--------+


Hail's expressions are especially important for interacting with fields in
tables and matrix tables. Throughout Hail documentation and tutorials, you will
see code like this:

    >>> ht2 = ht.annotate(C4 = ht.C3 + 3 * ht.C2 ** 2)

This snippet of code is adding a field, ``C4``, to a table, ``ht``, and
returning the result as a new table, ``ht2``. The code passed to the
:meth:`.Table.annotate` method is an expression that references the fields
``C3`` and ``C2`` in ``ht``.

Notice that ``3`` and ``2`` are not wrapped in constructor functions like
``hl.int32(3)``. In the same way that Hail expressions can be combined together
via operations like addition and multiplication, they can also be combined with
Python objects.

For example, we can add a Python :obj:`int` to an :class:`.Int32Expression`.

    >>> x + 3
    <Int32Expression of type int32>

Addition is commutative, so we can also add an :class:`.Int32Expression` to an
:obj:`int`.

    >>> 3 + x
    <Int32Expression of type int32>

Note that Hail expressions cannot be used in other modules, like :mod:`numpy`
or :mod:`scipy`.

Hail has many subclasses of :class:`.Expression` -- one for each Hail type. Each
subclass has its own constructor method. For example, if we have a list of Python
integers, we can convert this to a Hail :class:`.ArrayNumericExpression` with
:func:`~hail.expr.functions.array`:

    >>> a = hl.array([1, 2, -3, 0, 5])
    >>> a
    <ArrayNumericExpression of type array<int32>>

:class:`.Expression` objects keep track of their data type, which is
why we can see that ``a`` is of type ``array<int32>`` in the output above. An
expression's type can also be accessed with :meth:`.Expression.dtype`.

    >>> a.dtype
    dtype('array<int32>')

Hail arrays can be indexed and sliced like Python lists or :mod:`numpy` arrays:

    >>> a[1]
    <Int32Expression of type int32>

    >>> a[1:-1]
    <ArrayNumericExpression of type array<int32>>

In addition to constructor methods like :func:`~hail.expr.functions.array` and :func:`.bool`,
Hail expressions can also be constructed with the :func:`.literal` method,
which will impute the type of of the expression.

    >>> hl.literal([0,1,2])
    <ArrayNumericExpression of type array<int32>>

Boolean Logic
=============

Unlike Python, a Hail :class:`.BooleanExpression` cannot be used with the Python
keywords ``and``, ``or``, and ``not``. The Hail substitutes are ``&``, ``|``,
and ``~``.

    >>> s1 = hl.int32(3) == 4
    >>> s2 = hl.int32(3) != 4

    >>> s1 & s2
    <BooleanExpression of type bool>

    >>> s1 | s2
    <BooleanExpression of type bool>

    >>> ~s1
    <BooleanExpression of type bool>

Remember that you can use :func:`~hail.expr.eval`: to evaluate the expression.

    >>> hl.eval(~s1)
    True

.. caution::

    The operator precedence of ``&`` and ``|`` is different from ``and`` and
    ``or``. You will need parentheses around expressions like this:

    >>> (x == 3) & (x != 4)

Conditional Expressions
=======================

If/Else Statements
~~~~~~~~~~~~~~~~~~

Python ``if`` / ``else`` statements do not work with Hail expressions. Instead,
you must use the :func:`.if_else`, :func:`.case`, and :func:`.switch` functions.

A conditional expression has three components: the condition to evaluate, the
consequent value to return if the condition is ``True``, and the alternate to
return if the condition is ``False``. For example:

.. code-block:: python

    if (x > 0):
        return 1
    else:
        return 0

In the above conditional, the condition is ``x > 0``, the consequent is ``1``,
and the alternate is ``0``.

Here is the Hail expression equivalent with :func:`.if_else`:

    >>> hl.if_else(x > 0, 1, 0)
     <Int32Expression of type int32>

This example returns an :class:`.Int32Expression` which can be used in more
computations. We can add the conditional expression to our array ``a`` from
earlier:

    >>> a + hl.if_else(x > 0, 1, 0)
    <ArrayNumericExpression of type array<int32>>

Case Statements
~~~~~~~~~~~~~~~

More complicated conditional statements can be constructed with :func:`.case`.
For example, we might want to return ``1`` if ``x < -1``, ``2`` if
``-1 <= x <= 2`` and ``3`` if ``x > 2``.

    >>> (hl.case()
    ...   .when(x < -1, 1)
    ...   .when((x >= -1) & (x <= 2), 2)
    ...   .when(x > 2, 3)
    ...   .or_missing())
    <Int32Expression of type int32>

Notice that this expression ends with a call to :meth:`~hail.expr.builders.CaseBuilder.or_missing`,
which means that if none of the conditions are met, a missing value is returned.

Cases started with :func:`.case` can end with a call to
:meth:`~hail.expr.builders.CaseBuilder.or_missing`, :meth:`~hail.expr.builders.CaseBuilder.default`, or
:meth:`~hail.expr.builders.CaseBuilder.or_error`, depending on what you want to happen if none
of the *when* clauses are met.

It's important to note that missingness propagates up in Hail, so if the value
of the discriminant in a case statement is missing, then the result will be
missing as well.

>>> y = hl.missing(hl.tint32)
>>> result = hl.case().when(y > 0, 1).default(-1)
>>> hl.eval(result)

The value of ``result`` will be missing, not ``1`` or ``-1``, because the
discriminant, ``y``, is missing.

Switch Statements
~~~~~~~~~~~~~~~~~

Finally, Hail has the :func:`.switch` function to build a conditional tree based
on the value of an expression. In the example below, ``csq`` is a
:class:`.StringExpression` representing the functional consequence of a
mutation. If ``csq`` does not match one of the cases specified by
:meth:`~hail.expr.builders.SwitchBuilder.when`, it is set to missing with
:meth:`~hail.expr.builders.SwitchBuilder.or_missing`. Other switch statements are documented in the
:class:`.SwitchBuilder` class.

    >>> csq = hl.str('nonsense')

    >>> (hl.switch(csq)
    ...    .when("synonymous", False)
    ...    .when("intron", False)
    ...    .when("nonsense", True)
    ...    .when("indel", True)
    ...    .or_missing())
    <BooleanExpression of type bool>

As with case statements, missingness will propagate up through a switch
statement. If we changed the value of ``csq`` to the missing value
``hl.missing(hl.tstr)``, then the result of the switch statement above would also
be missing.

Missingness
===========

In Hail, all expressions can be missing. An expression representing a missing
value of a given type can be generated with the :func:`.null` function, which
takes the type as its single argument.

An example of generating a :class:`.Float64Expression` that is missing is:

    >>> hl.missing('float64')
    <Float64Expression of type float64>

These can be used with conditional statements to set values to missing if they
don't satisfy a condition:

    >>> hl.if_else(x > 2.0, x, hl.missing(hl.tfloat))
    <Float64Expression of type float64>

The Python representation of a missing value is ``None``. For example, if
we define ``cnull`` to be a missing value with type :obj:`.tcall`, calling
the method `is_het` will return ``None`` and not ``False``.

    >>> cnull = hl.missing('call')
    >>> hl.eval(cnull.is_het())
    None

Functions
=========

In addition to the methods exposed on each :class:`.Expression`, Hail also has
numerous functions that can be applied to expressions, which also return an
expression.

Take a look at the :ref:`sec-functions` page for full documentation.
