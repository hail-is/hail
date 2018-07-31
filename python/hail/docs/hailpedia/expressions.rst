-----------
Expressions
-----------

The Python language allows users to specify their computations using expressions.
For example, a simple expression is ``5 + 6``. This will be evaluated and return
``11``. You can also assign expressions to variables and then add variable expressions
together such as ``x = 5; y = 6; x + y``.

Throughout Hail documentation and tutorials, you will see Python code like this:

    >>> ht2 = ht.annotate(C4 = ht.C3 + 3 * ht.C2 ** 2)

However, Hail is not running Python code on your data. Instead, Hail is keeping
track of the computations applied to your data, then compiling these computations
into native code and running them in parallel.

This happens using the :class:`.Expression` class. Hail expressions operate much
like Python objects of the same type: for example, an :class:`.Int32Expression`
can be used in arithmetic with other integers or expressions in much the same
way a Python :obj:`int` can. However, you will be unable to use these
expressions with other modules, like :mod:`numpy` or :mod:`scipy`.

:class:`.Expression` objects keep track of their data type. This can be accessed
with :meth:`.Expression.dtype`:

    >>> i = hl.int32(100)
    >>> i.dtype
    dtype('int32')

The Hail equivalent of the Python example above would be as follows:

    >>> x = hl.int32(5)
    >>> y = hl.int32(6)

We can print `x` in a Python interpreter and see that `x` is an :class:`.Int32Expression`.
This makes sense because `x`  is a Python :obj:`int`.

    >>> x
    <Int32Expression of type int32>

We can add two :class:`.Int32Expression` objects together just like with Python
:obj:`int` objects. ``x + y`` returns another :class:`.Int32Expression` representing
the computation of ``x + y`` and not an actual value.

    >>> z = x + y
    >>> z
    <Int32Expression of type int32>

To peek at the value of this computation, there are two options:
:meth:`.Expression.value`, which returns a Python value, and
:meth:`.Expression.show`, which prints a human-readable representation of an
expression.

    >>> z.value
    11
    >>> z.show()
    +--------+
    | <expr> |
    +--------+
    |  int32 |
    +--------+
    |     11 |
    +--------+

Expressions like to bring Python objects into the world of expressions as well.
For example, we can add a Python :obj:`int` to an :class:`.Int32Expression`.

    >>> x + 3
    <Int32Expression of type int32>

Addition is commutative, so we can also add an :class:`.Int32Expression` to an
:obj:`int`.

    >>> 3 + x
    <Int32Expression of type int32>

Hail has many subclasses of :class:`.Expression` -- one for each Hail type. Each
subclass defines possible methods and operations that can be applied. For example,
if we have a list of Python integers, we can convert this to a Hail
:class:`.ArrayNumericExpression` with either :func:`.array` or :func:`.literal`:

    >>> a = hl.array([1, 2, -3, 0, 5])
    >>> a
    <ArrayNumericExpression of type array<int32>>

    >>> a.dtype
    dtype('array<int32>')

Hail arrays can be indexed and sliced like Python lists or :mod:`numpy` arrays:

    >>> a[1]
    >>> a[1:-1]


Boolean Logic
=============

Unlike Python, a Hail :class:`.BooleanExpression` cannot be used with ``and``,
``or``, and ``not``. The equivalents are ``&``, ``|``, and ``~``.

    >>> s1 = x == 3
    >>> s2 = x != 4

    >>> s1 & s2 # s1 and s2
    >>> s1 | s2 # s1 or s2
    >>> ~s1 # not s1

.. caution::

    The operator precedence of ``&`` and ``|`` is different from ``and`` and
    ``or``. You will need parentheses around expressions like this:

    >>> (x == 3) & (x != 4)

Conditionals
============

Python ``if`` / ``else`` do not work with Hail expressions. Instead, you must
use the :func:`.cond`, :func:`.case`, and :func:`.switch` functions.

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

Here is the Hail expression equivalent with :func:`.cond`:

    >>> hl.cond(x > 0, 1, 0)
     <Int32Expression of type int32>

This example returns an :class:`.Int32Expression` which can be used in more
computations:

    >>> a + hl.cond(x > 0, 1, 0)
    <ArrayNumericExpression of type array<int32>>

More complicated conditional statements can be constructed with :func:`.case`.
For example, we might want to emit ``1`` if ``x < -1``, ``2`` if
``-1 <= x <= 2`` and ``3`` if ``x > 2``.

    >>> (hl.case()
    ...   .when(x < -1, 1)
    ...   .when((x >= -1) & (x <= 2), 2)
    ...   .when(x > 2, 3)
    ...   .or_missing())
    <Int32Expression of type int32>

Finally, Hail has the :func:`.switch` function to build a conditional tree based
on the value of an expression. In the example below, `csq` is a
:class:`.StringExpression` representing the functional consequence of a
mutation. If `csq` does not match one of the cases specified by
:meth:`.SwitchBuilder.when`, it is set to missing with
:meth:`.SwitchBuilder.or_missing`. Other switch statements are documented in the
:class:`.SwitchBuilder` class.

    >>> csq = hl.str('nonsense')

    >>> (hl.switch(csq)
    ...    .when("synonymous", False)
    ...    .when("intron", False)
    ...    .when("nonsense", True)
    ...    .when("indel", True)
    ...    .or_missing())
    <BooleanExpression of type bool>


Missingness
===========

In Hail, all expressions can be missing.
An expression representing a missing value of a given type can be generated with
the :func:`.null` function, which takes the type as its single argument. An
example of generating a :class:`.Float64Expression` that is missing is:

    >>> hl.null('float64')

These can be used with conditional statements to set values to missing if they
don't satisfy a condition:

    >>> hl.cond(x > 2.0, x, hl.null(hl.tfloat))

The result of method calls on a missing value is ``None``. For example, if
we define ``cnull`` to be a missing value with type :class:`.tcall`, calling
the method `is_het` will return ``None`` and not ``False``.

    >>> cnull = hl.null('call')
    >>> cnull.is_het().value
    None


Binding Variables
=================

Hail inlines function calls each time an expression appears. This can result
in unexpected behavior when random values are used. For example, let `x` be
a random number generated with the function :func:`.rand_unif`:

    >>> x = hl.rand_unif(0, 1)

The value of `x` changes with each evaluation:

    >>> x.value
    0.4678132874101748

    >>> x.value
    0.9097632224065403

If we create a list with x repeated 3 times, we'd expect to get an array with identical
values. However, instead we see a list of 3 random numbers.

    >>> hl.array([x, x, x]).value
    [0.8846327207915881, 0.14415148553468504, 0.8202677741734825]

To solve this problem, we can use the :func:`.bind` function to bind an expression to a
value before applying it in a function.

    >>> expr = hl.bind(lambda x: [x, x, x], hl.rand_unif(0, 1))

    >>> expr.value
    [0.5562065047992025, 0.5562065047992025, 0.5562065047992025]


Functions
=========

In addition to the methods exposed on each :class:`.Expression`, Hail also has
numerous functions that can be applied to expressions, which also return an expression.

Take a look at the :ref:`sec-functions` page for full documentation.
