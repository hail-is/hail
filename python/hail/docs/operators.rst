.. _sec-operators:

=========
Operators
=========

-------
Numeric
-------

 - ``+`` -- Add two operands.

    .. code-block:: text
        :emphasize-lines: 2

        5 + 3
        8

 - ``-`` -- Subtract right operand from the left.

    .. code-block:: text
        :emphasize-lines: 2

        5 - 3
        2

 - ``-`` -- Negates a number.

    .. code-block:: text
        :emphasize-lines: 2

        -(1 + 2)
        -3

 - ``*`` -- Multiply two operands.

    .. code-block:: text
        :emphasize-lines: 2

        5 * 3
        15

 - ``/`` -- Divide left operand by the right one. Always results in a Double.

    .. code-block:: text
        :emphasize-lines: 2

        7 / 2
        3.5

 - ``%`` -- Remainder of the division of left operand by the right (modulus).

    .. code-block:: text
        :emphasize-lines: 2,5

        7 % 2
        1

        10 % 4
        2

 - ``//`` -- Floor division - division that results into whole number adjusted to the left in the number line.

    .. code-block:: text
        :emphasize-lines: 2,5

        7 // 2
        3

        -7 // 2
        -4

--------------
Array[Numeric]
--------------

If one of the two operands is a scalar, the operation will be applied to each element of the Array. If both operands are Arrays, the operation will be applied positionally. This will fail if the array dimensions do not match.

 - ``+`` -- Add two operands.

    .. code-block:: text
        :emphasize-lines: 2, 5

        [1, 2, 3] + [1, 1, 1]
        [2, 3, 4]

        [2, 0, 1] + 5
        [7, 5, 6]

 - ``-`` -- Subtract right operand from the left.

    .. code-block:: text
        :emphasize-lines: 2, 5, 8

        [1, 2, 3] - [1, 1, 1]
        [0, 1, 2]

        [2, 0, 1] - 5
        [-3, -5, -4]

        3 - [2, 4, 5]
        [1, -1, -2]

 - ``*`` -- Multiply two operands.

    .. code-block:: text
        :emphasize-lines: 2, 5

        [1, 2, 3] * [1, 1, 1]
        [1, 2, 3]

        [2, 0, 1] * 5
        [10, 0, 5]


 - ``/`` -- Divide left operand by the right one. Always results in a Double.

    .. code-block:: text
        :emphasize-lines: 2, 5, 8

        [1, 2, 3] / [1, 4, 9]
        [1.0, 0.5, 0.333]

        [2, 0, 1] / 5
        [0.4, 0.0, 0.2]

        5 / [2, 4, 1]
        [2.5, 1.25, 5.0]

----------
Comparison
----------

 - ``==`` -- True if the left operand is equal to the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        [1, 2, 3] == [1, 2, 3]
        true

 - ``!=`` -- True if the left operand is not equal to the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        [1, 2, 3] != [4, 5, 6]
        true

 - ``<`` -- True if the left operand is less than the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        5 < 3
        False

 - ``<=`` -- True if the left operand is less than or equal to the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        3 <= 5
        True

 - ``>`` -- True if the left operand is greater than the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        7 > 2
        True

 - ``>=`` -- True if the left operand is greater than or equal to the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        3 >= 9
        False

 - ``~`` -- True if a regular expression pattern matches the target string.

    .. code-block:: text
        :emphasize-lines: 2

        "1KG" ~ "Cohort_1KG_NA12878"
        True

-------
Logical
-------

 - ``&&`` -- True if both the left and right operands are true.

    .. code-block:: text
        :emphasize-lines: 2

        (5 >= 3) && (2 < 10)
        True

 - ``||`` -- True if at least one operand is true.

    .. code-block:: text
        :emphasize-lines: 2

        (5 <= 3) || (2 < 10)
        True

 - ``!`` -- Negates a boolean variable. Returns false if the variable is true and true if the variable is false.

    .. code-block:: text
        :emphasize-lines: 2

        !(5 >= 3)
        False

------
String
------

 - ``+`` -- Concatenate two strings together.

    .. code-block:: text
        :emphasize-lines: 2

        "a" + "b"
        "ab"
