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

        let x = 1 in -x
        -1

 - ``*`` -- Multiply two operands.

    .. code-block:: text
        :emphasize-lines: 2

        5 * 3
        15

 - ``/`` -- Divide left operand by the right one. Always results in a Double.

    .. code-block:: text
        :emphasize-lines: 2

        7 / 2
        result: 3.5

 - ``%`` -- Remainder of the division of left operand by the right (modulus).

    .. code-block:: text
        :emphasize-lines: 2,5

        7 % 2
        result: 1

        10 % 4
        result: 2

 - ``//`` -- Floor division - division that results into whole number adjusted to the left in the number line.

    .. code-block:: text
        :emphasize-lines: 2,5

        7 // 2
        result: 3

        -7 // 2
        result: -4

--------------
Array[Numeric]
--------------

If one of the two operands is a scalar, the operation will be applied to each element of the Array. If both operands are Arrays, the operation will be applied positionally. This will fail if the array dimensions do not match.

 - ``+`` -- Add two operands.

    .. code-block:: text
        :emphasize-lines: 2, 5

        let a = [1, 2, 3] and b = [1, 1, 1] in a + b
        result: [2, 3, 4]

        let c = [2, 0, 1] and d = 5 in c + d
        result: [7, 5, 6]

 - ``-`` -- Subtract right operand from the left.

    .. code-block:: text
        :emphasize-lines: 2, 5, 8

        let a = [1, 2, 3] and b = [1, 1, 1] in a - b
        result: [0, 1, 2]

        let c = [2, 0, 1] and d = 5 in c - d
        result: [-3, -5, -4]

        let e = 3 and f = [2, 4, 5] in e - f
        result: [1, -1, -2]

 - ``*`` -- Multiply two operands.

    .. code-block:: text
        :emphasize-lines: 2, 5

        let a = [1, 2, 3] and b = [1, 1, 1] in a * b
        result: [1, 2, 3]

        let c = [2, 0, 1] and d = 5 in c * d
        result: [10, 0, 5]


 - ``/`` -- Divide left operand by the right one. Always results in a Double.

    .. code-block:: text
        :emphasize-lines: 2, 5, 8

        let a = [1, 2, 3] and b = [1, 4, 9] in a / b
        result: [1.0, 0.5, 0.333]

        let c = [2, 0, 1] and d = 5 in c / d
        result: [0.4, 0.0, 0.2]

        let e = 5 and f = [2, 4, 1] in e / f
        result: [2.5, 1.25, 5.0]

----------
Comparison
----------

 - ``==`` -- True if the left operand is equal to the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [1, 2, 3] and b = [1, 2, 3] in a == b
        result: true

 - ``!=`` -- True if the left operand is not equal to the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [1, 2, 3] and b = [4, 5, 6] in a != b
        result: true

 - ``<`` -- True if the left operand is less than the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        5 < 3
        result: false

 - ``<=`` -- True if the left operand is less than or equal to the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        3 <= 5
        result: true

 - ``>`` -- True if the left operand is greater than the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        7 > 2
        result: true

 - ``>=`` -- True if the left operand is greater than or equal to the right operand.

    .. code-block:: text
        :emphasize-lines: 2

        3 >= 9
        result: false

 - ``~`` -- True if a regular expression pattern matches the target string.

    .. code-block:: text
        :emphasize-lines: 2

        let regex = '1kg' and target = '1kg-NA12878' in regex ~ target
        result: true

-------
Logical
-------

 - ``&&`` -- True if both the left and right operands are true.

    .. code-block:: text
        :emphasize-lines: 2

        (5 >= 3) && (2 < 10)
        result: true

 - ``||`` -- True if at least one operand is true.

    .. code-block:: text
        :emphasize-lines: 2

        (5 <= 3) || (2 < 10)
        result: true

 - ``!`` -- Negates a boolean variable. Returns false if the variable is true and true if the variable is false.

    .. code-block:: text
        :emphasize-lines: 2

        !(5 >= 3)
        result: false

------
String
------

 - ``+`` -- Concatenate two strings together.

    .. code-block:: text
        :emphasize-lines: 2

        "a" + "b"
        "ab"
