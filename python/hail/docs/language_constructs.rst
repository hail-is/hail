.. _sec-langconst:

===================
Language Constructs
===================

 - **v = e**

    Bind variable name ``v`` to the result of evaluating ``e``.

 - **if (p) a else b**

    The value of the conditional is the value of ``a`` or ``b`` depending on ``p``. If ``p`` is missing, the value of the conditional is missing.

    .. code-block:: text
        :emphasize-lines: 2

        if (5 % 2 == 0) true else false
        result: false

 - **let v1 = e1 and v2 = e2 and ... and vn = en in b**

    Bind variables ``v1`` through ``vn`` to result of evaluating the ``ei``. The value of the let is the value of ``b``. ``v1`` is visible in ``e2`` through ``en``, etc.

    .. code-block:: text
        :emphasize-lines: 2

        let v1 = 5 and v2 = 7 and v3 = 2 in v1 * v2 * v3
        result: 70