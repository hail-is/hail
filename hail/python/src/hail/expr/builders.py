from hail.typecheck import typecheck_method
from hail.expr.expressions import unify_types, unify_types_limited, expr_any, \
    expr_bool, ExpressionException, construct_expr, expr_str
from hail import ir


class ConditionalBuilder(object):
    def __init__(self):
        self._ret_type = None
        self._cases = []

    def _unify_type(self, t):
        if self._ret_type is None:
            self._ret_type = t
        else:
            r = unify_types_limited(self._ret_type, t)
            if not r:
                raise TypeError("'then' expressions must have same type, found '{}' and '{}'".format(
                    self._ret_type, t
                ))


class SwitchBuilder(ConditionalBuilder):
    """Class for generating conditional trees based on value of an expression.

    Examples
    --------

    >>> csq = hl.literal('loss of function')
    >>> expr = (hl.switch(csq)
    ...           .when('synonymous', 1)
    ...           .when('SYN', 1)
    ...           .when('missense', 2)
    ...           .when('MIS', 2)
    ...           .when('loss of function', 3)
    ...           .when('LOF', 3)
    ...           .or_missing())
    >>> hl.eval(expr)
    3

    Notes
    -----
    All expressions appearing as the `then` parameters to
    :meth:`~.SwitchBuilder.when` or :meth:`~.SwitchBuilder.default` method
    calls must be the same type.

    See Also
    --------
    :func:`.case`, :func:`.cond`, :func:`.switch`

    Parameters
    ----------
    expr : :class:`.Expression`
        Value to match against.
    """

    @typecheck_method(base=expr_any)
    def __init__(self, base):
        self._base = base
        self._has_missing_branch = False
        super(SwitchBuilder, self).__init__()

    def _finish(self, default):
        assert len(self._cases) > 0

        from hail.expr.functions import cond, bind

        def f(base):
            # build cond chain bottom-up
            expr = default
            for condition, then in self._cases[::-1]:
                expr = cond(condition, then, expr)
            return expr

        return bind(f, self._base)

    @typecheck_method(value=expr_any, then=expr_any)
    def when(self, value, then) -> 'SwitchBuilder':
        """Add a value test. If the `base` expression is equal to `value`, then
         returns `then`.

        Warning
        -------
        Missingness always compares to missing. Both ``NA == NA`` and
        ``NA != NA`` return ``NA``. Use :meth:`~SwitchBuilder.when_missing`
        to test missingness.

        Parameters
        ----------
        value : :class:`.Expression`
        then : :class:`.Expression`

        Returns
        -------
        :class:`.SwitchBuilder`
            Mutates and returns `self`.
        """
        can_compare = unify_types(self._base.dtype, value.dtype)
        if not can_compare:
            raise TypeError("cannot compare expressions of type '{}' and '{}'".format(
                self._base.dtype, value.dtype))

        self._unify_type(then.dtype)
        self._cases.append((self._base == value, then))
        return self

    @typecheck_method(then=expr_any)
    def when_missing(self, then) -> 'SwitchBuilder':
        """Add a test for missingness. If the `base` expression is missing,
        returns `then`.

        Parameters
        ----------
        then : :class:`.Expression`

        Returns
        -------
        :class:`.SwitchBuilder`
            Mutates and returns `self`.
        """
        if self._has_missing_branch:
            raise ExpressionException("'when_missing' can only be called once")
        self._unify_type(then.dtype)

        from hail.expr.functions import is_missing
        # need to insert at 0, because upstream missingness would propagate
        self._cases.insert(0, (is_missing(self._base), then))
        return self

    @typecheck_method(then=expr_any)
    def default(self, then):
        """Finish the switch statement by adding a default case.

        Notes
        -----
        If no value from a :meth:`~.SwitchBuilder.when` call is matched, then
        `then` is returned.

        Parameters
        ----------
        then : :class:`.Expression`

        Returns
        -------
        :class:`.Expression`
        """
        if len(self._cases) == 0:
            return then
        self._unify_type(then.dtype)
        return self._finish(then)

    def or_missing(self):
        """Finish the switch statement by returning missing.

        Notes
        -----
        If no value from a :meth:`~.SwitchBuilder.when` call is matched, then
        the result is missing.

        Parameters
        ----------
        then : :class:`.Expression`

        Returns
        -------
        :class:`.Expression`
        """
        if len(self._cases) == 0:
            raise ExpressionException("'or_missing' cannot be called without at least one 'when' call")
        from hail.expr.functions import null
        return self._finish(null(self._ret_type))


class CaseBuilder(ConditionalBuilder):
    """Class for chaining multiple if-else statements.


    Examples
    --------

    >>> x = hl.literal('foo bar baz')
    >>> expr = (hl.case()
    ...           .when(x[:3] == 'FOO', 1)
    ...           .when(x.length() == 11, 2)
    ...           .when(x == 'secret phrase', 3)
    ...           .default(0))
    >>> hl.eval(expr)
    2

    Notes
    -----
    All expressions appearing as the `then` parameters to
    :meth:`~.CaseBuilder.when` or :meth:`~.CaseBuilder.default` method calls
    must be the same type.

    Parameters
    ----------
    missing_false: :obj:`.bool`
        Treat missing predicates as ``False``.

    See Also
    --------
    :func:`.case`, :func:`.cond`, :func:`.switch`
    """

    def __init__(self, missing_false=False):
        super(CaseBuilder, self).__init__()
        self._missing_false = missing_false

    def _finish(self, default):
        assert len(self._cases) > 0

        from hail.expr.functions import cond

        expr = default
        for conditional, then in self._cases[::-1]:
            expr = cond(conditional, then, expr, missing_false=self._missing_false)
        return expr

    @typecheck_method(condition=expr_bool, then=expr_any)
    def when(self, condition, then) -> 'CaseBuilder':
        """Add a branch. If `condition` is ``True``, then returns `then`.

        Warning
        -------
        Missingness is treated similarly to :func:`.cond`. Missingness is
        **not** treated as ``False``. A `condition` that evaluates to missing
        will return a missing result, not proceed to the next case. Always
        test missingness first in a :class:`.CaseBuilder`.

        Parameters
        ----------
        condition: :class:`.BooleanExpression`
        then : :class:`.Expression`

        Returns
        -------
        :class:`.CaseBuilder`
            Mutates and returns `self`.
        """
        self._unify_type(then.dtype)
        self._cases.append((condition, then))
        return self

    @typecheck_method(then=expr_any)
    def default(self, then):
        """Finish the case statement by adding a default case.

        Notes
        -----
        If no condition from a :meth:`~.CaseBuilder.when` call is ``True``,
        then `then` is returned.

        Parameters
        ----------
        then : :class:`.Expression`

        Returns
        -------
        :class:`.Expression`
        """
        if len(self._cases) == 0:
            return then
        self._unify_type(then.dtype)
        return self._finish(then)

    def or_missing(self):
        """Finish the case statement by returning missing.

        Notes
        -----
        If no condition from a :meth:`.CaseBuilder.when` call is ``True``, then
        the result is missing.

        Parameters
        ----------
        then : :class:`.Expression`

        Returns
        -------
        :class:`.Expression`
        """
        if len(self._cases) == 0:
            raise ExpressionException("'or_missing' cannot be called without at least one 'when' call")
        from hail.expr.functions import null
        return self._finish(null(self._ret_type))

    @typecheck_method(message=expr_str)
    def or_error(self, message):
        """Finish the case statement by throwing an error with the given message.

        Notes
        -----
        If no condition from a :meth:`.CaseBuilder.when` call is ``True``, then
        an error is thrown.

        Parameters
        ----------
        message : :class:`.Expression` of type :data:`tstr`

        Returns
        -------
        :class:`.Expression`
        """
        if len(self._cases) == 0:
            raise ExpressionException("'or_error' cannot be called without at least one 'when' call")
        error_expr = construct_expr(ir.Die(message._ir, self._ret_type), self._ret_type)
        return self._finish(error_expr)
