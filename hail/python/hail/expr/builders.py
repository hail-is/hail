import hail as hl
from hail.typecheck import typecheck_method
from hail.expr.expressions import (
    unify_types,
    unify_types_limited,
    expr_any,
    expr_bool,
    ExpressionException,
    construct_expr,
    expr_str,
)
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
                raise TypeError("'then' expressions must have same type, found '{}' and '{}'".format(self._ret_type, t))


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
    :meth:`~hail.expr.builders.SwitchBuilder.when` or
    :meth:`~hail.expr.builders.SwitchBuilder.default` method calls must be the
    same type.

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
        self._when_missing_case = None
        super(SwitchBuilder, self).__init__()

    def _finish(self, default):
        assert len(self._cases) > 0 or self._when_missing_case is not None

        def f(base):
            # build cond chain bottom-up
            if default is self._base:
                expr = base
            else:
                expr = default
            for value, then in self._cases[::-1]:
                expr = hl.if_else(base == value, then, expr)
            # needs to be on the outside, because upstream missingness would propagate
            if self._when_missing_case is not None:
                expr = hl.if_else(hl.is_missing(base), self._when_missing_case, expr)
            return expr

        return hl.bind(f, self._base)

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
            raise TypeError("cannot compare expressions of type '{}' and '{}'".format(self._base.dtype, value.dtype))

        self._unify_type(then.dtype)
        self._cases.append((value, then))
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
        if self._when_missing_case is not None:
            raise ExpressionException("'when_missing' can only be called once")
        self._unify_type(then.dtype)

        self._when_missing_case = then
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
        if len(self._cases) == 0 and self._when_missing_case is None:
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
        from hail.expr.functions import missing

        return self._finish(missing(self._ret_type))

    @typecheck_method(message=expr_str)
    def or_error(self, message):
        """Finish the switch statement by throwing an error with the given message.

        Notes
        -----
        If no value from a :meth:`.SwitchBuilder.when` call is matched, then an
        error is thrown.

        Parameters
        ----------
        message : :class:`.Expression` of type :obj:`.tstr`

        Returns
        -------
        :class:`.Expression`
        """
        if len(self._cases) == 0:
            raise ExpressionException("'or_error' cannot be called without at least one 'when' call")
        error_expr = construct_expr(ir.Die(message._ir, self._ret_type), self._ret_type)
        return self._finish(error_expr)


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
    :meth:`~hail.expr.builders.CaseBuilder.when` or
    :meth:`~hail.expr.builders.CaseBuilder.default` method calls must be the
    same type.

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

        from hail.expr.functions import if_else

        expr = default
        for conditional, then in self._cases[::-1]:
            expr = if_else(conditional, then, expr, missing_false=self._missing_false)
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
        from hail.expr.functions import missing

        return self._finish(missing(self._ret_type))

    @typecheck_method(message=expr_str)
    def or_error(self, message):
        """Finish the case statement by throwing an error with the given message.

        Notes
        -----
        If no condition from a :meth:`.CaseBuilder.when` call is ``True``, then
        an error is thrown.

        Parameters
        ----------
        message : :class:`.Expression` of type :obj:`.tstr`

        Returns
        -------
        :class:`.Expression`
        """
        if len(self._cases) == 0:
            raise ExpressionException("'or_error' cannot be called without at least one 'when' call")
        error_expr = construct_expr(ir.Die(message._ir, self._ret_type), self._ret_type)
        return self._finish(error_expr)
