from hail.typecheck import *
from hail.utils.java import *
from hail.expr.types import hail_type
import hail as hl

interval_type = lazy()


class Interval(object):
    """
    An object representing a range of values between `start` and `end`.

    >>> interval2 = hl.Interval(3, 6)

    Parameters
    ----------
    start : any type
        Object with type `point_type`.
    end : any type
        Object with type `point_type`.
    includes_start : :obj:`bool`
        Interval includes start.
    includes_end : :obj:`bool`
        Interval includes end.
    """

    @typecheck_method(start=anytype,
                      end=anytype,
                      includes_start=bool,
                      includes_end=bool,
                      point_type=nullable(hail_type))
    def __init__(self, start, end, includes_start=True, includes_end=False, point_type=None):
        if point_type is None:
            from hail.expr.expressions import impute_type, unify_types_limited
            start_type = impute_type(start)
            end_type = impute_type(end)
            point_type = unify_types_limited(start_type, end_type)
            if point_type is None:
                raise TypeError("'start' and 'end' have incompatible types: '{}', '{}'.".format(start_type, end_type))

        self._point_type = point_type
        self._start = start
        self._end = end
        self._includes_start = includes_start
        self._includes_end = includes_end

    def __str__(self):
        if isinstance(self._start, Locus) and self._start.contig == self._end.contig:
            bounds = f'{self._start}-{self._end.position}'
        else:
            bounds = f'{self._start}-{self._end}'
        open = '[' if self._includes_start else '('
        close = ']' if self._includes_end else ')'
        return f'{open}{bounds}{close}'

    def __repr__(self):
        return 'Interval(start={}, end={}, includes_start={}, includes_end={})'\
            .format(repr(self.start), repr(self.end), repr(self.includes_start), repr(self._includes_end))

    def __eq__(self, other):
        return (isinstance(other, Interval)
                and self._start == other._start
                and self._end == other._end
                and self._includes_start == other._includes_start
                and self._includes_end == other._includes_end)

    def __hash__(self):
        return hash(self._start) ^ hash(self._end) ^ hash(self._includes_start) ^ hash(self._includes_end)

    @property
    def start(self):
        """Start point of the interval.

        Examples
        --------

        >>> interval2.start
        3

        Returns
        -------
        Object with type :meth:`.point_type`
        """

        return self._start

    @property
    def end(self):
        """End point of the interval.

        Examples
        --------

        >>> interval2.end
        6

        Returns
        -------
        Object with type :meth:`.point_type`
        """

        return self._end

    @property
    def includes_start(self):
        """True if interval is inclusive of start.

        Examples
        --------

        >>> interval2.includes_start
        True

        Returns
        -------
        :obj:`bool`
        """

        return self._includes_start

    @property
    def includes_end(self):
        """True if interval is inclusive of end.

        Examples
        --------

        >>> interval2.includes_end
        False

        Returns
        -------
        :obj:`bool`
        """

        return self._includes_end

    @property
    def point_type(self):
        """Type of each element in the interval.

        Examples
        --------

        >>> interval2.point_type
        dtype('int32')

        Returns
        -------
        :class:`.Type`
        """

        return self._point_type

    def contains(self, value):
        """True if `value` is contained within the interval.

        Examples
        --------

        >>> interval2.contains(5)
        True

        >>> interval2.contains(6)
        False

        Parameters
        ----------
        value :
            Object with type :meth:`.point_type`.

        Returns
        -------
        :obj:`bool`
        """

        return hl.eval(hl.literal(self, hl.tinterval(self._point_type)).contains(value))

    @typecheck_method(interval=interval_type)
    def overlaps(self, interval):
        """True if the the supplied interval contains any value in common with this one.

        Parameters
        ----------
        interval : :class:`.Interval`
            Interval object with the same point type.

        Returns
        -------
        :obj:`bool`
        """

        return hl.eval(hl.literal(self, hl.tinterval(self._point_type)).overlaps(interval))

interval_type.set(Interval)
