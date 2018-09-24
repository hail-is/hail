from hail.typecheck import *
from hail.utils.java import *
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
                      includes_end=bool)
    def __init__(self, start, end, includes_start=True, includes_end=False):
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

        self._jrep = scala_object(Env.hail().utils, 'Interval').apply(
            point_type._convert_to_j(start),
            point_type._convert_to_j(end),
            includes_start,
            includes_end)

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'Interval(start={}, end={}, includes_start={}, includes_end={})'\
            .format(repr(self.start), repr(self.end), repr(self.includes_start), repr(self._includes_end))

    def __eq__(self, other):
        return isinstance(other, Interval) and self._point_type == other._point_type and self._jrep.equals(other._jrep)

    def __hash__(self):
        return self._jrep.hashCode()

    @classmethod
    def _from_java(cls, jrep, point_type):
        interval = Interval.__new__(cls)
        interval._jrep = jrep
        interval._point_type = point_type
        interval._start = None
        interval._end = None
        interval._includes_start = None
        interval._includes_end = None
        super(Interval, interval).__init__()
        return interval

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

        if self._start is None:
            self._start = self.point_type._convert_to_py(self._jrep.start())
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

        if self._end is None:
            self._end = self.point_type._convert_to_py(self._jrep.end())
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

        if self._includes_start is None:
            self._includes_start = self._jrep.includesStart()
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

        if self._includes_end is None:
            self._includes_end = self._jrep.includesEnd()
        return self._includes_end

    @property
    def point_type(self):
        """Type of each element in the interval.

        Examples
        --------

        >>> interval2.point_type
        tint32

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

        from hail.expr.expressions import impute_type
        value_type = impute_type(value)
        if value_type != self.point_type:
            raise TypeError("'value' is incompatible with the interval point type: '{}', '{}'".format(value_type, self.point_type))
        return self._jrep.contains(self.point_type._jtype.ordering(), self.point_type._convert_to_j(value))

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

        if self.point_type != interval.point_type:
            raise TypeError("'interval' must have the point type '{}', but found '{}'".format(self.point_type, interval.point_type))
        return self._jrep.overlaps(self.point_type._jtype.ordering(), interval._jrep)

interval_type.set(Interval)
