from hail.typecheck import *
from hail.utils.java import *
from hail.genetics.reference_genome import reference_genome_type
import hail as hl

interval_type = lazy()


class Interval(object):
    """
    An object representing a range of values between `start` and `end`.

    >>> interval = hl.Interval(3, 6)

    Parameters
    ----------
    start : any type
        Object with type `point_type`.
    end : any type
        Object with type `point_type`.
    include_start : :obj:`bool`
        Interval includes start.
    include_end : :obj:`bool`
        Interval includes end.
    """

    @typecheck_method(start=anytype,
                      end=anytype,
                      include_start=bool,
                      include_end=bool)
    def __init__(self, start, end, include_start=True, include_end=False):
        from hail.expr.expressions import impute_type, unify_types_limited
        start_type = impute_type(start)
        end_type = impute_type(end)
        point_type = unify_types_limited(start_type, end_type)

        if point_type is None:
            raise TypeError("'start' and 'end' have incompatible types: '{}', '{}'.".format(start_type, end_type))

        self._point_type = point_type
        self._start = start
        self._end = end
        self._include_start = include_start
        self._include_end = include_end

        self._jrep = scala_object(Env.hail().utils, 'Interval').apply(
            point_type._convert_to_j(start),
            point_type._convert_to_j(end),
            include_start,
            include_end)

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'Interval(start={}, end={}, include_start={}, include_end={})'\
            .format(repr(self.start), repr(self.end), repr(self.include_start), repr(self._include_end))

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
        interval._include_start = None
        interval._include_end = None
        super(Interval, interval).__init__()
        return interval

    @property
    def start(self):
        """Start point of the interval.

        Examples
        --------

        .. doctest::

            >>> interval.start
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

        .. doctest::

            >>> interval.end
            6

        Returns
        -------
        Object with type :meth:`.point_type`
        """

        if self._end is None:
            self._end = self.point_type._convert_to_py(self._jrep.end())
        return self._end

    @property
    def include_start(self):
        """True if interval is inclusive of start.

        Examples
        --------

        .. doctest::

            >>> interval.include_start
            True

        Returns
        -------
        :obj:`bool`
        """

        if self._include_start is None:
            self._include_start = self._jrep.includeStart()
        return self._include_start

    @property
    def include_end(self):
        """True if interval is inclusive of end.

        Examples
        --------

        .. doctest::

            >>> interval.include_end
            False

        Returns
        -------
        :obj:`bool`
        """

        if self._include_end is None:
            self._include_end = self._jrep.includeEnd()
        return self._include_end

    @property
    def point_type(self):
        """Type of each element in the interval.

        Examples
        --------

        .. doctest::

            >>> interval.point_type
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

        .. doctest::

            >>> interval.contains(5)
            True

            >>> interval.contains(6)
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
        return self._jrep.mayOverlap(self.point_type._jtype.ordering(), interval._jrep)

    @classmethod
    @typecheck_method(string=str,
                      reference_genome=reference_genome_type)
    def parse_locus_interval(cls, string, reference_genome='default'):
        """Parses a genomic interval from string representation.

        Examples
        --------
        
        >>> interval_1 = hl.Interval.parse_locus_interval('X:100005-X:150020')
        >>> interval_2 = hl.Interval.parse_locus_interval('16:29500000-30200000')
        >>> interval_3 = hl.Interval.parse_locus_interval('16:29.5M-30.2M')  # same as interval_2
        >>> interval_4 = hl.Interval.parse_locus_interval('16:30000000-END')
        >>> interval_5 = hl.Interval.parse_locus_interval('16:30M-END')  # same as interval_4
        >>> interval_6 = hl.Interval.parse_locus_interval('1-22')  # autosomes
        >>> interval_7 = hl.Interval.parse_locus_interval('X')  # all of chromosome X

        Notes
        -----

        The start locus must precede the end locus. The default bounds of the
        interval are left-inclusive and right-exclusive. To change this, add
        one of ``[`` or ``(`` at the beginning of the string for left-inclusive
        or left-exclusive respectively. Likewise, add one of ``]`` or ``)`` at
        the end of the string for right-inclusive or right-exclusive
        respectively.

        >>> interval_8 = hl.Interval.parse_locus_interval("[15:1-1000]")
        >>> interval_9 = hl.Interval.parse_locus_interval("(15:1-1000]")

        ``CHR1:POS1-CHR2:POS2`` is the fully specified representation, and
        we use this to define the various shortcut representations.
        
        In a ``POS`` field, ``start`` (``Start``, ``START``) stands for 0.
        
        In a ``POS`` field, ``end`` (``End``, ``END``) stands for the maximum
        contig length.
        
        In a ``POS`` field, the qualifiers ``m`` (``M``) and ``k`` (``K``)
        multiply the given number by ``1,000,000`` and ``1,000``, respectively.
        ``1.6K`` is short for 1600, and ``29M`` is short for 29000000.
        
        ``CHR:POS1-POS2`` stands for ``CHR:POS1-CHR:POS2``
        
        ``CHR1-CHR2`` stands for ``CHR1:START-CHR2:END``
        
        ``CHR`` stands for ``CHR:START-CHR:END``

        Parameters
        ----------
        string : :obj:`str`
           String to parse.
        reference_genome : :obj:`str` or :class:`.ReferenceGenome`
           Reference genome to use.
        
        Returns
        -------
        :class:`.Interval`
        """

        jrep = scala_object(Env.hail().variant, 'Locus').parseInterval(string, reference_genome._jrep)
        return Interval._from_java(jrep, hl.tlocus(reference_genome))

interval_type.set(Interval)
