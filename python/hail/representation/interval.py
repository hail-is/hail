from hail.java import *
from hail.representation.variant import Locus
from hail.typecheck import *
from hail.history import *

interval_type = lazy()

class Interval(HistoryMixin):
    """
    A genomic interval marked by start and end loci.

    .. testsetup::

        interval1 = Interval.parse('X:100005-X:150020')
        interval2 = Interval.parse('16:29500000-30200000')

    :param start: inclusive start locus
    :type start: :class:`.Locus`
    :param end: exclusive end locus
    :type end: :class:`.Locus`
    """

    @handle_py4j
    @record_init
    def __init__(self, start, end):
        if not (isinstance(start, Locus) and isinstance(end, Locus)):
            raise TypeError('expect arguments of type (Locus, Locus) but found (%s, %s)' %
                            (str(type(start)), str(type(end))))
        jrep = scala_object(Env.hail().variant, 'Locus').makeInterval(start._jrep, end._jrep)
        self._init_from_java(jrep)
        super(Interval, self).__init__()

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'Interval(start=%s, end=%s)' % (repr(self.start), repr(self.end))

    def __eq__(self, other):
        return self._jrep.equals(other._jrep)

    def __hash__(self):
        return self._jrep.hashCode()

    def _init_from_java(self, jrep):
        self._jrep = jrep
        self._start = Locus._from_java(self._jrep.start())

    @classmethod
    @record_classmethod
    def _from_java(cls, jrep):
        interval = Interval.__new__(cls)
        interval._init_from_java(jrep)
        return interval

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(string=strlike)
    def parse(cls, string):
        """Parses a genomic interval from string representation.

        **Examples**:

        >>> interval_1 = Interval.parse('X:100005-X:150020')
        >>> interval_2 = Interval.parse('16:29500000-30200000')
        >>> interval_3 = Interval.parse('16:29.5M-30.2M')  # same as interval_2
        >>> interval_4 = Interval.parse('16:30000000-END')
        >>> interval_5 = Interval.parse('16:30M-END')  # same as interval_4
        >>> interval_6 = Interval.parse('1-22')  # autosomes
        >>> interval_7 = Interval.parse('X')  # all of chromosome X


        There are several acceptable representations.

        ``CHR1:POS1-CHR2:POS2`` is the fully specified representation, and
        we use this to define the various shortcut representations.

        In a ``POS`` field, ``start`` (``Start``, ``START``) stands for 0.

        In a ``POS`` field, ``end`` (``End``, ``END``) stands for max int.

        In a ``POS`` field, the qualifiers ``m`` (``M``) and ``k`` (``K``) multiply
        the given number by ``1,000,000`` and ``1,000``, respectively.  ``1.6K`` is short for
        1600, and ``29M`` is short for 29000000.

        ``CHR:POS1-POS2`` stands for ``CHR:POS1-CHR:POS2``

        ``CHR1-CHR2`` stands for ``CHR1:START-CHR2:END``

        ``CHR`` stands for ``CHR:START-CHR:END``

        Note that the start locus must precede the start locus.

        :rtype: :class:`.Interval`
        """

        jrep = scala_object(Env.hail().variant, 'Locus').parseInterval(string)
        return Interval._from_java(jrep)

    @property
    def start(self):
        """
        Locus object referring to the start of the interval (inclusive).

        :rtype: :class:`.Locus`
        """
        return Locus._from_java(self._jrep.start())

    @property
    def end(self):
        """
        Locus object referring to the end of the interval (exclusive).

        :rtype: :class:`.Locus`
        """
        return Locus._from_java(self._jrep.end())

    @handle_py4j
    @typecheck_method(locus=Locus)
    def contains(self, locus):
        """True if the supplied locus is contained within the interval.

        This membership check is left-inclusive, right-exclusive.  This
        means that the interval 1:100-101 includes 1:100 but not 1:101.

        :type: locus: :class:`.Locus.`
        :rtype: bool
        """

        return self._jrep.contains(locus._jrep)

    @handle_py4j
    @typecheck_method(interval=interval_type)
    def overlaps(self, interval):
        """True if the the supplied interval contains any locus in common with this one.

        The statement

        >>> interval1.overlaps(interval2)

        is equivalent to

        >>> interval1.contains(interval2.start) or interval2.contains(interval1.start)

        :type: interval: :class:`.Interval`
        :rtype: bool"""

        return self._jrep.overlaps(interval._jrep)

interval_type.set(Interval)