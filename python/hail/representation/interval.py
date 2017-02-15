from hail.java import scala_object, env, handle_py4j
from hail.representation import Locus


class Interval(object):
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
    def __init__(self, start, end):
        if not isinstance(start, Locus) and isinstance(end, Locus):
            raise TypeError('expect arguments of type (Locus, Locus) but found (%s, %s)' % (type(start), type(end)))
        jrep = scala_object(env.hail.variant, 'Locus').makeInterval(start._jrep, end._jrep)
        self._init_from_java(jrep)

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'Interval(%s, %s)' % (repr(self.start), repr(self.end))

    def __eq__(self, other):
        return self._jrep.equals(other._jrep)

    def __hash__(self):
        return self._jrep.hashCode()

    def _init_from_java(self, jrep):
        self._jrep = jrep
        self._start = Locus._from_java(self._jrep.start())

    @classmethod
    def _from_java(cls, jrep):
        interval = Interval.__new__(cls)
        interval._init_from_java(jrep)
        return interval

    @staticmethod
    @handle_py4j
    def parse(string):
        """Parses a genomic interval from string representation.

        There are two acceptable representations, CHR:POS-CHR:POS
        and CHR:POS-POS.  In the second case, the second locus is
        assumed to have the same chromosome.  The second locus must
        be ordered after the first (later chromosome / position).

        Example:

        >>> interval_1 = Interval.parse('X:100005-X:150020')
        >>> interval_2 = Interval.parse('16:29500000-30200000')

        :rtype: :class:`.Interval`
        """

        jrep = scala_object(env.hail.variant, 'Locus').parseInterval(string)
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
    def contains(self, locus):
        """True if the supplied locus is contained within the interval.

        This membership check is left-inclusive, right-exclusive.  This
        means that the interval 1:100-101 includes 1:100 but not 1:101.

        :type: locus: :class:`.Locus.`
        :rtype: bool
        """

        return self._jrep.contains(locus._jrep)

    @handle_py4j
    def overlaps(self, interval):
        """True if the the supplied interval contains any locus in common with this one.

        The statement

        >>> interval1.overlaps(interval2)

        is equivalent to

        >>> interval1.contains(interval2.start) or interval2.contains(interval1.start)

        :type: interval: :class:`.Interval`
        :rtype: bool"""

        return self._jrep.overlaps(interval._jrep)
