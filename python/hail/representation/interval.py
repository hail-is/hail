from hail.java import scala_object, Env
from hail.representation import Locus


class Interval(object):
    """A genomic interval marked by start and end loci.

    The start and end are :class:`.Locus` objects.
    """

    def __init__(self, start, end):
        """Create an Interval object from start and end loci.

        :param :class:`.Locus` start: inclusive start locus
        :param :class:`.Locus` end: exclusive end locus
        """

        if not isinstance(start, Locus) and isinstance(end, Locus):
            raise TypeError('expect arguments of type (Locus, Locus) but found (%s, %s)' % (type(start), type(end)))
        jrep = scala_object(Env.hail_package().variant, 'Locus').makeInterval(start._jrep, end._jrep)
        self._init_from_java(jrep)

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'Interval(%s, %s)' % (repr(self.start), repr(self.end))

    def __eq__(self, other):
        return self._jrep.equals(other._jrep)

    def _init_from_java(self, jrep):
        self._jrep = jrep
        self.start = Locus._from_java(jrep.start())
        self.end = Locus._from_java(jrep.end())

    @classmethod
    def _from_java(cls, jrep):
        interval = Interval.__new__(cls)
        interval._init_from_java(jrep)
        return interval

    @staticmethod
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

        jrep = scala_object(Env.hail_package().variant, 'Locus').parseInterval(string)
        return Interval._from_java(jrep)

    def contains(self, locus):
        """True if the supplied locus is contained within the interval.

        This membership check is left-inclusive, right-exclusive.  This
        means that the interval 1:100-101 includes 1:100 but not 1:101.

        :type: locus: :class:`.Locus.`
        :rtype: bool
        """

        return self._jrep.contains(locus._jrep)

    def overlaps(self, interval):
        """True if the the supplied interval contains any locus in common with this one.

        The statement

        >>> interval1.overlaps(interval2)

        is equivalent to

        >>> interval1.contains(interval2.start) or interval2.contains(interval1.start)

        :type: interval: :class:`.Interval`
        :rtype: bool"""

        return self._jrep.overlaps(interval._jrep)
