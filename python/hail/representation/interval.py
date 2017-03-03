from hail.java import *
from hail.representation.variant import Locus


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
        if not (isinstance(start, Locus) and isinstance(end, Locus)):
            raise TypeError('expect arguments of type (Locus, Locus) but found (%s, %s)' %
                            (str(type(start)), str(type(end))))
        jrep = scala_object(env.hail.variant, 'Locus').makeInterval(start._jrep, end._jrep)
        self._init_from_java(jrep)

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
    def _from_java(cls, jrep):
        interval = Interval.__new__(cls)
        interval._init_from_java(jrep)
        return interval

    @staticmethod
    @handle_py4j
    def parse(string):
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
        the given number by 10**6 and 10**3, respectively.  '1.6K' is short for
        1600, and 29M is short for 29000000

        ``CHR:POS1-POS2`` stands for ``CHR:POS1-CHR:POS2``

        ``CHR1-CHR2`` stands for ``CHR1:START-CHR2:END``

        ``CHR`` stands for ``CHR:START-CHR:END``

        Note that the start locus must precede the start locus.


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


class IntervalTree(object):
    """An object used for efficient filtering of genomic intervals.

    :param intervals: list of intervals to index in tree representation
    :type intervals: list of :class:`.Interval`
    """

    def __init__(self, intervals):
        jarr = jarray(env.hail.utils.Interval, [i._jrep for i in intervals])
        jrep = env.jutils.makeIntervalList([i._jrep for i in intervals])
        self._jrep = jrep

    @classmethod
    def _init_from_java(cls, jrep):
        itree = IntervalTree.__new__(cls)
        itree._jrep = jrep
        return itree

    @staticmethod
    @handle_py4j
    def parse_all(interval_strings):
        """Parse a list of strings into an interval list

        :param interval_strings: list of interval strings to be parsed
        :type interval_strings: list of str

        :rtype: :class:`.IntervalTree`
        """

        jrep = env.jutils.parseIntervalList(interval_strings)

        return IntervalTree._init_from_java(jrep)

    @staticmethod
    @handle_py4j
    def read(path):
        """Read an interval tree from a file.

        **The File Format**

        Hail expects an interval file to contain either three or five fields per
        line in the following formats:

        - ``contig:start-end``
        - ``contig  start  end`` (tab-separated)
        - ``contig  start  end  direction  target`` (tab-separated)

        .. note::

            ``start`` and ``end`` match positions inclusively, e.g. ``start <= position <= end``

        .. note::

            Hail uses the following ordering for contigs: 1-22 sorted numerically, then X, Y, MT,
            then alphabetically for any contig not matching the standard human chromosomes.

        .. caution::

            The interval parser for these files does not support the full range of formats supported
            by the python parser :py:meth:`.Interval.parse`.  'k', 'm', 'start', and 'end' are all
            invalid motifs in the ``contig:start-end`` format here.

        :param str path: file name

        :rtype: :class:`.IntervalTree`
        """

        print('hadoop conf is %s' % str(env.hc._jhc.hadoopConf()))
        print('hadoop conf is %s' % str(env.hc._jhc.hadoopConf))
        jrep = scala_object(env.hail.io.annotators, 'IntervalListAnnotator').read(path,
                                                                                  env.hc._jhc.hadoopConf(),
                                                                                  True)

        return IntervalTree._init_from_java(jrep)

    @property
    def intervals(self):
        """Access a sorted iterator of genomic intervals.

        :rtype: iterator of :class:`.Interval`
        """

        return (Interval._from_java(j_interval) for j_interval in
                env.jutils.iterableToArrayList(self._jrep.toIterable()))
