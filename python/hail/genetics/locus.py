from hail.genetics.genomeref import GenomeReference
from hail.history import *
from hail.typecheck import *
from hail.utils.java import scala_object, handle_py4j, Env


class Locus(HistoryMixin):
    """
    An object that represents a location in the genome.

    :param contig: chromosome identifier
    :type contig: str or int
    :param int position: chromosomal position (1-indexed)
    :param reference_genome: Reference genome to use. Default is :meth:`hail.default_reference`.
    :type reference_genome: :class:`.GenomeReference`
    """

    @handle_py4j
    @record_init
    @typecheck_method(contig=oneof(strlike, integral),
                      position=integral,
                      reference_genome=nullable(GenomeReference))
    def __init__(self, contig, position, reference_genome=None):
        if isinstance(contig, int):
            contig = str(contig)
        self._rg = reference_genome if reference_genome else Env.hc().default_reference
        jrep = scala_object(Env.hail().variant, 'Locus').apply(contig, position, self._rg._jrep)
        self._init_from_java(jrep)
        self._contig = contig
        self._position = position

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'Locus(contig=%s, position=%s, reference_genome=%s)' % (self.contig, self.position, self._rg)

    def __eq__(self, other):
        return isinstance(other, Locus) and self._jrep.equals(other._jrep) and self._rg._jrep == other._rg._jrep

    def __hash__(self):
        return self._jrep.hashCode()

    def _init_from_java(self, jrep):
        self._jrep = jrep

    @classmethod
    def _from_java(cls, jrep, reference_genome):
        l = Locus.__new__(cls)
        l._init_from_java(jrep)
        l._contig = jrep.contig()
        l._position = jrep.position()
        l._rg = reference_genome
        reference_genome._check_locus(jrep)
        super(Locus, l).__init__()
        return l

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(string=strlike,
                      reference_genome=nullable(GenomeReference))
    def parse(cls, string, reference_genome=None):
        """Parses a locus object from a CHR:POS string.

        **Examples**

        >>> l1 = Locus.parse('1:101230')
        >>> l2 = Locus.parse('X:4201230')

        :param str string: String to parse.
        :param reference_genome: Reference genome to use. Default is :meth:`hail.default_reference`.
        :type reference_genome: :class:`.GenomeReference`

        :rtype: :class:`.Locus`
        """
        rg = reference_genome if reference_genome else Env.hc().default_reference
        return Locus._from_java(scala_object(Env.hail().variant, 'Locus').parse(string, rg._jrep), rg)

    @property
    def contig(self):
        """
        Chromosome identifier.
        :rtype: str
        """
        return self._contig

    @property
    def position(self):
        """
        Chromosomal position (1-based).
        :rtype: int
        """
        return self._position

    @property
    @record_property
    def reference_genome(self):
        """Reference genome.

        :return: :class:`.GenomeReference`
        """
        return self._rg
