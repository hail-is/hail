from hail.genetics.reference_genome import ReferenceGenome, reference_genome_type
from hail.typecheck import *
from hail.utils.java import scala_object, Env
import hail as hl

class Locus(object):
    """
    An object that represents a location in the genome.

    :param contig: chromosome identifier
    :type contig: str or int
    :param int position: chromosomal position (1-indexed)
    :param reference_genome: Reference genome to use.
    :type reference_genome: :obj:`str` or :class:`.ReferenceGenome`
    """

    @typecheck_method(contig=oneof(str, int),
                      position=int,
                      reference_genome=reference_genome_type)
    def __init__(self, contig, position, reference_genome='default'):
        if isinstance(contig, int):
            contig = str(contig)

        self._rg = reference_genome

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
    @typecheck_method(string=str,
                      reference_genome=reference_genome_type)
    def parse(cls, string, reference_genome='default'):
        """Parses a locus object from a CHR:POS string.

        **Examples**

        >>> l1 = hl.Locus.parse('1:101230')
        >>> l2 = hl.Locus.parse('X:4201230')

        :param str string: String to parse.
        :param reference_genome: Reference genome to use. Default is :func:`~hail.default_reference`.
        :type reference_genome: :obj:`str` or :class:`.ReferenceGenome`

        :rtype: :class:`.Locus`
        """

        return Locus._from_java(scala_object(Env.hail().variant, 'Locus').parse(string, reference_genome._jrep), reference_genome)

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
    def reference_genome(self):
        """Reference genome.

        :return: :class:`.ReferenceGenome`
        """
        return self._rg
