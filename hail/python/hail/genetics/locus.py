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

        self._contig = contig
        self._position = position
        self._rg = reference_genome

    def __str__(self):
        return f'{self._contig}:{self._position}'

    def __repr__(self):
        return 'Locus(contig=%s, position=%s, reference_genome=%s)' % (self.contig, self.position, self._rg)

    def __eq__(self, other):
        return (isinstance(other, Locus)
                and self._contig == other._contig
                and self._position == other._position
                and self._rg == other._rg)

    def __hash__(self):
        return hash(self._contig) ^ hash(self._position) ^ hash(self._rg)

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
        contig, pos = string.split(':')
        if pos.lower() == 'end':
            pos = reference_genome.contig_length(contig)
        else:
            pos = int(pos)
        return Locus(contig, pos, reference_genome)

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
