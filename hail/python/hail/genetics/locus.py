from typing import Union

import hail as hl
from hail.genetics.reference_genome import reference_genome_type, ReferenceGenome
from hail.typecheck import typecheck_method


class Locus(object):
    """An object that represents a location in the genome.

    Parameters
    ----------
    contig : :class:`str`
        Chromosome identifier.
    position : :obj:`int`
        Chromosomal position (1-indexed).
    reference_genome : :class:`str` or :class:`.ReferenceGenome`
        Reference genome to use.

    Note
    ----
    This object refers to the Python value returned by taking or collecting
    Hail expressions, e.g. ``mt.locus.take(5)``. This is rare; it is much
    more common to manipulate the :class:`.LocusExpression` object, which is
    constructed using the following functions:

     - :func:`.locus`
     - :func:`.parse_locus`
     - :func:`.locus_from_global_position`
    """

    def __init__(self, contig, position, reference_genome: Union[str, ReferenceGenome] = 'default'):
        if isinstance(contig, int):
            contig = str(contig)

        if isinstance(reference_genome, str):
            reference_genome = hl.get_reference(reference_genome)

        assert isinstance(contig, str)
        assert isinstance(position, int)
        assert isinstance(reference_genome, ReferenceGenome)

        self._contig = contig
        self._position = position
        self._rg = reference_genome

    def __str__(self):
        return f'{self._contig}:{self._position}'

    def __repr__(self):
        return 'Locus(contig=%s, position=%s, reference_genome=%s)' % (self.contig, self.position, self._rg)

    def __eq__(self, other):
        return ( self._contig   == other._contig and
                 self._position == other._position and
                 self._rg       == other._rg
               ) if isinstance(other, Locus) else NotImplemented

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
        :type reference_genome: :class:`str` or :class:`.ReferenceGenome`

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
