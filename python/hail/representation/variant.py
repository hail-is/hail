from hail.java import scala_object, Env, handle_py4j
from hail.typecheck import *

class Variant(object):
    """
    An object that represents a genomic polymorphism.

    .. testsetup::

        v_biallelic = Variant.parse('16:20012:A:TT')
        v_multiallelic = Variant.parse('16:12311:T:C,TTT,A')

    :param contig: chromosome identifier
    :type contig: str or int
    :param int start: chromosomal position (1-based)
    :param str ref: reference allele
    :param alts: single alternate allele, or list of alternate alleles
    :type alts: str or list of str
    """

    @handle_py4j
    def __init__(self, contig, start, ref, alts):
        if isinstance(contig, int):
            contig = str(contig)
        jrep = scala_object(Env.hail().variant, 'Variant').apply(contig, start, ref, alts)
        self._init_from_java(jrep)
        self._contig = contig
        self._start = start
        self._ref = ref

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'Variant(contig=%s, start=%s, ref=%s, alts=%s)' % (self.contig, self.start, self.ref, self._alt_alleles)

    def __eq__(self, other):
        return self._jrep.equals(other._jrep)

    def __hash__(self):
        return self._jrep.hashCode()

    def _init_from_java(self, jrep):
        self._jrep = jrep
        self._alt_alleles = map(AltAllele._from_java, [jrep.altAlleles().apply(i) for i in xrange(jrep.nAltAlleles())])

    @classmethod
    def _from_java(cls, jrep):
        v = Variant.__new__(cls)
        v._init_from_java(jrep)
        v._contig = jrep.contig()
        v._start = jrep.start()
        v._ref = jrep.ref()
        return v

    @staticmethod
    @handle_py4j
    @typecheck(string=strlike)
    def parse(string):
        """Parses a variant object from a string.

        There are two acceptable formats: CHR:POS:REF:ALT, and
        CHR:POS:REF:ALT1,ALT2,...ALTN.  Below is an example of
        each:

        >>> v_biallelic = Variant.parse('16:20012:A:TT')
        >>> v_multiallelic = Variant.parse('16:12311:T:C,TTT,A')

        :rtype: :class:`.Variant`
        """
        jrep = scala_object(Env.hail().variant, 'Variant').parse(string)
        return Variant._from_java(jrep)

    @property
    def contig(self):
        """
        Chromosome identifier.

        :rtype: str
        """
        return self._contig

    @property
    def start(self):
        """
        Chromosomal position (1-based).

        :rtype: int
        """
        return self._start

    @property
    def ref(self):
        """
        Reference allele at this locus.

        :rtype: str
        """

        return self._ref

    @property
    def alt_alleles(self):
        """
        List of alternate allele objects in this polymorphism.

        :rtype: list of :class:`.AltAllele`
        """
        return self._alt_alleles

    def num_alt_alleles(self):
        """Returns the number of alternate alleles in this polymorphism.

        :rtype: int
        """

        return self._jrep.nAltAlleles()

    def is_biallelic(self):
        """True if there is only one alternate allele in this polymorphism.

        :rtype: bool
        """

        return self._jrep.isBiallelic()

    def alt_allele(self):
        """Returns the alternate allele object, assumes biallelic.

        Fails if called on a multiallelic variant.

        :rtype: :class:`.AltAllele`
        """

        return AltAllele._from_java(self._jrep.altAllele())

    def alt(self):
        """Returns the alternate allele string, assumes biallelic.

        Fails if called on a multiallelic variant.

        :rtype: str
        """

        return self._jrep.alt()

    def num_alleles(self):
        """Returns the number of total alleles in this polymorphism, including the reference.

        :rtype: int
        """

        return self._jrep.nAlleles()

    @handle_py4j
    @typecheck_method(i=integral)
    def allele(self, i):
        """Returns the string allele representation for the ith allele.

        The reference is included in the allele index.  The index of
        the first alternate allele is 1.  The following is true for all
        variants:

        >>> v_multiallelic.ref == v_multiallelic.allele(0)

        Additionally, the following is true for all biallelic variants:

        >>> v_biallelic.alt == v_biallelic.allele(1)

        :param int i: integer index of desired allele

        :return: string representation of ith allele
        :rtype: str
        """

        return self._jrep.allele(i)

    def num_genotypes(self):
        """Returns the total number of unique genotypes possible for this variant.

        For a biallelic variant, this value is 3: 0/0, 0/1, and 1/1.

        For a triallelic variant, this value is 6: 0/0, 0/1, 1/1, 0/2, 1/2, 2/2.

        For a variant with N alleles, this value is:

        .. math::

          \\frac{N * (N + 1)}{2}

        :rtype: int"""

        return self._jrep.nGenotypes()

    def locus(self):
        """Returns the locus object for this polymorphism.

        :rtype: :class:`.Locus`
        """
        return Locus._from_java(self._jrep.locus())

    def is_autosomal_or_pseudoautosomal(self):
        """True if this polymorphism is found on an autosome, or the PAR on X or Y.

        :rtype: bool
        """
        return self._jrep.isAutosomalOrPseudoAutosomal()

    def is_autosomal(self):
        """True if this polymorphism is located on an autosome.

        :rtype: bool
        """
        return self._jrep.isAutosomal()

    def is_mitochondrial(self):
        """True if this polymorphism is mapped to mitochondrial DNA.

        :rtype: bool
        """

        return self._jrep.isMitochondrial()

    def in_X_PAR(self):
        """True of this polymorphism is found on the pseudoautosomal region of chromosome X.

        :rtype: bool
        """

        return self._jrep.inXPar()

    def in_Y_PAR(self):
        """True of this polymorphism is found on the pseudoautosomal region of chromosome Y.

        :rtype: bool
        """

        return self._jrep.inYPar()

    def in_X_non_PAR(self):
        """True of this polymorphism is found on the non-pseudoautosomal region of chromosome X.

        :rtype: bool
        """

        return self._jrep.inXNonPar()

    def in_Y_non_PAR(self):
        """True of this polymorphism is found on the non-pseudoautosomal region of chromosome Y.

        :rtype: bool
        """

        return self._jrep.inYNonPar()


class AltAllele(object):
    """
    An object that represents an allele in a polymorphism deviating from the reference allele.

    :param str ref: reference allele
    :param str alt: alternate allele
    """

    @handle_py4j
    def __init__(self, ref, alt):
        jaa = scala_object(Env.hail().variant, 'AltAllele').apply(ref, alt)
        self._init_from_java(jaa)
        self._ref = ref
        self._alt = alt

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'AltAllele(ref=%s, alt=%s)' % (self.ref, self.alt)

    def __eq__(self, other):
        return self._jrep.equals(other._jrep)

    def __hash__(self):
        return self._jrep.hashCode()

    def _init_from_java(self, jrep):
        self._jrep = jrep

    @classmethod
    def _from_java(cls, jaa):
        aa = AltAllele.__new__(cls)
        aa._init_from_java(jaa)
        aa._ref = jaa.ref()
        aa._alt = jaa.alt()
        return aa

    @property
    def ref(self):
        """
        Reference allele.

        :rtype: str
        """
        return self._ref

    @property
    def alt(self):
        """
        Alternate allele.

        :rtype: str
        """
        return self._alt

    def num_mismatch(self):
        """Returns the number of mismatched bases in this alternate allele.

        Fails if the ref and alt alleles are not the same length.

        :rtype: int
        """

        return self._jrep.nMismatch()

    def stripped_snp(self):
        """Returns the one-character reduced SNP.

        Fails if called on an alternate allele that is not a SNP.

        :rtype: str, str
        """

        r = self._jrep.strippedSNP()
        return r._1(), r._2()

    def is_SNP(self):
        """True if this alternate allele is a single nucleotide polymorphism (SNP)

        :rtype: bool
        """

        return self._jrep.isSNP()

    def is_MNP(self):
        """True if this alternate allele is a multiple nucleotide polymorphism (MNP)

        :rtype: bool
        """

        return self._jrep.isMNP()

    def is_insertion(self):
        """True if this alternate allele is an insertion of one or more bases

        :rtype: bool
        """

        return self._jrep.isInsertion()

    def is_deletion(self):
        """True if this alternate allele is a deletion of one or more bases

        :rtype: bool
        """

        return self._jrep.isDeletion()

    def is_indel(self):
        """True if this alternate allele is either an insertion or deletion of one or more bases

        :rtype: bool
        """

        return self._jrep.isIndel()

    def is_complex(self):
        """True if this alternate allele does not fit into the categories of SNP, MNP, Insertion, or Deletion

        :rtype: bool
        """

        return self._jrep.isComplex()

    def is_transition(self):
        """True if this alternate allele is a transition SNP.

        This is true if the reference and alternate bases are
        both purine (A/G) or both pyrimidine (C/T). This method
        raises an exception if the polymorphism is not a SNP.

        :rtype: bool
        """

        return self._jrep.isTransition()

    def is_transversion(self):
        """True if this alternate allele is a transversion SNP.

        This is true if the reference and alternate bases contain
        one purine (A/G) and one pyrimidine (C/T). This method
        raises an exception if the polymorphism is not a SNP.

        :rtype: bool
        """

        return self._jrep.isTransversion()


class Locus(object):
    """
    An object that represents a location in the genome.

    :param contig: chromosome identifier
    :type contig: str or int
    :param int position: chromosomal position (1-indexed)
    """

    @handle_py4j
    def __init__(self, contig, position):
        if isinstance(contig, int):
            contig = str(contig)
        jrep = scala_object(Env.hail().variant, 'Locus').apply(contig, position)
        self._init_from_java(jrep)
        self._contig = contig
        self._position = position

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'Locus(contig=%s, position=%s)' % (self.contig, self.position)

    def __eq__(self, other):
        return self._jrep.equals(other._jrep)

    def __hash__(self):
        return self._jrep.hashCode()

    def _init_from_java(self, jrep):
        self._jrep = jrep

    @classmethod
    def _from_java(cls, jrep):
        l = Locus.__new__(cls)
        l._init_from_java(jrep)
        l._contig = jrep.contig()
        l._position = jrep.position()
        return l

    @staticmethod
    @handle_py4j
    @typecheck(string=strlike)
    def parse(string):
        """Parses a locus object from a CHR:POS string.

        :rtype: :class:`.Locus`
        """

        return Locus._from_java(scala_object(Env.hail().variant, 'Locus').parse(string))

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
