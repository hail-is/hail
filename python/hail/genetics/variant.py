from hail.genetics.genomeref import GenomeReference
from hail.history import *
from hail.typecheck import *
from hail.utils.java import scala_object, handle_py4j, Env


class Variant(HistoryMixin):
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
    :param reference_genome: Reference genome to use. Default is :meth:`hail.api1.HailContext.default_reference`.
    :type reference_genome: :class:`.GenomeReference`
    """

    @handle_py4j
    @record_init
    @typecheck_method(contig=oneof(strlike, integral),
                   start=integral,
                   ref=strlike,
                   alts=oneof(strlike, listof(strlike)),
                   reference_genome=nullable(GenomeReference))
    def __init__(self, contig, start, ref, alts, reference_genome=None):
        if isinstance(contig, int):
            contig = str(contig)
        self._rg = reference_genome if reference_genome else Env.hc().default_reference
        jrep = scala_object(Env.hail().variant, 'Variant').apply(contig, start, ref, alts, self._rg._jrep)
        self._init_from_java(jrep)
        self._contig = contig
        self._start = start
        self._ref = ref

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'Variant(contig=%s, start=%s, ref=%s, alts=%s, reference_genome=%s)' % (self.contig, self.start, self.ref, self._alt_alleles, self._rg)

    def __eq__(self, other):
        return isinstance(other, Variant) and self._jrep.equals(other._jrep) and self._rg._jrep == other._rg._jrep

    def __hash__(self):
        return self._jrep.hashCode()

    def _init_from_java(self, jrep):
        self._jrep = jrep
        self._alt_alleles = map(AltAllele._from_java, [jrep.altAlleles().apply(i) for i in xrange(jrep.nAltAlleles())])

    @classmethod
    def _from_java(cls, jrep, reference_genome):
        v = Variant.__new__(cls)
        v._init_from_java(jrep)
        v._contig = jrep.contig()
        v._start = jrep.start()
        v._ref = jrep.ref()
        v._rg = reference_genome
        reference_genome._check_variant(jrep)
        super(Variant, v).__init__()
        return v

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(string=strlike,
                      reference_genome=nullable(GenomeReference))
    def parse(cls, string, reference_genome=None):
        """Parses a variant object from a string.

        There are two acceptable formats: CHR:POS:REF:ALT, and
        CHR:POS:REF:ALT1,ALT2,...ALTN.  Below is an example of
        each:

        >>> v_biallelic = Variant.parse('16:20012:A:TT')
        >>> v_multiallelic = Variant.parse('16:12311:T:C,TTT,A')

        :param str string: String to parse.
        :param reference_genome: Reference genome to use. Default is :meth:`hail.api1.HailContext.default_reference`.
        :type reference_genome: :class:`.GenomeReference`

        :rtype: :class:`.Variant`
        """
        rg = reference_genome if reference_genome else Env.hc().default_reference
        jrep = scala_object(Env.hail().variant, 'Variant').parse(string, rg._jrep)
        return Variant._from_java(jrep, rg)

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

    @property
    @record_property
    def reference_genome(self):
        """Reference genome.

        :return: :class:`.GenomeReference`
        """
        return self._rg

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

    @record_method
    def locus(self):
        """Returns the locus object for this polymorphism.

        :rtype: :class:`.Locus`
        """
        return Locus._from_java(self._jrep.locus(), self._rg)

    def in_autosome(self):
        """True if this polymorphism is on an autosome.
        (not an X, Y, or MT contig).

        :rtype: bool
        """
        return self._jrep.isAutosomal(self._rg._jrep)

    def in_autosome_or_par(self):
        """True if this polymorphism is on an autosome,
        or a pseudoautosomal region on chromosome X or Y.

        :rtype: bool
        """
        return self._jrep.isAutosomalOrPseudoAutosomal(self._rg._jrep)

    def in_mito(self):
        """True if this polymorphism is on mitochondrial DNA.

        :rtype: bool
        """

        return self._jrep.isMitochondrial(self._rg._jrep)

    def in_x_par(self):
        """True if this polymorphism is on a pseudoautosomal region of chromosome X.

        :rtype: bool
        """

        return self._jrep.inXPar(self._rg._jrep)

    def in_y_par(self):
        """True if this polymorphism is on a pseudoautosomal region of chromosome Y.

        :rtype: bool
        """

        return self._jrep.inYPar(self._rg._jrep)

    def in_x_nonpar(self):
        """True if this polymorphism is on a non-pseudoautosomal region of chromosome X.

        :rtype: bool
        """

        return self._jrep.inXNonPar(self._rg._jrep)

    def in_y_nonpar(self):
        """True if this polymorphism is on a non-pseudoautosomal region of chromosome Y.

        :rtype: bool
        """

        return self._jrep.inYNonPar(self._rg._jrep)


class AltAllele(HistoryMixin):
    """
    An object that represents an allele in a polymorphism deviating from the reference allele.

    :param str ref: reference allele
    :param str alt: alternate allele
    """

    @handle_py4j
    @record_init
    @typecheck_method(ref=strlike,
                   alt=strlike)
    def __init__(self, ref, alt):
        jaa = scala_object(Env.hail().variant, 'AltAllele').apply(ref, alt)
        self._init_from_java(jaa)
        self._ref = ref
        self._alt = alt

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return "AltAllele(ref='{ref}', alt='{alt}')".format(ref=self.ref, alt=self.alt)

    def __eq__(self, other):
        return isinstance(other, AltAllele) and self._jrep.equals(other._jrep)

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
        super(AltAllele, aa).__init__()
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

    @handle_py4j
    def category(self):
        """Returns the type of alt, i.e one of
            SNP,
            Insertion,
            Deletion,
            Star,
            MNP,
            Complex

        :rtype: str
        """
        return self._jrep.altAlleleType()


class Locus(HistoryMixin):
    """
    An object that represents a location in the genome.

    :param contig: chromosome identifier
    :type contig: str or int
    :param int position: chromosomal position (1-indexed)
    :param reference_genome: Reference genome to use. Default is :meth:`hail.api1.HailContext.default_reference`.
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
        :param reference_genome: Reference genome to use. Default is :meth:`hail.api1.HailContext.default_reference`.
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
