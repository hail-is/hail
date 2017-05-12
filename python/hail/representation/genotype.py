from hail.java import *


class Genotype(object):
    """
    An object that represents an individual's genotype at a genomic locus.

    .. testsetup::

        g = Genotype(0, ad=[9,1], dp=11, gq=20, pl=[0,100,1000])

    :param gt: Genotype hard call
    :type gt: int or None
    :param ad: allelic depth (1 element per allele including reference)
    :type ad: list of int or None
    :param dp: total depth
    :type dp: int or None
    :param gq: genotype quality
    :type gq: int or None
    :param pl: phred-scaled posterior genotype likelihoods (1 element per possible genotype)
    :type pl: list of int or None
    """

    @handle_py4j
    def __init__(self, gt, ad=None, dp=None, gq=None, pl=None):
        """Initialize a Genotype object."""

        jvm = Env.jvm()
        jgt = joption(gt)
        if ad:
            jad = jsome(jarray(jvm.int, ad))
        else:
            jad = jnone()
        jdp = joption(dp)
        jgq = joption(gq)
        if pl:
            jpl = jsome(jarray(jvm.int, pl))
        else:
            jpl = jnone()

        jrep = scala_object(Env.hail().variant, 'Genotype').apply(
            jgt, jad, jdp, jgq, jpl, False, False)
        self._gt = gt
        self._ad = ad
        self._dp = dp
        self._gq = gq
        self._pl = pl
        self._init_from_java(jrep)

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        fake_ref = 'FakeRef=True' if self._jrep.fakeRef() else ''
        if self._jrep.isLinearScale():
            return 'Genotype(GT=%s, AD=%s, DP=%s, GQ=%s, GP=%s%s)' %\
                   (self.gt, self.ad, self.dp, self.gq, self.gp(), fake_ref)
        else:
            return 'Genotype(GT=%s, AD=%s, DP=%s, GQ=%s, PL=%s%s)' % \
                   (self.gt, self.ad, self.dp, self.gq, self.pl, fake_ref)

    def __eq__(self, other):
        return self._jrep.equals(other._jrep)

    def __hash__(self):
        return self._jrep.hashCode()

    def _init_from_java(self, jrep):
        self._jrep = jrep

    @classmethod
    def _from_java(cls, jrep):
        g = Genotype.__new__(cls)
        g._init_from_java(jrep)
        g._gt = from_option(jrep.gt())
        g._ad = jarray_to_list(from_option(jrep.ad()))
        g._dp = from_option(jrep.dp())
        g._gq = from_option(jrep.gq())
        g._pl = jarray_to_list(from_option(jrep.pl()))
        return g

    @property
    def gt(self):
        """Returns the hard genotype call.

        :rtype: int or None
        """

        return self._gt

    @property
    def ad(self):
        """Returns the allelic depth.

        :rtype: list of int or None
        """

        return self._ad

    @property
    def dp(self):
        """Returns the total depth.

        :rtype: int or None
        """

        return self._dp

    @property
    def gq(self):
        """Returns the phred-scaled genotype quality.

        :return: int or None
        """

        return self._gq

    @property
    def pl(self):
        """Returns the phred-scaled genotype posterior likelihoods.

        :rtype: list of int or None
        """

        return self._pl

    def od(self):
        """Returns the difference between the total depth and the allelic depth sum.

        Equivalent to:

        .. testcode::

            g.dp - sum(g.ad)

        :rtype: int or None
        """

        return from_option(self._jrep.od())

    @property
    def gp(self):
        """Returns the linear-scaled genotype probabilities.

        :rtype: list of float of None
        """

        return jarray_to_list(from_option(self._jrep.gp()))

    def dosage(self):
        """Returns the expected value of the genotype based on genotype probabilities,
        :math:`\\mathrm{P}(\\mathrm{Het}) + 2 \\mathrm{P}(\\mathrm{HomVar})`. Genotype must be bi-allelic.

        :rtype: float
        """

        return from_option(self._jrep.dosage())

    def is_hom_ref(self):
        """True if the genotype call is 0/0

        :rtype: bool
        """

        return self._jrep.isHomRef()

    def is_het(self):
        """True if the genotype call contains two different alleles.

        :rtype: bool
        """

        return self._jrep.isHet()

    def is_hom_var(self):
        """True if the genotype call contains two identical alternate alleles.

        :rtype: bool
        """

        return self._jrep.isHomVar()

    def is_called_non_ref(self):
        """True if the genotype call contains any non-reference alleles.

        :rtype: bool
        """

        return self._jrep.isCalledNonRef()

    def is_het_non_ref(self):
        """True if the genotype call contains two different alternate alleles.

        :rtype: bool
        """

        return self._jrep.isHetNonRef()

    def is_het_ref(self):
        """True if the genotype call contains one reference and one alternate allele.

        :rtype: bool
        """

        return self._jrep.isHetRef()

    def is_not_called(self):
        """True if the genotype call is missing.

        :rtype: bool
        """

        return self._jrep.isNotCalled()

    def is_called(self):
        """True if the genotype call is non-missing.

        :rtype: bool
        """

        return self._jrep.isCalled()

    def num_alt_alleles(self):
        """Returns the count of non-reference alleles.

        This function returns None if the genotype call is missing.

        :rtype: int or None
        """

        return from_option(self._jrep.nNonRefAlleles())

    @handle_py4j
    def one_hot_alleles(self, num_alleles):
        """Returns a list containing the one-hot encoded representation of the called alleles.

        This one-hot representation is the positional sum of the one-hot
        encoding for each called allele.  For a biallelic variant, the
        one-hot encoding for a reference allele is [1, 0] and the one-hot
        encoding for an alternate allele is [0, 1].  Thus, with the
        following variables:

        .. testcode::

            num_alleles = 2
            hom_ref = Genotype(0)
            het = Genotype(1)
            hom_var = Genotype(2)

        All the below statements are true:

        .. testcode::

            hom_ref.one_hot_alleles(num_alleles) == [2, 0]
            het.one_hot_alleles(num_alleles) == [1, 1]
            hom_var.one_hot_alleles(num_alleles) == [0, 2]

        This function returns None if the genotype call is missing.

        :param int num_alleles: number of possible alternate alleles
        :rtype: list of int or None
        """
        return jiterable_to_list(from_option(self._jrep.oneHotAlleles(num_alleles)))

    @handle_py4j
    def one_hot_genotype(self, num_genotypes):
        """Returns a list containing the one-hot encoded representation of the genotype call.

        A one-hot encoding is a vector with one '1' and many '0' values, like
        [0, 0, 1, 0] or [1, 0, 0, 0].  This function is useful for transforming
        the genotype call (gt) into a one-hot encoded array.  With the following
        variables:

        .. testcode::

            num_genotypes = 3
            hom_ref = Genotype(0)
            het = Genotype(1)
            hom_var = Genotype(2)

        All the below statements are true:

        .. testcode::

            hom_ref.one_hot_genotype(num_genotypes) == [1, 0, 0]
            het.one_hot_genotype(num_genotypes) == [0, 1, 0]
            hom_var.one_hot_genotype(num_genotypes) == [0, 0, 1]

        This function returns None if the genotype call is missing.

        :param int num_genotypes: number of possible genotypes
        :rtype: list of int or None
        """

        return jiterable_to_list(from_option(self._jrep.oneHotGenotype(num_genotypes)))

    @handle_py4j
    def p_ab(self, theta=0.5):
        """Returns the p-value associated with finding the given allele depth ratio.

        This function uses a one-tailed binomial test.

        This function returns None if the allelic depth (ad) is missing.

        :param float theta: null reference probability for binomial model
        :rtype: float or None
        """

        return from_option(self._jrep.pAB(theta))

    def fraction_reads_ref(self):
        """Returns the fraction of reads that are reference reads.

        Equivalent to:

        >>> g.ad[0] / sum(g.ad)

        :rtype: float or None
        """

        return from_option(self._jrep.fractionReadsRef())


class Call(object):
    """
    An object that represents an individual's call at a genomic locus.

    :param call: Genotype hard call
    :type call: int or None
    """

    _call_jobject = None

    @handle_py4j
    def __init__(self, call):
        """Initialize a Call object."""

        if not Call._call_jobject:
            Call._call_jobject = scala_object(Env.hail().variant, 'Call')

        jrep = Call._call_jobject.apply(call)
        self._init_from_java(jrep)

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'Call(gt=%s)' % self._jrep

    def __eq__(self, other):
        return self._jrep.equals(other._jrep)

    def __hash__(self):
        return self._jrep.hashCode()

    def _init_from_java(self, jrep):
        self._jcall = Call._call_jobject
        self._jrep = jrep

    @classmethod
    def _from_java(cls, jrep):
        c = Call.__new__(cls)
        c._init_from_java(jrep)
        return c

    @property
    def gt(self):
        """Returns the hard call.

        :rtype: int or None
        """

        return self._jrep

    def is_hom_ref(self):
        """True if the call is 0/0

        :rtype: bool
        """

        return self._jcall.isHomRef(self._jrep)

    def is_het(self):
        """True if the call contains two different alleles.

        :rtype: bool
        """

        return self._jcall.isHet(self._jrep)

    def is_hom_var(self):
        """True if the call contains two identical alternate alleles.

        :rtype: bool
        """

        return self._jcall.isHomVar(self._jrep)

    def is_called_non_ref(self):
        """True if the call contains any non-reference alleles.

        :rtype: bool
        """

        return self._jcall.isCalledNonRef(self._jrep)

    def is_het_non_ref(self):
        """True if the call contains two different alternate alleles.

        :rtype: bool
        """

        return self._jcall.isHetNonRef(self._jrep)

    def is_het_ref(self):
        """True if the call contains one reference and one alternate allele.

        :rtype: bool
        """

        return self._jcall.isHetRef(self._jrep)

    def is_not_called(self):
        """True if the call is missing.

        :rtype: bool
        """

        return self._jcall.isNotCalled(self._jrep)

    def is_called(self):
        """True if the call is non-missing.

        :rtype: bool
        """

        return self._jcall.isCalled(self._jrep)

    def num_alt_alleles(self):
        """Returns the count of non-reference alleles.

        This function returns None if the genotype call is missing.

        :rtype: int or None
        """

        return self._jcall.nNonRefAlleles(self._jrep)

    @handle_py4j
    def one_hot_alleles(self, num_alleles):
        """Returns a list containing the one-hot encoded representation of the called alleles.

        This one-hot representation is the positional sum of the one-hot
        encoding for each called allele.  For a biallelic variant, the
        one-hot encoding for a reference allele is [1, 0] and the one-hot
        encoding for an alternate allele is [0, 1].  Thus, with the
        following variables:

        .. testcode::

            num_alleles = 2
            hom_ref = Call(0)
            het = Call(1)
            hom_var = Call(2)

        All the below statements are true:

        .. testcode::

            hom_ref.one_hot_alleles(num_alleles) == [2, 0]
            het.one_hot_alleles(num_alleles) == [1, 1]
            hom_var.one_hot_alleles(num_alleles) == [0, 2]

        This function returns None if the call is missing.

        :param int num_alleles: number of possible alternate alleles
        :rtype: list of int or None
        """
        return jiterable_to_list(self._jcall.oneHotAlleles(self._jrep, num_alleles))

    @handle_py4j
    def one_hot_genotype(self, num_genotypes):
        """Returns a list containing the one-hot encoded representation of the genotype call.

        A one-hot encoding is a vector with one '1' and many '0' values, like
        [0, 0, 1, 0] or [1, 0, 0, 0].  This function is useful for transforming
        the genotype call (gt) into a one-hot encoded array.  With the following
        variables:

        .. testcode::

            num_genotypes = 3
            hom_ref = Call(0)
            het = Call(1)
            hom_var = Call(2)

        All the below statements are true:

        .. testcode::

            hom_ref.one_hot_genotype(num_genotypes) == [1, 0, 0]
            het.one_hot_genotype(num_genotypes) == [0, 1, 0]
            hom_var.one_hot_genotype(num_genotypes) == [0, 0, 1]

        This function returns None if the call is missing.

        :param int num_genotypes: number of possible genotypes
        :rtype: list of int or None
        """

        return jiterable_to_list(self._jcall.oneHotGenotype(self._jrep, num_genotypes))
    