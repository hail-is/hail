from hail.java import *


class Genotype(object):
    def __init__(self, gt, ad=None, dp=None, gq=None, pl=None):
        """Initialize a Genotype object.

        :param int gt: Genotype hard call
        :param ad: allelic depth
        :type ad: list of int
        :param int dp: total depth
        :param int gq: genotype quality
        :param pl: phred-scaled posterior genotype likelihoods
        :type pl: list of int
        """

        jvm = Env.jvm()
        gt = joption(gt)
        if ad:
            ad = jsome(jarray(jvm.int, ad))
        else:
            ad = jnone()
        dp = joption(dp)
        gq = joption(gq)
        if pl:
            pl = jsome(jarray(jvm.int, pl))
        else:
            pl = jnone()

        jrep = scala_object(Env.hail_package().variant, 'Genotype').apply(gt, ad, dp, gq, pl, False, False)
        self._init_from_java(jrep)

    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'Genotype(%s, %s, %s, %s, %s' % (self.gt(), self.ad(), self.dp(), self.gq(), self.pl())

    def __eq__(self, other):
        return self._jrep.equals(other._jrep)

    def _init_from_java(self, jrep):
        self._jrep = jrep

    @classmethod
    def _from_java(cls, jrep):
        l = Genotype.__new__(cls)
        l._init_from_java(jrep)
        return l

    def gt(self):
        """Returns the hard genotype call.

        :rtype: int or None
        """

        return strip_option(self._jrep.gt())

    def ad(self):
        """Returns the allelic depth.

        :rtype: list of int or None
        """

        result = strip_option(self._jrep.ad())
        if result:
            result = [x for x in scala_package_object(Env.hail_package().utils).arrayToArrayList(result)]
        return result

    def dp(self):
        """Returns the total depth.

        :rtype: int or None
        """

        return strip_option(self._jrep.dp())

    def gq(self):
        """Returns the phred-scaled genotype quality.

        :return: int or None
        """

        return strip_option(self._jrep.gq())

    def pl(self):
        """Returns the phred-scaled genotype posterior likelihoods.

        :rtype: list of int or None
        """

        result = strip_option(self._jrep.pl())
        if result:
            result = [x for x in scala_package_object(Env.hail_package().utils).arrayToArrayList(result)]
        return result

    def od(self):
        """Returns the difference between the total depth and the allelic depth sum.

        Equivalent to:

        >>> g.dp - sum(g.ad)

        :rtype: int or None
        """

        return strip_option(self._jrep.od())

    def dosage(self):
        """Returns the linear-scaled genotype probabilities.

        :rtype list of float
        """

        return strip_option(self._jrep.dosage())

    def is_hom_ref(self):
        """True if the genotype call is 0/0

        :rtype bool
        """

        return self._jrep.isHomRef()

    def is_het(self):
        """True if the genotype call contains two different alleles.

        :rtype bool
        """

        return self._jrep.isHet()

    def is_hom_var(self):
        """True if the genotype call contains two identical alternate alleles.

        :rtype bool
        """

        return self._jrep.isHomVar()

    def is_called_non_ref(self):
        """True if the genotype call contains any non-reference alleles.

        :rtype bool
        """

        return self._jrep.isCalledNonRef()

    def is_het_non_ref(self):
        """True if the genotype call contains two different alternate alleles.

        :rtype bool
        """

        return self._jrep.isHetNonRef()

    def is_het_ref(self):
        """True if the genotype call contains one reference and one alternate allele.

        :rtype bool
        """

        return self._jrep.isHetRef()

    def is_not_called(self):
        """True if the genotype call is missing.

        :rtype bool
        """

        return self._jrep.isNotCalled()

    def is_called(self):
        """True if the genotype call is non-missing.

        :rtype bool
        """

        return self._jrep.isCalled()

    def num_alt_alleles(self):
        """Returns the count of non-reference alleles.

        This function returns None if the genotype call is missing.

        :rtype int or None
        """

        return strip_option(self._jrep.nNonRefAlleles())

    def one_hot_alleles(self, num_alleles):
        """Returns a list containing the one-hot encoded representation of the called alleles.

        This one-hot representation is the positional sum of the one-hot
        encoding for each called allele.  For a biallelic variant, the
        one-hot encoding for a reference allele is [1, 0] and the one-hot
        encoding for an alternate allele is [0, 1].  Thus, with the
        following variables:

        >>> num_alleles = 2
        >>> hom_ref = Genotype(0)
        >>> het = Genotype(1)
        >>> hom_var = Genotype(2)

        All the below statements are true:

        >>> hom_ref.one_hot_alleles(num_alleles) == [2, 0]
        >>> het.one_hot_alleles(num_alleles) == [1, 1]
        >>> hom_var.one_hot_alleles(num_alleles) == [0, 2]

        This function returns None if the genotype call is missing.

        :param int num_alleles: number of possible alternate alleles
        :rtype: list of int or None
        """

        result = strip_option(self._jrep.oneHotAlleles(num_alleles))
        if result:
            result = [x for x in scala_package_object(Env.hail_package().utils).iterableToArrayList(result)]
        return result

    def one_hot_genotype(self, num_genotypes):
        """Returns a list containing the one-hot encoded representation of the genotype call.

        A one-hot encoding is a vector with one '1' and many '0' values, like
        [0, 0, 1, 0] or [1, 0, 0, 0].  This function is useful for transforming
        the genotype call (gt) into a one-hot encoded array.  With the following
        variables:

        >>> num_genotypes = 3
        >>> hom_ref = Genotype(0)
        >>> het = Genotype(1)
        >>> hom_var = Genotype(2)

        All the below statements are true:

        >>> hom_ref.one_hot_genotype(num_genotypes) == [1, 0, 0]
        >>> het.one_hot_genotype(num_genotypes) == [0, 1, 0]
        >>> hom_var.one_hot_genotype(num_genotypes) == [0, 0, 1]

        This function returns None if the genotype call is missing.

        :param int num_genotypes: number of possible genotypes
        :rtype: list of int or None
        """

        result = strip_option(self._jrep.oneHotGenotype(num_genotypes))
        if result:
            result = [x for x in scala_package_object(Env.hail_package().utils).iterableToArrayList(result)]
        return result

    def p_ab(self, theta=0.5):
        """Returns the p-value associated with finding the given allele depth ratio.

        This function uses a one-tailed binomial test.

        This function returns None if the allelic depth (ad) is missing.

        :param float theta: null reference probability for binomial model
        :rtype float or None
        """

        return strip_option(self._jrep.pAB(theta))

    def fraction_reads_ref(self):
        """Returns the fraction of reads that are reference reads.

        Equivalent to:

        >>> g.ad()[0] / sum(g.ad())

        :rtype: float or None
        """

        return strip_option(self._jrep.fractionReadsRef())
