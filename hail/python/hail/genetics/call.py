from hail.typecheck import *
from hail.utils.java import *


class Call(object):
    """
    An object that represents an individual's call at a genomic locus.

    Parameters
    ----------
    alleles : :obj:`list` of :obj:`int`
        List of alleles that compose the call.
    phased : :obj:`bool`
        If ``True``, the alleles are phased and the order is specified by
        `alleles`.
    """

    _cached_jobject = None

    @staticmethod
    def _call_jobject():
        if not Call._cached_jobject:
            Call._cached_jobject = scala_object(Env.hail().variant, 'Call')
        return Call._cached_jobject

    @typecheck_method(alleles=sequenceof(int),
                      phased=bool)
    def __init__(self, alleles, phased=False):
        if len(alleles) > 2:
            raise NotImplementedError("Calls with greater than 2 alleles are not supported.")
        self._phased = phased
        self._alleles = alleles
        self._ploidy = len(alleles)
        self._call = scala_object(Env.hail().variant, 'CallN').apply(alleles, phased)

    @classmethod
    def _from_java(cls, jc):
        c = Call.__new__(cls)
        c._call = jc
        c._alleles = None
        c._phased = None
        c._ploidy = None
        super(Call, c).__init__()
        return c

    def __str__(self):
        return Call._call_jobject().toString(self._call)

    def __repr__(self):
        return 'Call(alleles=%s, phased=%s)' % (self.alleles, self.phased)

    def __eq__(self, other):
        return isinstance(other, Call) and self._call == other._call

    def __hash__(self):
        # hash('Call') = 0x16f6c8bfbd18ab94
        return hash(self._call) ^ 0x16f6c8bfbd18ab94

    def __getitem__(self, item):
        """Get the i*th* allele.

        Returns
        -------
        :obj:`int`
        """
        return self.alleles[item]

    @property
    def alleles(self):
        """Get the alleles of this call.

        Returns
        -------
        :obj:`list` of :obj:`int`
        """

        if self._alleles is None:
            self._alleles = jarray_to_list(Call._call_jobject().alleles(self._call))
        return self._alleles

    @property
    def ploidy(self):
        """The number of alleles for this call.

        Returns
        -------
        :obj:`int`
        """

        if not self._ploidy:
            self._ploidy = Call._call_jobject().ploidy(self._call)
        return self._ploidy

    @property
    def phased(self):
        """True if the call is phased.

        Returns
        -------
        :obj:`bool`
        """

        if not self._phased:
            self._phased = Call._call_jobject().isPhased(self._call)
        return self._phased

    def is_haploid(self):
        """True if the ploidy == 1.

        :rtype: bool
        """

        return Call._call_jobject().isHaploid(self._call)

    def is_diploid(self):
        """True if the ploidy == 2.

        :rtype: bool
        """

        return Call._call_jobject().isDiploid(self._call)

    def is_hom_ref(self):
        """True if the call has no alternate alleles.

        :rtype: bool
        """

        return Call._call_jobject().isHomRef(self._call)

    def is_het(self):
        """True if the call contains two different alleles.

        :rtype: bool
        """

        return Call._call_jobject().isHet(self._call)

    def is_hom_var(self):
        """True if the call contains two identical alternate alleles.

        :rtype: bool
        """

        return Call._call_jobject().isHomVar(self._call)

    def is_non_ref(self):
        """True if the call contains any non-reference alleles.

        :rtype: bool
        """

        return Call._call_jobject().isNonRef(self._call)

    def is_het_non_ref(self):
        """True if the call contains two different alternate alleles.

        :rtype: bool
        """

        return Call._call_jobject().isHetNonRef(self._call)

    def is_het_ref(self):
        """True if the call contains one reference and one alternate allele.

        :rtype: bool
        """

        return Call._call_jobject().isHetRef(self._call)

    def n_alt_alleles(self):
        """Returns the count of non-reference alleles.

        :rtype: int
        """

        return Call._call_jobject().nNonRefAlleles(self._call)

    @typecheck_method(n_alleles=int)
    def one_hot_alleles(self, n_alleles):
        """Returns a list containing the one-hot encoded representation of the
        called alleles.

        Examples
        --------

        >>> n_alleles = 2
        >>> hom_ref = hl.Call([0, 0])
        >>> het = hl.Call([0, 1])
        >>> hom_var = hl.Call([1, 1])

        >>> het.one_hot_alleles(n_alleles)
        [1, 1]

        >>> hom_var.one_hot_alleles(n_alleles)
        [0, 2]

        Notes
        -----
        This one-hot representation is the positional sum of the one-hot
        encoding for each called allele.  For a biallelic variant, the
        one-hot encoding for a reference allele is [1, 0] and the one-hot
        encoding for an alternate allele is [0, 1].

        Parameters
        ----------
        n_alleles : :obj:`int`
            Number of total alleles, including the reference.

        Returns
        -------
        :obj:`list` of :obj:`int`
        """
        return jiterable_to_list(Call._call_jobject().oneHotAlleles(self._call, n_alleles))

    def unphased_diploid_gt_index(self):
        """Return the genotype index for unphased, diploid calls.

        Returns
        -------
        :obj:`int`
        """

        if self.ploidy != 2 or self.phased:
            raise FatalError(
                "'unphased_diploid_gt_index' is only valid for unphased, diploid calls. Found {}.".format(repr(self)))
        return Call._call_jobject().unphasedDiploidGtIndex(self._call)
