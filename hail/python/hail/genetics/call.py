from collections.abc import Sequence

from hail.typecheck import typecheck_method
from hail.utils import FatalError


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

    Note
    ----
    This object refers to the Python value returned by taking or collecting
    Hail expressions, e.g. ``mt.GT.take(5`)``. This is rare; it is much
    more common to manipulate the :class:`.CallExpression` object, which is
    constructed using the following functions:

     - :func:`.call`
     - :func:`.unphased_diploid_gt_index_call`
     - :func:`.parse_call`
    """

    def __init__(self, alleles, phased=False):
        # Intentionally not using the type check annotations which are too slow.
        assert isinstance(alleles, Sequence)
        assert isinstance(phased, bool)

        if len(alleles) > 2:
            raise NotImplementedError("Calls with greater than 2 alleles are not supported.")
        self._phased = phased
        ploidy = len(alleles)
        if phased or ploidy < 2:
            self._alleles = alleles
        else:
            assert ploidy == 2
            a0 = alleles[0]
            a1 = alleles[1]
            if a1 < a0:
                a0, a1 = a1, a0
            self._alleles = [a0, a1]

    def __str__(self):
        n = self.ploidy
        if n == 0:
            if self._phased:
                return '|-'
            return '-'

        if n == 1:
            if self._phased:
                return f'|{self._alleles[0]}'
            return str(self._alleles[0])

        assert n == 2
        a0 = self._alleles[0]
        a1 = self._alleles[1]
        if self._phased:
            return f'{a0}|{a1}'
        return f'{a0}/{a1}'

    def __repr__(self):
        return 'Call(alleles=%s, phased=%s)' % (self._alleles, self._phased)

    def __eq__(self, other):
        return (isinstance(other, Call)
                and self._phased == other._phased
                and self._alleles == other._alleles)

    def __hash__(self):
        return hash(self._phased) ^ hash(tuple(self._alleles))

    def __getitem__(self, item):
        """Get the i*th* allele.

        Returns
        -------
        :obj:`int`
        """
        return self._alleles[item]

    @property
    def alleles(self):
        """Get the alleles of this call.

        Returns
        -------
        :obj:`list` of :obj:`int`
        """
        return self._alleles

    @property
    def ploidy(self):
        """The number of alleles for this call.

        Returns
        -------
        :obj:`int`
        """
        return len(self._alleles)

    @property
    def phased(self):
        """True if the call is phased.

        Returns
        -------
        :obj:`bool`
        """
        return self._phased

    def is_haploid(self):
        """True if the ploidy == 1.

        :rtype: bool
        """
        return self.ploidy == 1

    def is_diploid(self):
        """True if the ploidy == 2.

        :rtype: bool
        """
        return self.ploidy == 2

    def is_hom_ref(self):
        """True if the call has no alternate alleles.

        :rtype: bool
        """
        if self.ploidy == 0:
            return False

        return all(a == 0 for a in self._alleles)

    def is_het(self):
        """True if the call contains two different alleles.

        :rtype: bool
        """
        if self.ploidy < 2:
            return False
        return self._alleles[0] != self._alleles[1]

    def is_hom_var(self):
        """True if the call contains identical alternate alleles.

        :rtype: bool
        """
        n = self.ploidy
        if n == 0:
            return False

        a0 = self._alleles[0]
        if a0 == 0:
            return False

        if n == 1:
            return True

        assert n == 2
        return self._alleles[1] == a0

    def is_non_ref(self):
        """True if the call contains any non-reference alleles.

        :rtype: bool
        """
        return any(a > 0 for a in self._alleles)

    def is_het_non_ref(self):
        """True if the call contains two different alternate alleles.

        :rtype: bool
        """
        n = self.ploidy
        if n < 2:
            return False

        assert n == 2
        a0 = self._alleles[0]
        a1 = self._alleles[1]
        return a0 > 0 and a1 > 0 and a0 != a1

    def is_het_ref(self):
        """True if the call contains one reference and one alternate allele.

        :rtype: bool
        """
        n = self.ploidy
        if n < 2:
            return False

        assert n == 2
        a0 = self._alleles[0]
        a1 = self._alleles[1]
        return (a0 == 0 and a1 > 0) or (a0 > 0 and a1 == 0)

    def n_alt_alleles(self):
        """Returns the count of non-reference alleles.

        :rtype: int
        """
        n = 0
        for a in self._alleles:
            if a > 0:
                n += 1
        return n

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
        r = [0] * n_alleles
        for a in self._alleles:
            r[a] += 1
        return r

    def unphased_diploid_gt_index(self):
        """Return the genotype index for unphased, diploid calls.

        Returns
        -------
        :obj:`int`
        """

        if self.ploidy != 2 or self.phased:
            raise FatalError(
                "'unphased_diploid_gt_index' is only valid for unphased, diploid calls. Found {}.".format(repr(self)))
        a0 = self._alleles[0]
        a1 = self._alleles[1]
        assert a0 <= a1
        return a1 * (a1 + 1) / 2 + a0
