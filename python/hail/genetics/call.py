from hail.history import *
from hail.typecheck import *
from hail.utils.java import *


class Call(HistoryMixin):
    """
    An object that represents an individual's call at a genomic locus.

    :param call: Genotype hard call
    :type call: int
    """

    _call_jobject = None

    @staticmethod
    def call_jobject():
        if not Call._call_jobject:
            Call._call_jobject = scala_object(Env.hail().variant, 'Call')
        return Call._call_jobject

    @handle_py4j
    @record_init
    @typecheck_method(gt=integral)
    def __init__(self, gt):
        """Initialize a Call object."""

        assert gt >= 0
        self._gt = gt

    def __str__(self):
        return Call.call_jobject().toString(self._gt)

    def __repr__(self):
        return 'Call(gt=%s)' % self._gt

    def __eq__(self, other):
        return isinstance(other, Call) and self.gt == other.gt

    def __hash__(self):
        # hash('Call') = 0x16f6c8bfbd18ab94
        return hash(self.gt) ^ 0x16f6c8bfbd18ab94

    @property
    def gt(self):
        """Returns the hard call.

        :rtype: int or None
        """

        return self._gt

    def is_hom_ref(self):
        """True if the call is 0/0

        :rtype: bool
        """

        return Call.call_jobject().isHomRef(self._gt)

    def is_het(self):
        """True if the call contains two different alleles.

        :rtype: bool
        """

        return Call.call_jobject().isHet(self._gt)

    def is_hom_var(self):
        """True if the call contains two identical alternate alleles.

        :rtype: bool
        """

        return Call.call_jobject().isHomVar(self._gt)

    def is_non_ref(self):
        """True if the call contains any non-reference alleles.

        :rtype: bool
        """

        return Call.call_jobject().isNonRef(self._gt)

    def is_het_non_ref(self):
        """True if the call contains two different alternate alleles.

        :rtype: bool
        """

        return Call.call_jobject().isHetNonRef(self._gt)

    def is_het_ref(self):
        """True if the call contains one reference and one alternate allele.

        :rtype: bool
        """

        return Call.call_jobject().isHetRef(self._gt)

    def num_alt_alleles(self):
        """Returns the count of non-reference alleles.

        This function returns None if the genotype call is missing.

        :rtype: int or None
        """

        return Call.call_jobject().nNonRefAlleles(self._gt)

    @handle_py4j
    @typecheck_method(num_alleles=integral)
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
        return jiterable_to_list(Call.call_jobject().oneHotAlleles(self._gt, num_alleles))
