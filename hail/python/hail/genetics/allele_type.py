from enum import IntEnum, auto


_ALLELE_STRS = [
    "Unknown",
    "SNP",
    "MNP",
    "Insertion",
    "Deletion",
    "Complex",
    "Star",
    "Symbolic",
    "Transition",
    "Transversion",
]


class AlleleType(IntEnum):
    """An enumeration for allele type.

    Notes
    -----
    The precise values of the enumeration constants are not guarenteed
    to be stable and must not be relied upon.
    """

    UNKNOWN = 0
    """Unknown Allele Type"""
    SNP = auto()
    """Single-nucleotide Polymorphism (SNP)"""
    MNP = auto()
    """Multi-nucleotide Polymorphism (MNP)"""
    INSERTION = auto()
    """Insertion"""
    DELETION = auto()
    """Deletion"""
    COMPLEX = auto()
    """Complex Polymorphism"""
    STAR = auto()
    """Star Allele (``alt=*``)"""
    SYMBOLIC = auto()
    """Symbolic Allele

    e.g. ``alt=<INS>``
    """
    TRANSITION = auto()
    """Transition SNP

    e.g. ``ref=A alt=G``

    Note
    ----
    This is only really used internally in :func:`hail.vds.sample_qc` and
    :func:`hail.methods.sample_qc`.
    """
    TRANSVERSION = auto()
    """Transversion SNP

    e.g. ``ref=A alt=C``

    Note
    ----
    This is only really used internally in :func:`hail.vds.sample_qc` and
    :func:`hail.methods.sample_qc`.
    """

    def __str__(self):
        return _ALLELE_STRS[self.value]

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None
        return cls.__members__.get(value.upper())

    @staticmethod
    def strings():
        """Returns the names of the allele types, for use with
        :func:`~hail.expr.functions.literal`
        """
        return _ALLELE_STRS[:]
