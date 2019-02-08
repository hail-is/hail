from hail.typecheck import *
from hail.utils.java import *


class Trio(object):
    """Class containing information about nuclear family relatedness and sex.

    :param str s: Sample ID of proband.

    :param fam_id: Family ID.
    :type fam_id: str or None

    :param pat_id: Sample ID of father.
    :type pat_id: str or None

    :param mat_id: Sample ID of mother.
    :type mat_id: str or None

    :param is_female: Sex of proband.
    :type is_female: bool or None
    """

    @typecheck_method(s=str,
                      fam_id=nullable(str),
                      pat_id=nullable(str),
                      mat_id=nullable(str),
                      is_female=nullable(bool))
    def __init__(self, s, fam_id=None, pat_id=None, mat_id=None, is_female=None):
        jobject = Env.hail().variant.Sex
        if is_female is not None:
            jsex = jsome(jobject.Female()) if is_female else jsome(jobject.Male())
        else:
            jsex = jnone()

        self._jrep = Env.hail().methods.BaseTrio(s, joption(fam_id), joption(pat_id), joption(mat_id), jsex)
        self._fam_id = fam_id
        self._s = s
        self._pat_id = pat_id
        self._mat_id = mat_id
        self._is_female = is_female

    @classmethod
    def _from_java(cls, jrep):
        trio = Trio.__new__(cls)
        trio._jrep = jrep
        super(Trio, trio).__init__()
        return trio

    def __repr__(self):
        return 'Trio(s=%s, fam_id=%s, pat_id=%s, mat_id=%s, is_female=%s)' % (
            repr(self.s), repr(self.fam_id), repr(self.pat_id),
            repr(self.mat_id), repr(self.is_female))

    def __str__(self):
        return 'Trio(s=%s, fam_id=%s, pat_id=%s, mat_id=%s, is_female=%s)' % (
            str(self.s), str(self.fam_id), str(self.pat_id),
            str(self.mat_id), str(self.is_female))

    def __eq__(self, other):
        return isinstance(other, Trio) and self._jrep == other._jrep

    def __hash__(self):
        return self._jrep.hashCode()

    @property
    def s(self):
        """ID of proband in trio, never missing.

        :rtype: str
        """
        if not hasattr(self, '_s'):
            self._s = self._jrep.kid()
        return self._s

    @property
    def pat_id(self):
        """ID of father in trio, may be missing.

        :rtype: str or None
        """

        if not hasattr(self, '_pat_id'):
            self._pat_id = from_option(self._jrep.dad())
        return self._pat_id

    @property
    def mat_id(self):
        """ID of mother in trio, may be missing.

        :rtype: str or None
        """

        if not hasattr(self, '_mat_id'):
            self._mat_id = from_option(self._jrep.mom())
        return self._mat_id

    @property
    def fam_id(self):
        """Family ID.

        :rtype: str or None
        """

        if not hasattr(self, '_fam_id'):
            self._fam_id = from_option(self._jrep.fam())
        return self._fam_id

    @property
    def is_male(self):
        """Returns ``True`` if the proband is a reported male,
        ``False`` if reported female, and ``None`` if no sex is defined.

        :rtype: bool or None
        """
        if not hasattr(self, '_is_female'):
            j_female = self._jrep.isFemale()
            j_male = self._jrep.isFemale()
            if not j_female and not j_male:
                self._is_female = None
            else:
                self._is_female = j_female
        return self._is_female is False

    @property
    def is_female(self):
        """Returns ``True`` if the proband is a reported female,
        ``False`` if reported male, and ``None`` if no sex is defined.

        :rtype: bool or None
        """

        if not hasattr(self, '_is_female'):
            j_female = self._jrep.isFemale()
            j_male = self._jrep.isFemale()
            if not j_female and not j_male:
                self._is_female = None
            else:
                self._is_female = j_female
        return self._is_female is True

    def is_complete(self):
        """Returns True if the trio has a defined mother, father, and sex.

        The considered fields are :meth:`mat_id`, :meth:`pat_id`, and
        :meth:`is_female`. Recall that ``s`` may never be missing. The
        :meth:`fam_id` field may be missing in a complete trio.

        :rtype: bool
        """

        if not hasattr(self, '_complete'):
            self._complete = self._jrep.isComplete()
        return self._complete


class Pedigree(object):
    """Class containing a list of trios, with extra functionality.

    :param trios: list of trio objects to include in pedigree
    :type trios: list of :class:`.Trio`
    """

    def __init__(self, trios):
        self._jrep = Env.hail().methods.Pedigree(jindexed_seq([t._jrep for t in trios]))
        self._trios = trios

    @classmethod
    def _from_java(cls, jrep):
        ped = Pedigree.__new__(cls)
        ped._jrep = jrep
        ped._trios = None
        super(Pedigree, ped).__init__()
        return ped

    def __eq__(self, other):
        return isinstance(other, Pedigree) and self._jrep == other._jrep

    def __hash__(self):
        return self._jrep.hashCode()

    @classmethod
    @typecheck_method(fam_path=str,
                      delimiter=str)
    def read(cls, fam_path, delimiter='\\s+') -> 'Pedigree':
        """Read a PLINK .fam file and return a pedigree object.

        **Examples**

        >>> ped = hl.Pedigree.read('data/test.fam')

        Notes
        -------

        See `PLINK .fam file <https://www.cog-genomics.org/plink2/formats#fam>`_ for
        the required format.

        :param str fam_path: path to .fam file.

        :param str delimiter: Field delimiter.

        :rtype: :class:`.Pedigree`
        """

        jrep = Env.hail().methods.Pedigree.read(fam_path, Env.hc()._jhc.hadoopConf(), delimiter)
        return Pedigree._from_java(jrep)

    @property
    def trios(self):
        """List of trio objects in this pedigree.

        :rtype: list of :class:`.Trio`
        """

        if not self._trios:
            self._trios = [Trio._from_java(t) for t in jiterable_to_list(self._jrep.trios())]
        return self._trios

    def complete_trios(self):
        """List of trio objects that have a defined father, mother, and sex.

        :rtype: list of :class:`.Trio`
        """
        return list(filter(lambda t: t.is_complete(), self.trios))

    @typecheck_method(samples=sequenceof(nullable(str)))
    def filter_to(self, samples):
        """Filter the pedigree to a given list of sample IDs.

        **Notes**

        For any trio, the following steps will be applied:

         - If the proband is not in the list of samples provided, the trio is removed.
         - If the father is not in the list of samples provided, `pat_id` is set to ``None``.
         - If the mother is not in the list of samples provided, `mat_id` is set to ``None``.

        :param samples: list of sample IDs to keep
        :type samples: list of str

        :rtype: :class:`.Pedigree`
        """

        return Pedigree._from_java(self._jrep.filterTo(jset(samples)))

    @typecheck_method(path=str)
    def write(self, path):
        """Write a .fam file to the given path.

        **Examples**

        >>> ped = hl.Pedigree.read('data/test.fam')
        >>> ped.write('out.fam')

        **Notes**

        This method writes a `PLINK .fam file <https://www.cog-genomics.org/plink2/formats#fam>`_.

        .. caution::

            Phenotype information is not preserved in the Pedigree data structure in Hail.
            Reading and writing a PLINK .fam file will result in loss of this information.
            Use the key table method :meth:`~hail.KeyTable.import_fam` to manipulate this
            information.

        :param path: output path
        :type path: str
        """

        self._jrep.write(path, Env.hc()._jhc.hadoopConf())
