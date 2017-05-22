from hail.java import *


class Trio(object):
    """Class containing information about nuclear family relatedness and sex.

    :param str proband: Sample ID of proband.

    :param fam: Family ID.
    :type fam: str or None

    :param father: Sample ID of father.
    :type father: str or None

    :param mother: Sample ID of mother.
    :type mother: str or None

    :param is_female: Sex of proband.
    :type is_female: bool or None
    """

    @handle_py4j
    def __init__(self, proband, fam=None, father=None, mother=None, is_female=None):
        jobject = Env.hail().variant.Sex
        if is_female is not None:
            jsex = jsome(jobject.Female()) if is_female else jsome(jobject.Male())
        else:
            jsex = jnone()

        self._jrep = Env.hail().methods.BaseTrio(proband, joption(fam), joption(father), joption(mother), jsex)
        self._fam = fam
        self._proband = proband
        self._father = father
        self._mother = mother
        self._is_female = is_female

    @classmethod
    def _from_java(cls, jrep):
        trio = Trio.__new__(cls)
        trio._jrep = jrep
        return trio

    def __repr__(self):
        return 'Trio(proband=%s, fam=%s, father=%s, mother=%s, is_female=%s)' % (
            repr(self.proband), repr(self.fam), repr(self.father),
            repr(self.mother), repr(self.is_female))

    def __str__(self):
        return 'Trio(proband=%s, fam=%s, father=%s, mother=%s, is_female=%s)' % (
            str(self.proband), str(self.fam), str(self.father),
            str(self.mother), str(self.is_female))

    def __eq__(self, other):
        if not isinstance(other, Trio):
            return False
        else:
            return self._jrep == other._jrep

    def __hash__(self):
        return self._jrep.hashCode()

    @property
    def proband(self):
        """ID of proband in trio, never missing.

        :rtype: str
        """
        if not hasattr(self, '_proband'):
            self._proband = self._jrep.kid()
        return self._proband

    @property
    def father(self):
        """ID of father in trio, may be missing.

        :rtype: str or None
        """

        if not hasattr(self, '_father'):
            self._father = from_option(self._jrep.dad())
        return self._father

    @property
    def mother(self):
        """ID of mother in trio, may be missing.

        :rtype: str or None
        """

        if not hasattr(self, '_mother'):
            self._mother = from_option(self._jrep.mom())
        return self._mother

    @property
    def fam(self):
        """Family ID.

        :rtype: str or None
        """

        if not hasattr(self, '_fam'):
            self._fam = from_option(self._jrep.fam())
        return self._fam

    @property
    def is_male(self):
        """Returns True if the proband is a reported male, False if reported female, and None if no sex is defined.

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
        """Returns True if the proband is a reported female, False if reported male, and None if no sex is defined.

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

        The considered fields are ``mother``, ``father``, and ``sex``.
        Recall that ``proband`` may never be missing. The ``fam`` field 
        may be missing in a complete trio.

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

    @handle_py4j
    def __init__(self, trios):

        self._jrep = Env.hail().methods.Pedigree(jindexed_seq([t._jrep for t in trios]))
        self._trios = trios

    @classmethod
    def _from_java(cls, jrep):
        ped = Pedigree.__new__(cls)
        ped._jrep = jrep
        ped._trios = None
        return ped

    def __eq__(self, other):
        if not isinstance(other, Pedigree):
            return False
        else:
            return self._jrep == other._jrep

    def __hash__(self):
        return self._jrep.hashCode()

    @staticmethod
    @handle_py4j
    def read(fam_path, delimiter='\\s+'):
        """Read a .fam file and return a pedigree object.
        
        **Examples**
        
        >>> ped = Pedigree.read('data/test.fam')
        
        **Notes**

        This method reads a `PLINK .fam file <https://www.cog-genomics.org/plink2/formats#fam>`_.

        Hail expects a file in the same spec as PLINK outlines.

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
        return filter(lambda t: t.is_complete(), self._trios)

    @handle_py4j
    def filter_to(self, samples):
        """Filter the pedigree to a given list of sample IDs.
        
        **Notes**
        
        For any trio, the following steps will be applied:

         - If the proband is not in the list of samples provided, the trio is removed.
         - If the father is not in the list of samples provided, the father is set to ``None``.
         - If the mother is not in the list of samples provided, the mother is set to ``None``.

        :param samples: list of sample IDs to keep
        :type samples: list of str

        :rtype: :class:`.Pedigree`
        """

        return Pedigree._from_java(self._jrep.filterTo(jset(samples)))

    @handle_py4j
    def write(self, path):
        """Write a .fam file to the given path.

        **Examples**
        
        >>> ped = Pedigree.read('data/test.fam')
        >>> ped.write('out.fam')
        
        **Notes**

        This method writes a `PLINK .fam file <https://www.cog-genomics.org/plink2/formats#fam>`_.

        .. caution::
        
            Phenotype information is not preserved in the Pedigree data structure in Hail.
            Reading and writing a PLINK .fam file will result in loss of this information.
            Use the key table method :py:meth:`~hail.KeyTable.import_fam` to manipulate this 
            information.

        :param str path: output path
        """

        self._jrep.write(path, Env.hc()._jhc.hadoopConf())
