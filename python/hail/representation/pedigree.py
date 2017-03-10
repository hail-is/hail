from hail.java import *


class Trio(object):
    """Class containing information about sample sex, case status, and nuclear family relatedness.

    :param str proband: sample ID of proband

    :param father: sample ID of proband
    :type father: str or None

    :param mother: sample ID of proband
    :type mother: str or None

    :param sex: sex of proband: 1 for male, 2 for female
    :type sex: int or None

    :param case_status: case status of proband: 1 for control, 2 for case
    :type case_status: int or None
    """

    _sex_jobject = None
    _pheno_jobject = None

    @staticmethod
    def _get_sex_jobject():
        if not Trio._sex_jobject:
            Trio._sex_jobject = scala_object(env.hail.variant, 'Sex')
        return Trio._sex_jobject

    @staticmethod
    def _get_pheno_jobject():
        if not Trio._pheno_jobject:
            Trio._pheno_jobject = scala_object(env.hail.variant, 'Phenotype')
        return Trio._pheno_jobject

    @handle_py4j
    def __init__(self, proband, fam=None, father=None, mother=None, sex=None, case_status=None):
        if not isinstance(proband, str) and not isinstance(proband, unicode):
            raise TypeError("param 'proband' must be of type str or unicode, but  found '%s'" % type(proband))

        if fam and not isinstance(fam, str) and not isinstance(fam, unicode):
            raise TypeError("param 'fam' must be of type str or unicode, but  found '%s'" % type(proband))

        if father and not isinstance(father, str) and not isinstance(father, unicode):
            raise TypeError("param 'father' must be of type str or unicode, but  found '%s'" % type(proband))

        if mother and not isinstance(mother, str) and not isinstance(mother, unicode):
            raise TypeError("param 'mother' must be of type str or unicode, but  found '%s'" % type(proband))

        if sex and sex != 1 and sex != 2:
            raise ValueError("sex must be 1, 2, or None, but found '%s'" % sex)

        if case_status and case_status != 1 and case_status != 2:
            raise ValueError("case_status must be 1, 2, or None, but found '%s'" % sex)

        if sex:
            if not Trio._sex_jobject:
                Trio._sex_jobject = scala_object(env.hail.variant, 'Sex')
            jsex = jsome(Trio._sex_jobject.Male()) if sex == 1 else jsome(Trio._sex_jobject.Female())
        else:
            jsex = jnone()

        if case_status:
            jpheno = jsome(Trio._get_pheno_jobject().Control()) if case_status == 1 else jsome(
                Trio._get_pheno_jobject().Case())
        else:
            jpheno = jnone()

        self._jrep = env.hail.methods.BaseTrio(proband, joption(fam), joption(father), joption(mother), jsex, jpheno)
        self._fam = fam
        self._proband = proband
        self._father = father
        self._mother = mother
        self._case = case_status
        self._sex = sex
        self._calc_fam = True
        self._calc_father = True
        self._calc_mother = True
        self._calc_sex = True
        self._calc_case = True
        self._calc_complete = False

    @classmethod
    def _from_java(cls, jrep):
        trio = Trio.__new__(cls)
        trio._jrep = jrep
        trio._proband = None
        trio._calc_fam = False
        trio._calc_father = False
        trio._calc_mother = False
        trio._calc_sex = False
        trio._calc_case = False
        trio._calc_complete = False
        return trio

    def __repr__(self):
        return 'Trio(proband=%s, fam=%s, father=%s, mother=%s, sex=%s, case_status=%s)' % (
            repr(self.proband), repr(self.fam), repr(self.father),
            repr(self.mother), repr(self.sex), repr(self.case_status)
        )

    def __str__(self):
        return 'Trio(proband=%s, fam=%s, father=%s, mother=%s, sex=%s, case_status=%s)' % (
            str(self.proband), str(self.fam), str(self.father),
            str(self.mother), str(self.sex), str(self.case_status)
        )

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
        if not self._proband:
            self._proband = self._jrep.kid()
        return self._proband

    @property
    def father(self):
        """ID of father in trio, may be missing.

        :rtype: str or None
        """

        if not self._calc_father:
            self._father = from_option(self._jrep.dad())
            self._calc_father = True
        return self._father

    @property
    def mother(self):
        """ID of mother in trio, may be missing.

        :rtype: str or None
        """

        if not self._calc_mother:
            self._mother = from_option(self._jrep.mom())
            self._calc_mother = True
        return self._mother

    @property
    def fam(self):
        """Family ID.

        :rtype: str or None
        """

        if not self._calc_fam:
            self._fam = from_option(self._jrep.fam())
            self._calc_fam = True
        return self._fam

    @property
    def sex(self):
        """Sex of proband: 1 for male, 2 for female. May be missing.

        :rtype: int or None
        """

        if not self._calc_sex:
            jsex = self._jrep.sex()
            if jsex.isDefined():
                self._sex = 1 if jsex.get() == Trio._get_sex_jobject().Male() else 2
            else:
                self._sex = None
            self._calc_sex = True
        return self._sex

    @property
    def case_status(self):
        """Case status of proband: 1 for control, 2 for case. May be missing.

        :rtype: int or None
        """

        if not self._calc_case:
            jpheno = self._jrep.pheno()
            if jpheno.isDefined():
                self._case = 1 if jpheno.get() == Trio._get_pheno_jobject().Control() else 2
            else:
                self._case = None
            self._calc_case = True

        return self._case

    def is_male(self):
        """Returns True if the proband is a reported male, False if reported female or missing.

        :rtype: bool
        """

        return self.sex is 1

    def is_female(self):
        """Returns True if the proband is a reported female, False if reported male or missing.

        :rtype: bool
        """

        return self.sex is 2

    def is_control(self):
        """Returns True if the proband is a reported control, False if reported case or missing.

        :rtype: bool
        """

        return self.case_status == 2

    def is_case(self):
        """Returns True if the proband is a reported case, False if reported control or missing.

        :rtype: bool
        """

        return self.case_status == 1

    def is_complete(self):
        """Returns True if the trio has a defined mother, father, case_status, and sex.

        The considered fields are ``mother``, ``father``, ``case_status``, and ``sex``.
        Recall that ``proband`` may not be missing. The ``fam`` field may be missing in
        a complete trio.

        :rtype: bool
        """

        if not self._calc_complete:
            self._complete = self._jrep.isComplete()
            self._calc_complete = True
        return self._complete


class Pedigree(object):
    """Class containing a list of trios, with extra functionality.

    :param trios: list of trio objects to include in pedigree
    :type trios: list of :class:`.Trio`
    """

    @handle_py4j
    def __init__(self, trios):
        if not isinstance(trios, list):
            raise TypeError("parameter 'trios' must be of type list, but found '%s'" % type(trios))
        for t in trios:
            if not isinstance(t, Trio):
                raise TypeError("all elements of list 'trios' must be of type Trio, but found '%s'" % type(t))

        self._jrep = env.hail.methods.Pedigree(jindexed_seq([t._jrep for t in trios]))
        self._trios = trios
        self._complete_trios = None

    @classmethod
    def _from_java(cls, jrep):
        ped = Pedigree.__new__(cls)
        ped._jrep = jrep
        ped._trios = None
        ped._complete_trios = None
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
    def read_fam(fam_path, delimiter='\\s+'):
        """Read a .fam file and return a pedigree object.

        This method reads a `PLINK .fam file <https://www.cog-genomics.org/plink2/formats#fam>`_.

        Hail expects a file in the same spec as PLINK outlines.

        :param str fam_path: path to .fam file.
        :param str delimiter: field delimiter

        :rtype: :class:`.Pedigree`
        """

        jrep = scala_object(env.hail.methods, 'Pedigree').fromFam(fam_path, env.hc._jhc.hadoopConf(), delimiter)
        return Pedigree._from_java(jrep)

    @property
    def trios(self):
        """List of trio objects in this pedigree.

        :rtype: list of :class:`.Trio`
        """

        if not self._trios:
            self._trios = [Trio._from_java(t) for t in jiterable_to_list(self._jrep.trios())]
        return self._trios

    @property
    def complete_trios(self):
        """List of trio objects that have a defined father, mother, case status, and sex.

        :rtype: list of :class:`.Trio`
        """

        if not self._complete_trios:
            self._complete_trios = filter(lambda t: t.is_complete(), self._trios)
        return self._complete_trios

    @handle_py4j
    def filter_to(self, samples):
        """Filter the pedigree to a given list of sample IDs.

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
    def write_fam(self, path):
        """Write a .fam file to the given path.

        This method writes a `PLINK .fam file <https://www.cog-genomics.org/plink2/formats#fam>`_.

        :param str path: output path
        """

        self._jrep.write(path, env.hc._jhc.hadoopConf())
