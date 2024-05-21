import re
from collections import Counter

from hail.typecheck import nullable, sequenceof, typecheck_method
from hail.utils.java import Env, FatalError, warning


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

    @typecheck_method(s=str, fam_id=nullable(str), pat_id=nullable(str), mat_id=nullable(str), is_female=nullable(bool))
    def __init__(self, s, fam_id=None, pat_id=None, mat_id=None, is_female=None):
        self._fam_id = fam_id
        self._s = s
        self._pat_id = pat_id
        self._mat_id = mat_id
        self._is_female = is_female

    def __repr__(self):
        return 'Trio(s=%s, fam_id=%s, pat_id=%s, mat_id=%s, is_female=%s)' % (
            repr(self.s),
            repr(self.fam_id),
            repr(self.pat_id),
            repr(self.mat_id),
            repr(self.is_female),
        )

    def __str__(self):
        return 'Trio(s=%s, fam_id=%s, pat_id=%s, mat_id=%s, is_female=%s)' % (
            str(self.s),
            str(self.fam_id),
            str(self.pat_id),
            str(self.mat_id),
            str(self.is_female),
        )

    def __eq__(self, other):
        return (
            isinstance(other, Trio)
            and self._s == other._s
            and self._mat_id == other._mat_id
            and self._pat_id == other._pat_id
            and self._fam_id == other._fam_id
            and self._is_female == other._is_female
        )

    def __hash__(self):
        return hash((self._s, self._pat_id, self._mat_id, self._fam_id, self._is_female))

    @property
    def s(self):
        """ID of proband in trio, never missing.

        :rtype: str
        """

        return self._s

    @property
    def pat_id(self):
        """ID of father in trio, may be missing.

        :rtype: str or None
        """

        return self._pat_id

    @property
    def mat_id(self):
        """ID of mother in trio, may be missing.

        :rtype: str or None
        """

        return self._mat_id

    @property
    def fam_id(self):
        """Family ID.

        :rtype: str or None
        """

        return self._fam_id

    @property
    def is_male(self):
        """Returns ``True`` if the proband is a reported male,
        ``False`` if reported female, and ``None`` if no sex is defined.

        :rtype: bool or None
        """

        if self._is_female is None:
            return None

        return self._is_female is False

    @property
    def is_female(self):
        """Returns ``True`` if the proband is a reported female,
        ``False`` if reported male, and ``None`` if no sex is defined.

        :rtype: bool or None
        """

        if self._is_female is None:
            return None

        return self._is_female is True

    def is_complete(self):
        """Returns True if the trio has a defined mother and father.

        The considered fields are :meth:`mat_id` and :meth:`pat_id`.
        Recall that ``s`` may never be missing. The :meth:`fam_id`
        and :meth:`is_female` fields may be missing in a complete trio.

        :rtype: bool
        """

        return self._pat_id is not None and self._mat_id is not None

    def _restrict_to(self, ids):
        if self._s not in ids:
            return None

        return Trio(
            self._s,
            self._fam_id,
            self._pat_id if self._pat_id in ids else None,
            self._mat_id if self._mat_id in ids else None,
            self._is_female,
        )

    def _sex_as_numeric_string(self):
        if self._is_female is None:
            return "0"
        return "2" if self.is_female else "1"

    def _to_fam_file_line(self):
        def sample_id_or_else_zero(sample_id):
            if sample_id is None:
                return "0"
            return sample_id

        line_list = [
            sample_id_or_else_zero(self._fam_id),
            self._s,
            sample_id_or_else_zero(self._pat_id),
            sample_id_or_else_zero(self._mat_id),
            self._sex_as_numeric_string(),
            "0",
        ]
        return "\t".join(line_list)


class Pedigree(object):
    """Class containing a list of trios, with extra functionality.

    :param trios: list of trio objects to include in pedigree
    :type trios: list of :class:`.Trio`
    """

    @typecheck_method(trios=sequenceof(Trio))
    def __init__(self, trios):
        self._trios = tuple(trios)

    def __eq__(self, other):
        return isinstance(other, Pedigree) and self._trios == other._trios

    def __hash__(self):
        return hash(self._trios)

    def __iter__(self):
        return self._trios.__iter__()

    @classmethod
    @typecheck_method(fam_path=str, delimiter=str)
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

        trios = []
        missing_sex_count = 0
        missing_sex_values = set()
        with Env.fs().open(fam_path) as file:
            for line in file:
                split_line = re.split(delimiter, line.strip())
                num_fields = len(split_line)
                if num_fields != 6:
                    raise FatalError(
                        "Require 6 fields per line in .fam, but this line has {}: {}".format(num_fields, line)
                    )
                (fam, kid, dad, mom, sex, _) = tuple(split_line)
                # 1 is male, 2 is female, 0 is unknown.
                is_female = sex == "2" if sex == "1" or sex == "2" else None

                if is_female is None:
                    missing_sex_count += 1
                    missing_sex_values.add(kid)

                trio = Trio(
                    kid,
                    fam if fam != "0" else None,
                    dad if dad != "0" else None,
                    mom if mom != "0" else None,
                    is_female,
                )
                trios.append(trio)

        only_ids = [trio.s for trio in trios]
        duplicate_ids = [id for id, count in Counter(only_ids).items() if count > 1]
        if duplicate_ids:
            raise FatalError("Invalid pedigree: found duplicate proband IDs\n{}".format(duplicate_ids))

        if missing_sex_count > 0:
            warning(
                "Found {} samples with missing sex information (not 1 or 2).\n Missing samples: [{}]".format(
                    missing_sex_count, missing_sex_values
                )
            )

        return Pedigree(trios)

    @property
    def trios(self):
        """List of trio objects in this pedigree.

        :rtype: list of :class:`.Trio`
        """
        return self._trios

    def complete_trios(self):
        """List of trio objects that have a defined father and mother.

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

        Parameters
        ----------
        samples: :obj:`list` [:obj:`str`]
            Sample IDs to keep.

        Returns
        -------
        :class:`.Pedigree`
        """
        sample_set = set(samples)

        filtered_trios = []
        for trio in self._trios:
            restricted_trio = trio._restrict_to(sample_set)
            if restricted_trio is not None:
                filtered_trios.append(restricted_trio)

        return Pedigree(filtered_trios)

    @typecheck_method(path=str)
    def write(self, path):
        """Write a .fam file to the given path.

        **Examples**

        >>> ped = hl.Pedigree.read('data/test.fam')
        >>> ped.write('output/out.fam')

        **Notes**

        This method writes a `PLINK .fam file <https://www.cog-genomics.org/plink2/formats#fam>`_.

        .. caution::

            Phenotype information is not preserved in the Pedigree data
            structure in Hail.  Reading and writing a PLINK .fam file will
            result in loss of this information.  Use :func:`~.import_fam` to
            manipulate this information.

        :param path: output path
        :type path: str
        """

        lines = [t._to_fam_file_line() for t in self._trios]

        with Env.fs().open(path, mode="w") as file:
            for line in lines:
                file.write(line + "\n")
