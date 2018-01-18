from hail.genetics.pedigree import Pedigree
from hail.typecheck import *
from hail.utils.java import handle_py4j
from hail.api2 import MatrixTable, Table
from .misc import require_biallelic


@handle_py4j
@typecheck(dataset=MatrixTable,
           pedigree=Pedigree,
           complete_trios=bool)
def trio_matrix(dataset, pedigree, complete_trios=False):
    """Builds and returns a matrix where columns correspond to trios and entries contain genotypes for the trio.

    **Examples**

    Create a trio matrix:

    .. testsetup::

        dataset = vds.annotate_samples_expr('sa = drop(sa, qc)').to_hail2()
        from hail.methods import trio_matrix

    >>> pedigree = Pedigree.read('data/case_control_study.fam')
    >>> trio_dataset = trio_matrix(dataset, pedigree, complete_trios=True)

    **Notes**

    This method builds a new matrix table with one column per trio. If ``complete_trios`` is true,
    then only trios that satisfy :meth:`~hail.representation.Trio.is_complete`
    are included. In this new dataset, the column identifiers
    are the sample IDs of the trio probands. The column annotations and
    entries of the matrix are changed in the following ways:

    The new column fields are:

     - **proband.id** (*String*) - Proband sample ID, same as trio column key.
     - **proband.annotations** (*Struct*) - Annotations on the proband.
     - **father.id** (*String*) - Father sample ID.
     - **father.annotations** (*Struct*) - Annotations on the father.
     - **mother.id** (*String*) - Mother sample ID.
     - **mother.annotations** (*Struct*) - Annotations on the mother.
     - **isFemale** (*Boolean*) - Proband is female. True for female, False for male, missing if unknown.
     - **famID** (*String*) - Family identifier.

    The new entry fields are:

    - **proband_entry** (*Struct*) - Proband entry fields.
    - **father_entry** (*Struct*) - Father entry fields.
    - **mother_entry** (*Struct*) - Mother entry fields.

    :param pedigree: Collection of trios.
    :type pedigree: :class:`.hail.representation.Pedigree`

    :rtype: :class:`.VariantDataset`
    """
    return MatrixTable(dataset._jvds.trioMatrix(pedigree._jrep, complete_trios)
                       .annotateGenotypesExpr('g = {proband_entry: g.proband, father_entry: g.father, mother_entry: g.mother}'))

@handle_py4j
@typecheck(dataset=MatrixTable,
           pedigree=Pedigree)
def mendel_errors(dataset, pedigree):
    """Find Mendel errors; count per variant, individual and nuclear family.

    .. include:: ../_templates/req_tvariant.rst
    .. include:: ../_templates/req_tstring.rst
    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    Find all violations of Mendelian inheritance in each (dad, mom, kid) trio in
    a pedigree and return four tables (all errors, errors by family, errors by
    individual, errors by variant):

    >>> ped = Pedigree.read('data/trios.fam')
    >>> all, per_fam, per_sample, per_variant = methods.mendel_errors(dataset, ped)

    Export all mendel errors to a text file:

    >>> all.export('output/all_mendel_errors.tsv')

    Annotate columns with the number of Mendel errors:

    >>> annotated_samples = dataset.annotate_cols(mendel=per_sample[dataset.s])

    Annotate rows with the number of Mendel errors:

    >>> annotated_variants = dataset.annotate_rows(mendel=per_variant[dataset.v])

    Notes
    -----

    The example above returns four tables, which contain Mendelian violations
    grouped in various ways. These tables are modeled after the `PLINK mendel
    formats <https://www.cog-genomics.org/plink2/formats#mendel>`_. The four
    tables contain the following columns:

    **First table:** all Mendel errors. This table contains one row per Mendel
    error in the dataset; it is possible that a variant or sample may be found
    on more than one row. This table closely reflects the structure of the
    ".mendel" PLINK format detailed below.

    Columns:

        - **fid** (*String*) -- Family ID.
        - **s** (*String*) -- Proband ID.
        - **v** (*Variant*) -- Variant in which the error was found.
        - **code** (*Int32*) -- Mendel error code, see below.
        - **error** (*String*) -- Readable representation of Mendel error.

    **Second table:** errors per nuclear family. This table contains one row
    per nuclear family in the dataset. This table closely reflects the structure
    of the ".fmendel" PLINK format detailed below.

    Columns:

        - **fid** (*String*) -- Family ID.
        - **father** (*String*) -- Paternal ID.
        - **mother** (*String*) -- Maternal ID.
        - **nChildren** (*Int32*) -- Number of children in this nuclear family.
        - **nErrors** (*Int32*) -- Number of Mendel errors in this nuclear family.
        - **nSNP** (*Int32*) -- Number of Mendel errors at SNPs in this nuclear
          family.

    **Third table:** errors per individual. This table contains one row per
    individual in the dataset, including founders. This table closely reflects
    the structure of the ".imendel" PLINK format detailed below.

    Columns:

        - **s** (*String*) -- Sample ID (key column).
        - **fid** (*String*) -- Family ID.
        - **nErrors** (*Int64*) -- Number of Mendel errors found involving this
          individual.
        - **nSNP** (*Int64*) -- Number of Mendel errors found involving this
          individual at SNPs.
        - **error** (*String*) -- Readable representation of Mendel error.

    **Fourth table:** errors per variant. This table contains one row per
    variant in the dataset.

    Columns:

        - **v** (*Variant*) -- Variant (key column).
        - **nErrors** (*Int32*) -- Number of Mendel errors in this variant.

    **PLINK Mendel error formats:**

        - ``*.mendel`` -- all mendel errors: FID KID CHR SNP CODE ERROR
        - ``*.fmendel`` -- error count per nuclear family: FID PAT MAT CHLD N
        - ``*.imendel`` -- error count per individual: FID IID N
        - ``*.lmendel`` -- error count per variant: CHR SNP N

    In the PLINK formats, **FID**, **KID**, **PAT**, **MAT**, and **IID** refer
    to family, kid, dad, mom, and individual ID, respectively, with missing
    values set to ``0``. SNP denotes the variant identifier ``chr:pos:ref:alt``.
    N is the error count. CHLD is the number of children in a nuclear family.

    The CODE of each Mendel error is determined by the table below, extending
    the `Plink classification
    <https://www.cog-genomics.org/plink2/basic_stats#mendel>`__.

    The copy state of a locus with respect to a trio is defined as follows,
    where PAR is the `pseudoautosomal region
    <https://en.wikipedia.org/wiki/Pseudoautosomal_region>`__ (PAR) defined by
    the reference genome of the locus.

    - HemiX -- in non-PAR of X, male child
    - HemiY -- in non-PAR of Y, male child
    - Auto -- otherwise (in autosome or PAR, female child)

    :math:`Any` refers to :math:`\{ HomRef, Het, HomVar, NoCall \}` and :math:`~`
    denotes complement in this set.

    +--------+------------+------------+----------+------------------+
    |Code    | Dad        | Mom        |     Kid  |   Copy State     |
    +========+============+============+==========+==================+
    |    1   | HomVar     | HomVar     | Het      | Auto             |
    +--------+------------+------------+----------+------------------+
    |    2   | HomRef     | HomRef     | Het      | Auto             |
    +--------+------------+------------+----------+------------------+
    |    3   | HomRef     |  ~ HomRef  |  HomVar  | Auto             |
    +--------+------------+------------+----------+------------------+
    |    4   |  ~ HomRef  | HomRef     |  HomVar  | Auto             |
    +--------+------------+------------+----------+------------------+
    |    5   | HomRef     | HomRef     |  HomVar  | Auto             |
    +--------+------------+------------+----------+------------------+
    |    6   | HomVar     |  ~ HomVar  |  HomRef  | Auto             |
    +--------+------------+------------+----------+------------------+
    |    7   |  ~ HomVar  | HomVar     |  HomRef  | Auto             |
    +--------+------------+------------+----------+------------------+
    |    8   | HomVar     | HomVar     |  HomRef  | Auto             |
    +--------+------------+------------+----------+------------------+
    |    9   | Any        | HomVar     |  HomRef  | HemiX            |
    +--------+------------+------------+----------+------------------+
    |   10   | Any        | HomRef     |  HomVar  | HemiX            |
    +--------+------------+------------+----------+------------------+
    |   11   | HomVar     | Any        |  HomRef  | HemiY            |
    +--------+------------+------------+----------+------------------+
    |   12   | HomRef     | Any        |  HomVar  | HemiY            |
    +--------+------------+------------+----------+------------------+

    This method only considers children with two parents and a defined sex.

    This method assumes all contigs apart from those defined as
    :meth:`~hail.representation.GenomeReference.x_contigs` or
    :meth:`~hail.representation.GenomeReference.y_contigs` by the reference
    genome are fully autosomal. Mitochondria, decoys, etc. are not given special
    treatment.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    pedigree : :class:`.Pedigree`
        Sample pedigree.

    Returns
    -------
    (:class:`.Table`, :class:`.Table`, :class:`.Table`, :class:`.Table`)
        Four tables as detailed in notes with Mendel error statistics.
    """

    kts = require_biallelic(dataset, 'mendel_errors')._jvds.mendelErrors(pedigree._jrep)
    return Table(kts._1()), Table(kts._2()), \
           Table(kts._3()), Table(kts._4())