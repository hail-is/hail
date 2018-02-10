import hail as hl
from hail.genetics.pedigree import Pedigree
from hail.typecheck import *
import hail.expr.aggregators as agg
from hail.utils.java import handle_py4j
from hail.matrixtable import MatrixTable
from hail.table import Table
from .misc import require_biallelic


@handle_py4j
@typecheck(dataset=MatrixTable,
           pedigree=Pedigree,
           complete_trios=bool)
def trio_matrix(dataset, pedigree, complete_trios=False):
    """Builds and returns a matrix where columns correspond to trios and entries contain genotypes for the trio.

    .. include:: ../_templates/req_tstring.rst

    Examples
    --------

    Create a trio matrix:

    >>> pedigree = Pedigree.read('data/case_control_study.fam')
    >>> trio_dataset = hl.trio_matrix(dataset, pedigree, complete_trios=True)

    Notes
    -----

    This method builds a new matrix table with one column per trio. If
    `complete_trios` is ``True``, then only trios that satisfy
    :meth:`.Trio.is_complete` are included. In this new dataset, the column
    identifiers are the sample IDs of the trio probands. The column fields and
    entries of the matrix are changed in the following ways:

    The new column fields consist of three Structs (`proband`, `father`,
    `mother`), a Boolean field, and a String field:

    - proband.id** (*String*) - Proband sample ID, same as trio column key.
    - proband.fields** (*Struct*) - Column fields on the proband.
    - father.id** (*String*) - Father sample ID.
    - father.fields** (*Struct*) - Column fields on the father.
    - mother.id** (*String*) - Mother sample ID.
    - mother.fields** (*Struct*) - Column fields on the mother.
    - is_female** (*Boolean*) - Proband is female.
      True for female, false for male, missing if unknown.
    - **fam_id** (*String*) - Family ID.

    The new entry fields are:

    - **proband_entry** (*Struct*) - Proband entry fields.
    - **father_entry** (*Struct*) - Father entry fields.
    - **mother_entry** (*Struct*) - Mother entry fields.

    Parameters
    ----------
    pedigree : :class:`.Pedigree`

    Returns
    -------
    :class:`.MatrixTable`
    """
    return MatrixTable(dataset._jvds.trioMatrix(pedigree._jrep, complete_trios))

@handle_py4j
@typecheck(dataset=MatrixTable,
           pedigree=Pedigree)
def mendel_errors(dataset, pedigree):
    """Find Mendel errors; count per variant, individual and nuclear family.

    .. include:: ../_templates/req_tstring.rst

    .. include:: ../_templates/req_tvariant.rst
    
    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    Find all violations of Mendelian inheritance in each (dad, mom, kid) trio in
    a pedigree and return four tables (all errors, errors by family, errors by
    individual, errors by variant):

    >>> ped = Pedigree.read('data/trios.fam')
    >>> all_errors, per_fam, per_sample, per_variant = hl.mendel_errors(dataset, ped)

    Export all mendel errors to a text file:

    >>> all_errors.export('output/all_mendel_errors.tsv')

    Annotate columns with the number of Mendel errors:

    >>> annotated_samples = dataset.annotate_cols(mendel=per_sample[dataset.s])

    Annotate rows with the number of Mendel errors:

    >>> annotated_variants = dataset.annotate_rows(mendel=per_variant[dataset.locus, dataset.alleles])

    Notes
    -----

    The example above returns four tables, which contain Mendelian violations
    grouped in various ways. These tables are modeled after the `PLINK mendel
    formats <https://www.cog-genomics.org/plink2/formats#mendel>`_, resembling
    the ``.mendel``, ``.fmendel``, ``.imendel``, and ``.lmendel`` formats,
    respectively.

    **First table:** all Mendel errors. This table contains one row per Mendel
    error, keyed by the variant and proband id.

        - `fam_id` (:class:`.TString`) -- Family ID.
        - `locus` (:class:`.TLocus`) -- Variant locus, key field.
        - `alleles` (:class:`.TArray` of :class:`.TString`) -- Variant alleles, key field.
        - `s` (:class:`.TString`) -- Proband ID, key field.
        - `code` (:class:`.TInt32`) -- Mendel error code, see below.
        - `error` (:class:`.TString`) -- Readable representation of Mendel error.

    **Second table:** errors per nuclear family. This table contains one row
    per nuclear family, keyed by the parents.

        - `fam_id` (:class:`.TString`) -- Family ID.
        - `pat_id` (:class:`.TString`) -- Paternal ID. (key field)
        - `mat_id` (:class:`.TString`) -- Maternal ID. (key field)
        - `children` (:class:`.TInt32`) -- Number of children in this nuclear family.
        - `errors` (:class:`.TInt32`) -- Number of Mendel errors in this nuclear family.
        - `snp_errors` (:class:`.TInt32`) -- Number of Mendel errors at SNPs in this
          nuclear family.

    **Third table:** errors per individual. This table contains one row per
    individual. Each error is counted toward the proband, father, and mother
    according to the `Implicated` in the table below.

        - `s` (:class:`.TString`) -- Sample ID (key field).
        - `fam_id` (:class:`.TString`) -- Family ID.
        - `errors` (:class:`.TInt64`) -- Number of Mendel errors involving this
          individual.
        - `snp_errors` (:class:`.TInt64`) -- Number of Mendel errors involving this
          individual at SNPs.

    **Fourth table:** errors per variant.

        - `locus` (:class:`.TLocus`) -- Variant locus, key field.
        - `alleles` (:class:`.TArray` of :class:`.TString`) -- Variant alleles, key field.
        - `errors` (:class:`.TInt32`) -- Number of Mendel errors in this variant.

    This method only considers complete trios (two parents and proband with
    defined sex). The code of each Mendel error is determined by the table
    below, extending the
    `Plink classification <https://www.cog-genomics.org/plink2/basic_stats#mendel>`__.

    In the table, the copy state of a locus with respect to a trio is defined
    as follows, where PAR is the `pseudoautosomal region
    <https://en.wikipedia.org/wiki/Pseudoautosomal_region>`__ (PAR) of X and Y
    defined by the reference genome and the autosome is defined by
    :meth:`~hail.genetics.Locus.in_autosome`.

    - Auto -- in autosome or in PAR or female child
    - HemiX -- in non-PAR of X and male child
    - HemiY -- in non-PAR of Y and male child

    `Any` refers to the set \{ HomRef, Het, HomVar, NoCall \} and `~`
    denotes complement in this set.

    +------+---------+---------+--------+----------------------------+
    | Code | Dad     | Mom     | Kid    | Copy State | Implicated    |
    +======+=========+=========+========+============+===============+
    |    1 | HomVar  | HomVar  | Het    | Auto       | Dad, Mom, Kid |
    +------+---------+---------+--------+------------+---------------+
    |    2 | HomRef  | HomRef  | Het    | Auto       | Dad, Mom, Kid |
    +------+---------+---------+--------+------------+---------------+
    |    3 | HomRef  | ~HomRef | HomVar | Auto       | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    4 | ~HomRef | HomRef  | HomVar | Auto       | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    5 | HomRef  | HomRef  | HomVar | Auto       | Kid           |
    +------+---------+---------+--------+------------+---------------+
    |    6 | HomVar  | ~HomVar | HomRef | Auto       | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    7 | ~HomVar | HomVar  | HomRef | Auto       | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    8 | HomVar  | HomVar  | HomRef | Auto       | Kid           |
    +------+---------+---------+--------+------------+---------------+
    |    9 | Any     | HomVar  | HomRef | HemiX      | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |   10 | Any     | HomRef  | HomVar | HemiX      | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |   11 | HomVar  | Any     | HomRef | HemiY      | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |   12 | HomRef  | Any     | HomVar | HemiY      | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+

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

    dataset = require_biallelic(dataset, 'mendel_errors')

    kts = dataset._jvds.mendelErrors(pedigree._jrep)
    return Table(kts._1()), Table(kts._2()), \
           Table(kts._3()), Table(kts._4())

@handle_py4j
@typecheck(dataset=MatrixTable,
           pedigree=Pedigree)
def tdt(dataset, pedigree):
    """Performs the transmission disequilibrium test on trios.

    .. include:: ../_templates/req_tstring.rst

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------
    Compute TDT association statistics and show the first two results:

    .. testsetup::

        tdt_dataset = hl.import_vcf('data/tdt_tiny.vcf')

    .. doctest::
    
        >>> pedigree = Pedigree.read('data/tdt_trios.fam')
        >>> tdt_table = hl.tdt(tdt_dataset, pedigree)
        >>> tdt_table.show(2)
        +------------------+-------+-------+-------------+-------------+
        | v                |     t |     u |        chi2 |        pval |
        +------------------+-------+-------+-------------+-------------+
        | Variant(GRCh37)  | Int32 | Int32 |     Float64 |     Float64 |
        +------------------+-------+-------+-------------+-------------+
        | 1:246714629:C:A  |     0 |     4 | 4.00000e+00 | 4.55003e-02 |
        | 2:167262169:T:C  |     0 |     0 |         NaN |         NaN |
        +------------------+-------+-------+-------------+-------------+

    Export variants with p-values below 0.001:

    >>> tdt_table = tdt_table.filter(tdt_table.pval < 0.001)
    >>> tdt_table.export("output/tdt_results.tsv")

    Notes
    -----
    The
    `transmission disequilibrium test <https://en.wikipedia.org/wiki/Transmission_disequilibrium_test#The_case_of_trios:_one_affected_child_per_family>`__
    compares the number of times the alternate allele is transmitted (t) versus
    not transmitted (u) from a heterozgyous parent to an affected child. The null
    hypothesis holds that each case is equally likely. The TDT statistic is given by

    .. math::

        (t - u)^2 \over (t + u)

    and asymptotically follows a chi-squared distribution with one degree of
    freedom under the null hypothesis.

    :func:`tdt` only considers complete trios (two parents and a proband with
    defined sex) and only returns results for the autosome, as defined by
    :meth:`~hail.genetics.Locus.in_autosome`, and chromosome X. Transmissions
    and non-transmissions are counted only for the configurations of genotypes
    and copy state in the table below, in order to filter out Mendel errors and
    configurations where transmission is guaranteed. The copy state of a locus
    with respect to a trio is defined as follows:

    - Auto -- in autosome or in PAR of X or female child
    - HemiX -- in non-PAR of X and male child

    Here PAR is the `pseudoautosomal region
    <https://en.wikipedia.org/wiki/Pseudoautosomal_region>`__
    of X and Y defined by :class`.GenomeReference`, which many variant callers
    map to chromosome X.

    +--------+--------+--------+------------+---+---+
    |  Kid   | Dad    | Mom    | Copy State | t | u |
    +========+========+========+============+===+===+
    | HomRef | Het    | Het    | Auto       | 0 | 2 |
    +--------+--------+--------+------------+---+---+
    | HomRef | HomRef | Het    | Auto       | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | HomRef | Het    | HomRef | Auto       | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | Het    | Het    | Het    | Auto       | 1 | 1 |
    +--------+--------+--------+------------+---+---+
    | Het    | HomRef | Het    | Auto       | 1 | 0 |
    +--------+--------+--------+------------+---+---+
    | Het    | Het    | HomRef | Auto       | 1 | 0 |
    +--------+--------+--------+------------+---+---+
    | Het    | HomVar | Het    | Auto       | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | Het    | Het    | HomVar | Auto       | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | HomVar | Het    | Het    | Auto       | 2 | 0 |
    +--------+--------+--------+------------+---+---+
    | HomVar | Het    | HomVar | Auto       | 1 | 0 |
    +--------+--------+--------+------------+---+---+
    | HomVar | HomVar | Het    | Auto       | 1 | 0 |
    +--------+--------+--------+------------+---+---+
    | HomRef | HomRef | Het    | HemiX      | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | HomRef | HomVar | Het    | HemiX      | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | HomVar | HomRef | Het    | HemiX      | 1 | 0 |
    +--------+--------+--------+------------+---+---+
    | HomVar | HomVar | Het    | HemiX      | 1 | 0 |
    +--------+--------+--------+------------+---+---+

    :func:`tdt` produces a table with the following columns:

     - `locus` (:class:`.TLocus`) -- Locus.
     - `alleles` (:class:`.TArray` of :class:`.TString`) -- Alleles.
     - `t` (:class:`.TInt32`) -- Number of transmitted alternate alleles.
     - `u` (:class:`.TInt32`) -- Number of untransmitted alternate alleles.
     - `chi2` (:class:`.TFloat64`) -- TDT statistic.
     - `pval` (:class:`.TFloat64`) -- p-value.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    pedigree : :class:`~hail.genetics.Pedigree`
        Sample pedigree.

    Returns
    -------
    :class:`.Table`
        Table of TDT results.
    """

    dataset = require_biallelic(dataset, 'tdt')
    dataset = dataset.annotate_rows(auto_or_x_par = dataset.locus.in_autosome() | dataset.locus.in_x_par())
    dataset = dataset.filter_rows(dataset.auto_or_x_par | dataset.locus.in_x_nonpar())

    hom_ref = 0
    het = 1
    hom_var = 2

    auto = 2
    hemi_x = 1

    #                     kid,     dad,     mom,   copy, t, u
    config_counts = [(hom_ref,     het,     het,   auto, 0, 2),
                     (hom_ref, hom_ref,     het,   auto, 0, 1),
                     (hom_ref,     het, hom_ref,   auto, 0, 1),
                     (    het,     het,     het,   auto, 1, 1),
                     (    het, hom_ref,     het,   auto, 1, 0),
                     (    het,     het, hom_ref,   auto, 1, 0),
                     (    het, hom_var,     het,   auto, 0, 1),
                     (    het,     het, hom_var,   auto, 0, 1),
                     (hom_var,     het,     het,   auto, 2, 0),
                     (hom_var,     het, hom_var,   auto, 1, 0),
                     (hom_var, hom_var,     het,   auto, 1, 0),
                     (hom_ref, hom_ref,     het, hemi_x, 0, 1),
                     (hom_ref, hom_var,     het, hemi_x, 0, 1),
                     (hom_var, hom_ref,     het, hemi_x, 1, 0),
                     (hom_var, hom_var,     het, hemi_x, 1, 0)]

    count_map = hl.broadcast({hl.capture([c[0], c[1], c[2], c[3]]): [c[4], c[5]] for c in config_counts})

    tri = trio_matrix(dataset, pedigree, complete_trios=True)

    # this filter removes mendel error of het father in x_nonpar. It also avoids
    #   building and looking up config in common case that neither parent is het
    parent_is_valid_het = hl.bind(tri.father_entry.GT.is_het(),
        lambda father_is_het: (father_is_het & tri.auto_or_x_par) | 
                              (tri.mother_entry.GT.is_het() & ~father_is_het))

    copy_state = hl.cond(tri.auto_or_x_par | tri.is_female, 2, 1)

    config = [tri.proband_entry.GT.num_alt_alleles(),
              tri.father_entry.GT.num_alt_alleles(),
              tri.mother_entry.GT.num_alt_alleles(),
              copy_state]

    tri = tri.annotate_rows(counts = agg.array_sum(agg.filter(parent_is_valid_het, count_map.get(config))))

    tab = tri.rows_table().select('locus', 'alleles', 'counts')
    tab = tab.transmute(t = tab.counts[0], u = tab.counts[1])
    tab = tab.annotate(chi2 = ((tab.t - tab.u) ** 2) / (tab.t + tab.u))
    tab = tab.annotate(pval = hl.pchisqtail(tab.chi2, 1.0))

    return tab.cache()
