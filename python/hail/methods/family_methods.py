from typing import *
import hail as hl
import hail.expr.aggregators as agg
from hail.genetics.pedigree import Pedigree
from hail.matrixtable import MatrixTable
from hail.expr import expr_call
from hail.table import Table
from hail.typecheck import *
from .misc import require_biallelic


@typecheck(dataset=MatrixTable,
           pedigree=Pedigree,
           complete_trios=bool)
def trio_matrix(dataset, pedigree, complete_trios=False) -> MatrixTable:
    """Builds and returns a matrix where columns correspond to trios and entries contain genotypes for the trio.

    .. include:: ../_templates/req_tstring.rst

    Examples
    --------

    Create a trio matrix:

    >>> pedigree = hl.Pedigree.read('data/case_control_study.fam')
    >>> trio_dataset = hl.trio_matrix(dataset, pedigree, complete_trios=True)

    Notes
    -----

    This method builds a new matrix table with one column per trio. If
    `complete_trios` is ``True``, then only trios that satisfy
    :meth:`.Trio.is_complete` are included. In this new dataset, the column
    identifiers are the sample IDs of the trio probands. The column fields and
    entries of the matrix are changed in the following ways:

    The new column fields consist of three structs (`proband`, `father`,
    `mother`), a Boolean field, and a string field:

    - **proband** (:class:`.tstruct`) - Column fields on the proband.
    - **father** (:class:`.tstruct`) - Column fields on the father.
    - **mother** (:class:`.tstruct`) - Column fields on the mother.
    - **is_female** (:py:data:`.tbool`) - Proband is female.
      ``True`` for female, ``False`` for male, missing if unknown.
    - **fam_id** (:py:data:`.tstr`) - Family ID.

    The new entry fields are:

    - **proband_entry** (:class:`.tstruct`) - Proband entry fields.
    - **father_entry** (:class:`.tstruct`) - Father entry fields.
    - **mother_entry** (:class:`.tstruct`) - Mother entry fields.

    Parameters
    ----------
    pedigree : :class:`.Pedigree`

    Returns
    -------
    :class:`.MatrixTable`
    """
    return MatrixTable(dataset._jvds.trioMatrix(pedigree._jrep, complete_trios))

@typecheck(call=expr_call,
           pedigree=Pedigree)
def mendel_errors(call, pedigree) -> Tuple[Table, Table, Table, Table]:
    """Find Mendel errors; count per variant, individual and nuclear family.

    .. include:: ../_templates/req_tstring.rst

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    Find all violations of Mendelian inheritance in each (dad, mom, kid) trio in
    a pedigree and return four tables (all errors, errors by family, errors by
    individual, errors by variant):

    >>> ped = hl.Pedigree.read('data/trios.fam')
    >>> all_errors, per_fam, per_sample, per_variant = hl.mendel_errors(dataset['GT'], ped)

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

        - `locus` (:class:`.tlocus`) -- Variant locus, key field.
        - `alleles` (:class:`.tarray` of :py:data:`.tstr`) -- Variant alleles, key field.
        - (column key of `dataset`) (:py:data:`.tstr`) -- Proband ID, key field.
        - `fam_id` (:py:data:`.tstr`) -- Family ID.
        - `mendel_code` (:py:data:`.tint32`) -- Mendel error code, see below.

    **Second table:** errors per nuclear family. This table contains one row
    per nuclear family, keyed by the parents.

        - `pat_id` (:py:data:`.tstr`) -- Paternal ID. (key field)
        - `mat_id` (:py:data:`.tstr`) -- Maternal ID. (key field)
        - `fam_id` (:py:data:`.tstr`) -- Family ID.
        - `children` (:py:data:`.tint32`) -- Number of children in this nuclear family.
        - `errors` (:py:data:`.tint64`) -- Number of Mendel errors in this nuclear family.
        - `snp_errors` (:py:data:`.tint64`) -- Number of Mendel errors at SNPs in this
          nuclear family.

    **Third table:** errors per individual. This table contains one row per
    individual. Each error is counted toward the proband, father, and mother
    according to the `Implicated` in the table below.

        - (column key of `dataset`) (:py:data:`.tstr`) -- Sample ID (key field).
        - `fam_id` (:py:data:`.tstr`) -- Family ID.
        - `errors` (:py:data:`.tint64`) -- Number of Mendel errors involving this
          individual.
        - `snp_errors` (:py:data:`.tint64`) -- Number of Mendel errors involving this
          individual at SNPs.

    **Fourth table:** errors per variant.

        - `locus` (:class:`.tlocus`) -- Variant locus, key field.
        - `alleles` (:class:`.tarray` of :py:data:`.tstr`) -- Variant alleles, key field.
        - `errors` (:py:data:`.tint64`) -- Number of Mendel errors in this variant.

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

    See Also
    --------
    :func:`.mendel_error_code`

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
    pedigree : :class:`.Pedigree`

    Returns
    -------
    (:class:`.Table`, :class:`.Table`, :class:`.Table`, :class:`.Table`)
    """
    source = call._indices.source
    if not isinstance(source, MatrixTable):
        raise ValueError("'mendel_errors': expected 'call' to be an expression of 'MatrixTable', found {}".format(
            "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'))

    source = source.select_entries(__GT=call)
    dataset = require_biallelic(source, 'mendel_errors')
    tm = trio_matrix(dataset, pedigree, complete_trios=True)
    tm = tm.select_entries(mendel_code=hl.mendel_error_code(
        tm.locus,
        tm.is_female,
        tm.father_entry['__GT'],
        tm.mother_entry['__GT'],
        tm.proband_entry['__GT']
    ))
    ck_name = next(iter(source.col_key))
    tm = tm.filter_entries(hl.is_defined(tm.mendel_code))
    tm = tm.rename({'id' : ck_name})

    entries = tm.entries()

    table1 = entries.select(*tm.row_key, *tm.col_key, 'fam_id', 'mendel_code')

    fam_counts = (
        entries
            .group_by(pat_id=entries.father[ck_name], mat_id=entries.mother[ck_name])
            .partition_hint(min(entries.n_partitions(), 8))
            .aggregate(children=hl.len(hl.agg.collect_as_set(entries[ck_name])),
                       errors=hl.agg.count_where(hl.is_defined(entries.mendel_code)),
                       snp_errors=hl.agg.count_where(hl.is_snp(entries.alleles[0], entries.alleles[1]) &
                                                     hl.is_defined(entries.mendel_code)))
    )
    table2 = tm.cols()
    table2 = table2.select(pat_id=table2.father[ck_name],
                           mat_id=table2.mother[ck_name],
                           fam_id=table2.fam_id,
                           **fam_counts[table2.father[ck_name], table2.mother[ck_name]])
    table2 = table2.key_by('mat_id', 'pat_id').distinct()
    table2 = table2.annotate(errors=hl.or_else(table2.errors, hl.int64(0)),
                             snp_errors=hl.or_else(table2.snp_errors, hl.int64(0)))

    # in implicated, idx 0 is dad, idx 1 is mom, idx 2 is child
    implicated = hl.literal([
        [0, 0, 0],  # dummy
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
    ], dtype=hl.tarray(hl.tarray(hl.tint64)))

    table3 = tm.annotate_cols(all_errors=hl.or_else(hl.agg.array_sum(implicated[tm.mendel_code]), [0, 0, 0]),
                              snp_errors=hl.or_else(
                                  hl.agg.array_sum(hl.agg.filter(hl.is_snp(tm.alleles[0], tm.alleles[1]),
                                                                 implicated[tm.mendel_code])),
                                  [0, 0, 0])).cols()

    table3 = table3.select(xs=[
        hl.struct(**{ck_name: table3.father[ck_name],
                     'fam_id': table3.fam_id,
                     'errors': table3.all_errors[0],
                     'snp_errors': table3.snp_errors[0]}),
        hl.struct(**{ck_name: table3.mother[ck_name],
                     'fam_id': table3.fam_id,
                     'errors': table3.all_errors[1],
                     'snp_errors': table3.snp_errors[1]}),
        hl.struct(**{ck_name: table3.proband[ck_name],
                     'fam_id': table3.fam_id,
                     'errors': table3.all_errors[2],
                     'snp_errors': table3.snp_errors[2]}),
    ])
    table3 = table3.explode('xs')
    table3 = table3.select(**table3.xs)
    table3 = table3.group_by(ck_name, 'fam_id').aggregate(errors=hl.agg.sum(table3.errors),
                                                          snp_errors=hl.agg.sum(table3.snp_errors))

    table4 = tm.select_rows(*tm.row_key,
                            errors=hl.agg.count_where(hl.is_defined(tm.mendel_code))).rows()

    return table1, table2, table3, table4

@typecheck(dataset=MatrixTable,
           pedigree=Pedigree)
def transmission_disequilibrium_test(dataset, pedigree) -> Table:
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
    
        >>> pedigree = hl.Pedigree.read('data/tdt_trios.fam')
        >>> tdt_table = hl.transmission_disequilibrium_test(tdt_dataset, pedigree)
        >>> tdt_table.show(2)
        +---------------+------------+-------+-------+-------------+-------------+
        | locus         | alleles    |     t |     u |        chi2 |    p_values |
        +---------------+------------+-------+-------+-------------+-------------+
        | locus<GRCh37> | array<str> | int32 | int32 |     float64 |     float64 |
        +---------------+------------+-------+-------+-------------+-------------+
        | 1:246714629   | ["C","A"]  |     0 |     4 | 4.00000e+00 | 4.55003e-02 |
        | 2:167262169   | ["T","C"]  |    NA |    NA |          NA |          NA |
        +---------------+------------+-------+-------+-------------+-------------+

    Export variants with p-values below 0.001:

    >>> tdt_table = tdt_table.filter(tdt_table.p_value < 0.001)
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

    :func:`transmission_disequilibrium_test` only considers complete trios (two
    parents and a proband with defined sex) and only returns results for the
    autosome, as defined by :meth:`~hail.genetics.Locus.in_autosome`, and
    chromosome X. Transmissions and non-transmissions are counted only for the
    configurations of genotypes and copy state in the table below, in order to
    filter out Mendel errors and configurations where transmission is
    guaranteed. The copy state of a locus with respect to a trio is defined as
    follows:

    - Auto -- in autosome or in PAR of X or female child
    - HemiX -- in non-PAR of X and male child

    Here PAR is the `pseudoautosomal region
    <https://en.wikipedia.org/wiki/Pseudoautosomal_region>`__
    of X and Y defined by :class:`.ReferenceGenome`, which many variant callers
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

     - `locus` (:class:`.tlocus`) -- Locus.
     - `alleles` (:class:`.tarray` of :py:data:`.tstr`) -- Alleles.
     - `t` (:py:data:`.tint32`) -- Number of transmitted alternate alleles.
     - `u` (:py:data:`.tint32`) -- Number of untransmitted alternate alleles.
     - `chi2` (:py:data:`.tfloat64`) -- TDT statistic.
     - `p_value` (:py:data:`.tfloat64`) -- p-value.

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

    dataset = require_biallelic(dataset, 'transmission_disequilibrium_test')
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

    count_map = hl.literal({(c[0], c[1], c[2], c[3]): [c[4], c[5]] for c in config_counts})

    tri = trio_matrix(dataset, pedigree, complete_trios=True)

    # this filter removes mendel error of het father in x_nonpar. It also avoids
    #   building and looking up config in common case that neither parent is het
    parent_is_valid_het = hl.bind(tri.father_entry.GT.is_het(),
        lambda father_is_het: (father_is_het & tri.auto_or_x_par) |
                              (tri.mother_entry.GT.is_het() & ~father_is_het))

    copy_state = hl.cond(tri.auto_or_x_par | tri.is_female, 2, 1)

    config = (tri.proband_entry.GT.n_alt_alleles(),
              tri.father_entry.GT.n_alt_alleles(),
              tri.mother_entry.GT.n_alt_alleles(),
              copy_state)

    tri = tri.annotate_rows(counts = agg.array_sum(agg.filter(parent_is_valid_het, count_map.get(config))))

    tab = tri.rows().select('locus', 'alleles', 'counts')
    tab = tab.transmute(t = tab.counts[0], u = tab.counts[1])
    tab = tab.annotate(chi2 = ((tab.t - tab.u) ** 2) / (tab.t + tab.u))
    tab = tab.annotate(p_value = hl.pchisqtail(tab.chi2, 1.0))

    return tab.cache()
