from typing import *
import hail as hl
import hail.expr.aggregators as agg
from hail.genetics.pedigree import Pedigree
from hail.matrixtable import MatrixTable
from hail.expr import expr_call, expr_float64
from hail.table import Table
from hail.typecheck import *
from hail.utils.java import Env
from .misc import require_biallelic, require_col_key_str


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
    - **id** (:py:data:`.tstr`) - Column key for the proband.
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
    mt = dataset
    require_col_key_str(mt, "trio_matrix")

    k = mt.col_key.dtype.fields[0]
    samples = mt[k].collect()

    pedigree = pedigree.filter_to(samples)
    trios = pedigree.complete_trios() if complete_trios else pedigree.trios
    n_trios = len(trios)

    sample_idx = {}
    for i, s in enumerate(samples):
        sample_idx[s] = i

    trios = [hl.Struct(
        id=sample_idx[t.s],
        pat_id=None if t.pat_id is None else sample_idx[t.pat_id],
        mat_id=None if t.mat_id is None else sample_idx[t.mat_id],
        is_female=t.is_female,
        fam_id=t.fam_id) for t in trios]
    trios_type = hl.dtype('array<struct{id:int,pat_id:int,mat_id:int,is_female:bool,fam_id:str}>')

    trios_sym = Env.get_uid()
    entries_sym = Env.get_uid()
    cols_sym = Env.get_uid()

    mt = mt.annotate_globals(**{trios_sym: hl.literal(trios, trios_type)})
    mt = mt._localize_entries(entries_sym, cols_sym)
    mt = mt.annotate_globals(**{
        cols_sym: hl.map(lambda i:
                         hl.bind(lambda t: hl.struct(id=mt[cols_sym][t.id][k],
                                                     proband=mt[cols_sym][t.id],
                                                     father=mt[cols_sym][t.pat_id],
                                                     mother=mt[cols_sym][t.mat_id],
                                                     is_female=t.is_female,
                                                     fam_id=t.fam_id),
                                 mt[trios_sym][i]),
                         hl.range(0, n_trios))})
    mt = mt.annotate(**{
        entries_sym: hl.map(lambda i:
                            hl.bind(lambda t: hl.struct(proband_entry=mt[entries_sym][t.id],
                                                        father_entry=mt[entries_sym][t.pat_id],
                                                        mother_entry=mt[entries_sym][t.mat_id]),
                                    mt[trios_sym][i]),
                            hl.range(0, n_trios))})
    mt = mt.drop(trios_sym)

    return mt._unlocalize_entries(entries_sym, cols_sym, ['id'])

@typecheck(call=expr_call,
           pedigree=Pedigree)
def mendel_errors(call, pedigree) -> Tuple[Table, Table, Table, Table]:
    r"""Find Mendel errors; count per variant, individual and nuclear family.

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

    table1 = entries.select('fam_id', 'mendel_code')

    t2 = tm.annotate_cols(
        errors=hl.agg.count(),
        snp_errors=hl.agg.count_where(hl.is_snp(tm.alleles[0], tm.alleles[1])))
    table2 = t2.key_cols_by().cols()
    table2 = table2.select(pat_id=table2.father[ck_name],
                           mat_id=table2.mother[ck_name],
                           fam_id=table2.fam_id,
                           errors=table2.errors,
                           snp_errors=table2.snp_errors)
    table2 = table2.group_by('pat_id', 'mat_id').aggregate(
        fam_id=hl.agg.take(table2.fam_id, 1)[0],
        children=hl.int32(hl.agg.count()),
        errors=hl.agg.sum(table2.errors),
        snp_errors=hl.agg.sum(table2.snp_errors))
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
                                  hl.agg.filter(hl.is_snp(tm.alleles[0], tm.alleles[1]),
                                                hl.agg.array_sum(implicated[tm.mendel_code])),
                                  [0, 0, 0])).key_cols_by().cols()

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
    table3 = (table3.group_by(ck_name, 'fam_id')
              .aggregate(errors=hl.agg.sum(table3.errors),
                         snp_errors=hl.agg.sum(table3.snp_errors))
              .key_by(ck_name))

    table4 = tm.select_rows(errors=hl.agg.count_where(hl.is_defined(tm.mendel_code))).rows()

    return table1, table2, table3, table4

@typecheck(dataset=MatrixTable,
           pedigree=Pedigree)
def transmission_disequilibrium_test(dataset, pedigree) -> Table:
    r"""Performs the transmission disequilibrium test on trios.

    .. include:: ../_templates/req_tstring.rst

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------
    Compute TDT association statistics and show the first two results:

    >>> pedigree = hl.Pedigree.read('data/tdt_trios.fam')
    >>> tdt_table = hl.transmission_disequilibrium_test(tdt_dataset, pedigree)
    >>> tdt_table.show(2)  # doctest: +NOTEST
    +---------------+------------+-------+-------+----------+----------+
    | locus         | alleles    |     t |     u |   chi_sq |  p_value |
    +---------------+------------+-------+-------+----------+----------+
    | locus<GRCh37> | array<str> | int64 | int64 |  float64 |  float64 |
    +---------------+------------+-------+-------+----------+----------+
    | 1:246714629   | ["C","A"]  |     0 |     4 | 4.00e+00 | 4.55e-02 |
    | 2:167262169   | ["T","C"]  |    NA |    NA |       NA |       NA |
    +---------------+------------+-------+-------+----------+----------+

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
     - `chi_sq` (:py:data:`.tfloat64`) -- TDT statistic.
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
    father_is_het = tri.father_entry.GT.is_het()
    parent_is_valid_het = ((father_is_het & tri.auto_or_x_par) |
                           (tri.mother_entry.GT.is_het() & ~father_is_het))

    copy_state = hl.cond(tri.auto_or_x_par | tri.is_female, 2, 1)

    config = (tri.proband_entry.GT.n_alt_alleles(),
              tri.father_entry.GT.n_alt_alleles(),
              tri.mother_entry.GT.n_alt_alleles(),
              copy_state)

    tri = tri.annotate_rows(counts = agg.filter(parent_is_valid_het, agg.array_sum(count_map.get(config))))

    tab = tri.rows().select('counts')
    tab = tab.transmute(t = tab.counts[0], u = tab.counts[1])
    tab = tab.annotate(chi_sq = ((tab.t - tab.u) ** 2) / (tab.t + tab.u))
    tab = tab.annotate(p_value = hl.pchisqtail(tab.chi_sq, 1.0))

    return tab.cache()

@typecheck(mt=MatrixTable,
           pedigree=Pedigree,
           pop_frequency_prior=expr_float64,
           min_gq=int,
           min_p=numeric,
           max_parent_ab=numeric,
           min_child_ab=numeric,
           min_dp_ratio=numeric)
def de_novo(mt: MatrixTable,
            pedigree: Pedigree,
            pop_frequency_prior,
            *,
            min_gq: int = 20,
            min_p: float = 0.05,
            max_parent_ab: float = 0.05,
            min_child_ab: float = 0.20,
            min_dp_ratio: float = 0.10) -> Table:
    r"""Call putative *de novo* events from trio data.

    .. include:: ../_templates/req_tstring.rst

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    Call de novo events:

    >>> pedigree = hl.Pedigree.read('data/trios.fam')
    >>> priors = hl.import_table('data/gnomadFreq.tsv', impute=True)
    >>> priors = priors.transmute(**hl.parse_variant(priors.Variant)).key_by('locus', 'alleles')
    >>> de_novo_results = hl.de_novo(dataset, pedigree, pop_frequency_prior=priors[dataset.row_key].AF)

    Notes
    -----
    This method assumes the GATK high-throughput sequencing fields exist:
    `GT`, `AD`, `DP`, `GQ`, `PL`.

    This method replicates the functionality of `Kaitlin Samocha's de novo
    caller <https://github.com/ksamocha/de_novo_scripts>`__. The version
    corresponding to git commit ``bde3e40`` is implemented in Hail with her
    permission and assistance.

    This method produces a :class:`.Table` with the following fields:

     - `locus` (``locus``) -- Variant locus.
     - `alleles` (``array<str>``) -- Variant alleles.
     - `id` (``str``) -- Proband sample ID.
     - `prior` (``float64``) -- Site frequency prior. It is the maximum of:
       the computed dataset alternate allele frequency, the
       `pop_frequency_prior` parameter, and the global prior
       ``1 / 3e7``.
     - `proband` (``struct``) -- Proband column fields from `mt`.
     - `father` (``struct``) -- Father column fields from `mt`.
     - `mother` (``struct``) -- Mother column fields from `mt`.
     - `proband_entry` (``struct``) -- Proband entry fields from `mt`.
     - `father_entry` (``struct``) -- Father entry fields from `mt`.
     - `proband_entry` (``struct``) -- Mother entry fields from `mt`.
     - `is_female` (``bool``) -- ``True`` if proband is female.
     - `p_de_novo` (``float64``) -- Unfiltered posterior probability
       that the event is *de novo* rather than a missed heterozygous
       event in a parent.
     - `confidence` (``str``) Validation confidence. One of: ``'HIGH'``,
       ``'MEDIUM'``, ``'LOW'``.

    The key of the table is ``['locus', 'alleles', 'id']``.

    The model looks for de novo events in which both parents are homozygous
    reference and the proband is a heterozygous. The model makes the simplifying
    assumption that when this configuration ``x = (AA, AA, AB)`` of calls
    occurs, exactly one of the following is true:

     - ``d``: a de novo mutation occurred in the proband and all calls are
       accurate.
     - ``m``: at least one parental allele is actually heterozygous and
       the proband call is accurate.

    We can then estimate the posterior probability of a de novo mutation as:

    .. math::

        \mathrm{P_{\text{de novo}}} = \frac{\mathrm{P}(d\,|\,x)}{\mathrm{P}(d\,|\,x) + \mathrm{P}(m\,|\,x)}

    Applying Bayes rule to the numerator and denominator yields

    .. math::

        \frac{\mathrm{P}(x\,|\,d)\,\mathrm{P}(d)}{\mathrm{P}(x\,|\,d)\,\mathrm{P}(d) +
        \mathrm{P}(x\,|\,m)\,\mathrm{P}(m)}

    The prior on de novo mutation is estimated from the rate in the literature:

    .. math::

        \mathrm{P}(d) = \frac{1 \text{mutation}}{30,000,000\, \text{bases}}

    The prior used for at least one alternate allele between the parents
    depends on the alternate allele frequency:

    .. math::

        \mathrm{P}(m) = 1 - (1 - AF)^4

    The likelihoods :math:`\mathrm{P}(x\,|\,d)` and :math:`\mathrm{P}(x\,|\,m)`
    are computed from the PL (genotype likelihood) fields using these
    factorizations:

    .. math::

        \mathrm{P}(x = (AA, AA, AB) \,|\,d) = \Big(
        &\mathrm{P}(x_{\mathrm{father}} = AA \,|\, \mathrm{father} = AA) \\
        \cdot &\mathrm{P}(x_{\mathrm{mother}} = AA \,|\, \mathrm{mother} =
        AA) \\ \cdot &\mathrm{P}(x_{\mathrm{proband}} = AB \,|\,
        \mathrm{proband} = AB) \Big)

    .. math::

        \mathrm{P}(x = (AA, AA, AB) \,|\,m) = \Big( &
        \mathrm{P}(x_{\mathrm{father}} = AA \,|\, \mathrm{father} = AB)
        \cdot \mathrm{P}(x_{\mathrm{mother}} = AA \,|\, \mathrm{mother} =
        AA) \\ + \, &\mathrm{P}(x_{\mathrm{father}} = AA \,|\,
        \mathrm{father} = AA) \cdot \mathrm{P}(x_{\mathrm{mother}} = AA
        \,|\, \mathrm{mother} = AB) \Big) \\ \cdot \,
        &\mathrm{P}(x_{\mathrm{proband}} = AB \,|\, \mathrm{proband} = AB)

    (Technically, the second factorization assumes there is exactly (rather
    than at least) one alternate allele among the parents, which may be
    justified on the grounds that it is typically the most likely case by far.)

    While this posterior probability is a good metric for grouping putative de
    novo mutations by validation likelihood, there exist error modes in
    high-throughput sequencing data that are not appropriately accounted for by
    the phred-scaled genotype likelihoods. To this end, a number of hard filters
    are applied in order to assign validation likelihood.

    These filters are different for SNPs and insertions/deletions. In the below
    rules, the following variables are used:

     - ``DR`` refers to the ratio of the read depth in the proband to the
       combined read depth in the parents.
     - ``AB`` refers to the read allele balance of the proband (number of
       alternate reads divided by total reads).
     - ``AC`` refers to the count of alternate alleles across all individuals
       in the dataset at the site.
     - ``p`` refers to :math:`\mathrm{P_{\text{de novo}}}`.
     - ``min_p`` refers to the ``min_p`` function parameter.

    HIGH-quality SNV:

    .. code-block:: text

        p > 0.99 && AB > 0.3 && DR > 0.2
            or
        p > 0.99 && AB > 0.3 && AC == 1

    MEDIUM-quality SNV:

    .. code-block:: text

        p > 0.5 && AB > 0.3
            or
        p > 0.5 && AB > 0.2 && AC == 1

    LOW-quality SNV:

    .. code-block:: text

        p > min_p && AB > 0.2

    HIGH-quality indel:

    .. code-block:: text

        p > 0.99 && AB > 0.3 && DR > 0.2
            or
        p > 0.99 && AB > 0.3 && AC == 1

    MEDIUM-quality indel:

    .. code-block:: text

        p > 0.5 && AB > 0.3
            or
        p > 0.5 && AB > 0.2 and AC == 1

    LOW-quality indel:

    .. code-block:: text

        p > min_p && AB > 0.2

    Additionally, de novo candidates are not considered if the proband GQ is
    smaller than the ``min_gq`` parameter, if the proband allele balance is
    lower than the ``min_child_ab`` parameter, if the depth ratio between the
    proband and parents is smaller than the ``min_depth_ratio`` parameter, or if
    the allele balance in a parent is above the ``max_parent_ab`` parameter.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        High-throughput sequencing dataset.
    pedigree : :class:`.Pedigree`
        Sample pedigree.
    pop_frequency_prior : :class:`.Float64Expression`
        Expression for population alternate allele frequency prior.
    min_gq
        Minimum proband GQ to be considered for *de novo* calling.
    min_p
        Minimum posterior probability to be considered for *de novo* calling.
    max_parent_ab
        Maximum parent allele balance.
    min_child_ab
        Minimum proband allele balance/
    min_dp_ratio
        Minimum ratio between proband read depth and parental read depth.

    Returns
    -------
    :class:`.Table`
    """
    DE_NOVO_PRIOR = 1 / 30000000
    MIN_POP_PRIOR = 100 / 30000000

    required_entry_fields = {'GT', 'AD', 'DP', 'GQ', 'PL'}
    missing_fields = required_entry_fields - set(mt.entry)
    if missing_fields:
        raise ValueError(f"'de_novo': expected 'MatrixTable' to have at least {required_entry_fields}, "
                         f"missing {missing_fields}")

    mt = mt.annotate_rows(__prior=pop_frequency_prior,
                          __alt_alleles=hl.agg.sum(mt.GT.n_alt_alleles()),
                          __total_alleles=2 * hl.agg.sum(hl.is_defined(mt.GT)))
    # subtract 1 from __alt_alleles to correct for the observed genotype
    mt = mt.annotate_rows(__site_freq=hl.max((mt.__alt_alleles - 1) / mt.__total_alleles, mt.__prior, MIN_POP_PRIOR))
    mt = require_biallelic(mt, 'de_novo')

    # FIXME check that __site_freq is between 0 and 1 when possible in expr
    tm = trio_matrix(mt, pedigree, complete_trios=True)

    autosomal = tm.locus.in_autosome_or_par() | (tm.locus.in_x_nonpar() & tm.is_female)
    hemi_x = tm.locus.in_x_nonpar() & ~tm.is_female
    hemi_y = tm.locus.in_y_nonpar() & ~tm.is_female
    hemi_mt = tm.locus.in_mito() & tm.is_female

    is_snp = hl.is_snp(tm.alleles[0], tm.alleles[1])
    n_alt_alleles = tm.__alt_alleles
    prior = tm.__site_freq
    het_hom_hom = tm.proband_entry.GT.is_het() & tm.father_entry.GT.is_hom_ref() & tm.mother_entry.GT.is_hom_ref()
    kid_ad_fail = tm.proband_entry.AD[1] / hl.sum(tm.proband_entry.AD) < min_child_ab

    failure = hl.null(hl.tstruct(p_de_novo=hl.tfloat64, confidence=hl.tstr))

    kid = tm.proband_entry
    dad = tm.father_entry
    mom = tm.mother_entry

    kid_linear_pl = 10 ** (-kid.PL / 10)
    kid_pp = hl.bind(lambda x: x / hl.sum(x), kid_linear_pl)

    dad_linear_pl = 10 ** (-dad.PL / 10)
    dad_pp = hl.bind(lambda x: x / hl.sum(x), dad_linear_pl)

    mom_linear_pl = 10 ** (-mom.PL / 10)
    mom_pp = hl.bind(lambda x: x / hl.sum(x), mom_linear_pl)

    kid_ad_ratio = kid.AD[1] / hl.sum(kid.AD)
    dp_ratio = kid.DP / (dad.DP + mom.DP)

    def call_auto(kid_pp, dad_pp, mom_pp, kid_ad_ratio):
        p_data_given_dn = dad_pp[0] * mom_pp[0] * kid_pp[1] * DE_NOVO_PRIOR
        p_het_in_parent = 1 - (1 - prior) ** 4
        p_data_given_missed_het = (dad_pp[1] * mom_pp[0] + dad_pp[0] * mom_pp[1]) * kid_pp[1] * p_het_in_parent
        p_de_novo = p_data_given_dn / (p_data_given_dn + p_data_given_missed_het)

        def solve(p_de_novo):
            return (
                hl.case()
                    .when(kid.GQ < min_gq, failure)
                    .when((kid.DP / (dad.DP + mom.DP) < min_dp_ratio) |
                          ~(kid_ad_ratio >= min_child_ab), failure)
                    .when((hl.sum(mom.AD) == 0) | (hl.sum(dad.AD) == 0), failure)
                    .when((mom.AD[1] / hl.sum(mom.AD) > max_parent_ab) |
                          (dad.AD[1] / hl.sum(dad.AD) > max_parent_ab), failure)
                    .when(p_de_novo < min_p, failure)
                    .when(~is_snp, hl.case()
                          .when((p_de_novo > 0.99) & (kid_ad_ratio > 0.3) & (n_alt_alleles == 1),
                                hl.struct(p_de_novo=p_de_novo, confidence='HIGH'))
                          .when((p_de_novo > 0.5) & (kid_ad_ratio > 0.3) & (n_alt_alleles <= 5),
                                hl.struct(p_de_novo=p_de_novo, confidence='MEDIUM'))
                          .when((p_de_novo > 0.05) & (kid_ad_ratio > 0.2),
                                hl.struct(p_de_novo=p_de_novo, confidence='LOW'))
                          .or_missing())
                    .default(hl.case()
                             .when(((p_de_novo > 0.99) & (kid_ad_ratio > 0.3) & (dp_ratio > 0.2)) |
                                   ((p_de_novo > 0.99) & (kid_ad_ratio > 0.3) & (n_alt_alleles == 1)) |
                                   ((p_de_novo > 0.5) & (kid_ad_ratio > 0.3) & (n_alt_alleles < 10) & (kid.DP > 10)),
                                   hl.struct(p_de_novo=p_de_novo, confidence='HIGH'))
                             .when((p_de_novo > 0.5) & ((kid_ad_ratio > 0.3) | (n_alt_alleles == 1)),
                                   hl.struct(p_de_novo=p_de_novo, confidence='MEDIUM'))
                             .when((p_de_novo > 0.05) & (kid_ad_ratio > 0.2),
                                   hl.struct(p_de_novo=p_de_novo, confidence='LOW'))
                             .or_missing()
                             )
            )

        return hl.bind(solve, p_de_novo)

    def call_hemi(kid_pp, parent, parent_pp, kid_ad_ratio):
        p_data_given_dn = parent_pp[0] * kid_pp[1] * DE_NOVO_PRIOR
        p_het_in_parent = 1 - (1 - prior) ** 4
        p_data_given_missed_het = (parent_pp[1] + parent_pp[2]) * kid_pp[2] * p_het_in_parent
        p_de_novo = p_data_given_dn / (p_data_given_dn + p_data_given_missed_het)

        def solve(p_de_novo):
            return (
                hl.case()
                    .when(kid.GQ < min_gq, failure)
                    .when((kid.DP / (parent.DP) < min_dp_ratio) |
                          (kid_ad_ratio < min_child_ab), failure)
                    .when((hl.sum(parent.AD) == 0), failure)
                    .when(parent.AD[1] / hl.sum(parent.AD) > max_parent_ab, failure)
                    .when(p_de_novo < min_p, failure)
                    .when(~is_snp, hl.case()
                          .when((p_de_novo > 0.99) & (kid_ad_ratio > 0.3) & (n_alt_alleles == 1),
                                hl.struct(p_de_novo=p_de_novo, confidence='HIGH'))
                          .when((p_de_novo > 0.5) & (kid_ad_ratio > 0.3) & (n_alt_alleles <= 5),
                                hl.struct(p_de_novo=p_de_novo, confidence='MEDIUM'))
                          .when((p_de_novo > 0.05) & (kid_ad_ratio > 0.3),
                                hl.struct(p_de_novo=p_de_novo, confidence='LOW'))
                          .or_missing())
                    .default(hl.case()
                             .when(((p_de_novo > 0.99) & (kid_ad_ratio > 0.3) & (dp_ratio > 0.2)) |
                                   ((p_de_novo > 0.99) & (kid_ad_ratio > 0.3) & (n_alt_alleles == 1)) |
                                   ((p_de_novo > 0.5) & (kid_ad_ratio > 0.3) & (n_alt_alleles < 10) & (kid.DP > 10)),
                                   hl.struct(p_de_novo=p_de_novo, confidence='HIGH'))
                             .when((p_de_novo > 0.5) & ((kid_ad_ratio > 0.3) | (n_alt_alleles == 1)),
                                   hl.struct(p_de_novo=p_de_novo, confidence='MEDIUM'))
                             .when((p_de_novo > 0.05) & (kid_ad_ratio > 0.2),
                                   hl.struct(p_de_novo=p_de_novo, confidence='LOW'))
                             .or_missing()
                             )
            )

        return hl.bind(solve, p_de_novo)

    de_novo_call = (
        hl.case()
            .when(~het_hom_hom | kid_ad_fail, failure)
            .when(autosomal, hl.bind(call_auto, kid_pp, dad_pp, mom_pp, kid_ad_ratio))
            .when(hemi_x | hemi_mt, hl.bind(call_hemi, kid_pp, mom, mom_pp, kid_ad_ratio))
            .when(hemi_y, hl.bind(call_hemi, kid_pp, dad, dad_pp, kid_ad_ratio))
            .or_missing()
    )

    tm = tm.annotate_entries(__call=de_novo_call)
    tm = tm.filter_entries(hl.is_defined(tm.__call))
    entries = tm.entries()
    return (entries.select('__site_freq',
                           'proband',
                           'father',
                           'mother',
                           'proband_entry',
                           'father_entry',
                           'mother_entry',
                           'is_female',
                           **entries.__call)
            .rename({'__site_freq': 'prior'}))
