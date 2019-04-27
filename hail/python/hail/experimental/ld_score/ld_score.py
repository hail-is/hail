
import hail as hl
from hail.expr.expressions import *
from hail.expr.types import *
from hail.typecheck import *
from hail.table import Table
from hail.matrixtable import MatrixTable
from hail.linalg import BlockMatrix
from hail.utils import new_temp_file, wrap_to_list
from hail.methods.misc import *


def _assign_blocks(mt, n_blocks, two_step_threshold):
    """Assign variants to jackknife blocks."""

    col_keys = list(mt.col_key)
    ht = mt.localize_entries(entries_array_field_name='__entries',
                             columns_array_field_name='__cols')

    if two_step_threshold:
        ht = ht.annotate(__entries=hl.rbind(
            hl.scan.array_agg(
                lambda entry: hl.scan.count_where(entry.__in_step1),
                ht.__entries),
            lambda step1_indices: hl.map(
                lambda i: hl.rbind(
                    hl.int(hl.or_else(step1_indices[i], 0)),
                    ht.__cols[i].__m_step1,
                    ht.__entries[i],
                    lambda step1_idx, m_step1, entry: hl.rbind(
                        hl.map(
                            lambda j: hl.int(hl.floor(j * (m_step1 / n_blocks))),
                            hl.range(0, n_blocks + 1)),
                        lambda step1_separators: hl.rbind(
                            hl.set(step1_separators).contains(step1_idx),
                            hl.sum(
                                hl.map(
                                    lambda s1: step1_idx >= s1,
                                    step1_separators)) - 1,
                            lambda is_separator, step1_block: entry.annotate(
                                __step1_block=step1_block,
                                __step2_block=hl.cond(~entry.__in_step1 & is_separator,
                                                      step1_block - 1,
                                                      step1_block))))),
                hl.range(0, hl.len(ht.__entries)))))
    else:
        ht = ht.annotate(__entries=hl.rbind(
            hl.scan.array_agg(
                lambda entry: hl.scan.count_where(entry.__in_step1),
                ht.__entries),
            lambda step1_indices: hl.map(
                lambda i: hl.rbind(
                    hl.int(hl.or_else(step1_indices[i], 0)),
                    ht.__cols[i].__m_step1,
                    ht.__entries[i],
                    lambda step1_idx, m_step1, entry: hl.rbind(
                        hl.map(
                            lambda j: hl.int(hl.floor(j * (m_step1 / n_blocks))),
                            hl.range(0, n_blocks + 1)),
                        lambda step1_separators: entry.annotate(
                            __step1_block=hl.sum(
                                hl.map(
                                    lambda s1: step1_idx >= s1,
                                    step1_separators)) - 1))),
                hl.range(0, hl.len(ht.__entries)))))

    mt = ht._unlocalize_entries('__entries', '__cols', col_keys)

    tmp = new_temp_file()
    mt.write(tmp)
    mt = hl.read_matrix_table(tmp)

    return mt


def _block_betas(block_expr, include_expr, y_expr, covariates, w_expr, n_blocks):
    return hl.agg.array_agg(
        lambda i: hl.agg.filter((block_expr != i) & include_expr,
                                hl.agg.linreg(y=y_expr,
                                              x=covariates,
                                              weight=w_expr).beta),
        hl.range(0, n_blocks))


def _pseudovalues(estimate, block_estimates):
    return hl.rbind(
        hl.len(block_estimates),
        lambda n: hl.map(
            lambda block: n * estimate - (n - 1) * block,
            block_estimates))


def _variance(x_array):
    return hl.rbind(
        hl.len(x_array),
        lambda n: (hl.sum(x_array**2) - (hl.sum(x_array)**2 / n)) / (n - 1) / n)


def _require_first_key_field_locus(dataset, method):
    if isinstance(dataset, Table):
        key = dataset.key
    else:
        assert isinstance(dataset, MatrixTable)
        key = dataset.row_key
    if (len(key) == 0 or
            not isinstance(key[0].dtype, tlocus) or
            list(key)[0] != 'locus'):
        raise ValueError("Method '{}' requires first key field of type 'locus<any>'.\n"
                         "  Found:{}".format(method, ''.join(
            "\n    '{}': {}".format(k, str(dataset[k].dtype)) for k in key)))


@typecheck(ld_matrix=BlockMatrix,
           annotation_exprs=oneof(expr_numeric,
                                  sequenceof(expr_numeric)))
def ld_scores(ld_matrix,
              annotation_exprs) -> Table:
    """Compute LD scores.

    Given an LD matrix, such as the one returned by the :func:`.ld_matrix`
    method, and one or more locus-keyed ``annotation_exprs``,
    :func:`.ld_scores` computes LD scores for each locus and annotation.

    The univariate LD score of a variant :math:`j` is defined as the sum
    of the squared correlations between variant :math:`j` and all other
    variants :math:`k` in the reference panel:

    .. math::

        l_j = \\sum_{k=1}^{M}r_{jk}^2

    In practice, the formula above is approximated using only a window of
    variants around variant :math:`j`. See the :func:`.ld_matrix` method for
    more details.

    Given a categorical annotation :math:`C`, the LD score of variant
    :math:`j` with respect to that annotation is the sum of squared correlations
    between variant :math:`j` and all other variants :math:`k` that are both
    in the reference panel and in annotation category :math:`C`:

    .. math::

        l(j, C) = \\sum_{k\\in{C}}r_{jk}^2


    Example
    -------

    Compute LD scores from an LD matrix and a set of annotations.

    >>> # Create locus-keyed table with annotations
    >>> ht = hl.import_table(
    ...     paths='data/ldsc.annot',
    ...     types={'BP': hl.tint,
    ...            'univariate': hl.tfloat,
    ...            'binary': hl.tfloat,
    ...            'continuous': hl.tfloat})
    >>> ht = ht.annotate(locus=hl.locus(ht.CHR, ht.BP))
    >>> ht = ht.key_by(ht.locus)

    >>> # Read pre-computed LD matrix.
    >>> r2 = BlockMatrix.read('data/ldsc.ld_matrix.bm')

    >>> # Use LD matrix and annotations to compute LD scores
    >>> ht_scores = hl.experimental.ld_score.ld_scores(
    ...                 ld_matrix=r2,
    ...                 annotation_exprs=[
    ...                     ht.univariate,
    ...                     ht.binary,
    ...                     ht.continuous])


    Notes
    -----

    All ``annotation_exprs`` must originate from the same table or matrix
    table, and the first row key field of this source must be a field
    ``"locus"`` of type :py:data:`.tlocus`.

    The number of rows in the ``annotation_exprs`` table or matrix table
    must match the number of rows and columns in the ``ld_matrix``.


    Warning
    -------

    The method will raise an error if the number of rows in the
    ``annotation_exprs`` table or matrix table differs from the dimensions of
    the ``ld_matrix``.

    However, if the dimensions match and the rows (variants) of the table or
    matrix table are shuffled relative to the ``ld_matrix`` rows/columns,
    then the method will report inaccurate LD scores.


    Parameters
    ----------
    ld_matrix : :class:`.BlockMatrix`
        A pre-computed LD matrix, such as one returned by :func:`.ld_matrix`.
    annotation_exprs : :class:`.NumericExpression` or
                       :obj:`list` of :class:`.NumericExpression`
        A single numeric annotation expression or a list of numeric annotation
        expressions. LD scores will be calculated for each of the expressions
        provided to this argument.

    Returns
    -------
    :class:`.Table`
        Table keyed by a field ``"locus"`` of type :py:data:`.tlocus` with a row
        field ``"ld_scores"`` of type :py:data:`.tarray`, where each element is
        of type :py:data:`.tfloat`.
    """
    
    annotation_exprs = wrap_to_list(annotation_exprs)
    ds = annotation_exprs[0]._indices.source
    block_size = ld_matrix.block_size
    n_variants = ds.count()

    for i, expr in enumerate(annotation_exprs):
        analyze(f'calculate_ld_scores/annotation_exprs{i}',
                expr,
                ds._row_indices)

    if isinstance(ds, MatrixTable):
        if (list(ds.row_key)[0] != 'locus' or
                not isinstance(ds.row_key[0].dtype, tlocus)):
            raise ValueError(
                """The first row key field of an "annotation_exprs" matrix
                table must be a field "locus" of type 'locus<any>'.""")           
        ds = ds.select_rows(_annotations=hl.array([
            hl.struct(_a=hl.float(a)) for a in annotation_exprs]))
        ds = ds.rows()

    else:
        assert isinstance(ds, Table)
        if (list(ds.key)[0] != 'locus' or
                not isinstance(ds.key[0].dtype, tlocus)):
            raise ValueError(
                """The first key field of an "annotation_exprs" table must
                be a field "locus" of type 'locus<any>'.""")
        ds = ds.select(_annotations=hl.array([
            hl.struct(_a=hl.float(a)) for a in annotation_exprs]))

    ds = ds.annotate_globals(_names=hl.array([
        hl.struct(_n=f'a{i}') for i, _ in enumerate(annotation_exprs)]))

    ds = ds.repartition(n_variants / block_size)
    ds = ds._unlocalize_entries('_annotations', '_names', ['_n'])
    ds = ds.add_row_index()

    a_tmp = new_temp_file()
    BlockMatrix.write_from_entry_expr(  
        entry_expr=(ds._a),
        path=a_tmp,
        mean_impute=True,
        center=False,
        normalize=False,
        block_size=block_size)

    a = BlockMatrix.read(a_tmp)
    l2 = (ld_matrix @ a).to_table_row_major()

    scores = ds.annotate(
        ld_scores=l2[ds.row_idx].entries)
    scores = scores.key_by(scores.locus)

    scores = scores.select_globals()
    scores = scores.select(scores.ld_scores)

    scores_tmp = new_temp_file()
    scores.write(scores_tmp)

    return hl.read_table(scores_tmp)


@typecheck(z_expr=expr_numeric,
           n_samples_expr=expr_numeric,
           ld_score_exprs=oneof(expr_numeric,
                                sequenceof(expr_numeric)),
           weight_expr=expr_numeric,
           n_blocks=int,
           two_step_threshold=nullable(int),
           n_reference_panel_variants=nullable(int))
def ld_score_regression(z_expr,
                        n_samples_expr,
                        ld_score_exprs,
                        weight_expr,
                        n_blocks=200,
                        two_step_threshold=None,
                        n_reference_panel_variants=None):

    ds = z_expr._indices.source

    analyze('ld_score_regression/weight_expr',
            weight_expr,
            ds._row_indices)

    for i, expr in ld_score_exprs:
        analyze(f'ld_score_regression/ld_score_expr{i}',
                expr,
                ds._row_indices)

    if isinstance(ds, MatrixTable):
        analyze('ld_score_regression/n_samples_expr',
                n_samples_expr,
                ds._entry_indices)
        
@typecheck(z_expr=expr_numeric,
           n_samples_expr=expr_numeric,
           weight_expr=expr_numeric,
           ld_score_expr=expr_numeric,
           annotation_exprs=nullable(oneof(expr_numeric,
                                           sequenceof(expr_numeric))),
           n_blocks=int,
           two_step_threshold=nullable(int),
           n_reference_panel_variants=nullable(int),
           _return_block_estimates=bool)
def estimate_heritability(z_expr,
                          n_samples_expr,
                          weight_expr,
                          ld_score_expr,
                          annotation_exprs=None,
                          n_blocks=200,
                          two_step_threshold=30,
                          n_reference_panel_variants=None,
                          _return_block_estimates=False) -> Table:
    r"""Estimate SNP-heritability and level of confounding biases from
    GWAS summary statistics.

    Given genome-wide association study (GWAS) summary statistics,
    :func:`.estimate_heritability` estimates the heritability of a
    trait or set of traits and the level of confounding biases present in
    the underlying association studies using the LD score regression
    method. This approach leverages the model:

    .. math::

        \mathrm{E}[\chi_j^2] = 1 + Na + \frac{Nh_g^2}{M}l_j

    *  :math:`\mathrm{E}[\chi_j^2]` is the expected chi-squared statistic
       for variant :math:`j` resulting from a test of association between
       variant :math:`j` and a trait.
    *  :math:`l_j = \sum_{k} r_{jk}^2` is the LD score of variant
       :math:`j`, calculated as the sum of squared correlation coefficients
       between variant :math:`j` and nearby variants. See
       :func:`calculate_ld_scores` for further details.
    *  :math:`a` captures the contribution of confounding biases, such as
       cryptic relatedness and uncontrolled population structure, to the
       association test statistic.
    *  :math:`h_g^2` is the SNP-heritability, or the proportion of variation
       in the trait explained by the effects of variants included in the
       regression model above.
    *  :math:`M` is the number of variants used to estimate :math:`h_g^2`.
    *  :math:`N` is the number of samples in the underlying association study.

    For more details on the method implemented in this function, see:

    * `LD Score regression distinguishes confounding from polygenicity in genome-wide association studies (Bulik-Sullivan et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4495769/>`__

    Examples
    --------

    Run the method on a matrix table of summary statistics, where the rows
    are variants and the columns are different traits:

    >>> mt_gwas = hl.read_matrix_table('data/ld_score.sumstats.mt')
    >>> ht_results = hl.experimental.estimate_heritability(
    ...     z_expr=mt_gwas.Z,
    ...     n_samples_exprs=mt_gwas.N,
    ...     weight_expr=mt_gwas.ld_score,
    ...     ld_score_expr=mt_gwas.ld_score)


    Run the method on a table with summary statistics for a single
    trait:

    >>> ht_gwas = hl.read_table('data/ld_score.sumstats.ht')
    >>> ht_results = hl.experimental.estimate_heritability(
    ...     z_expr=ht_gwas.Z_50_irnt,
    ...     n_samples_expr=N_50_irnt,
    ...     weight_expr=ht_gwas.ld_score,
    ...     ld_score_expr=ht_gwas.ld_score)


    Notes
    -----

    The ``exprs`` provided as arguments to :func:`.estimate_heritability`
    must all originate from the same object, either a :class:`Table` or a
    :class:`MatrixTable`.

    **If the arguments originate from a table:**

    *  The table must be keyed by fields ``locus`` of type
       :class:`.tlocus` and ``alleles``, a :py:data:`.tarray` of
       :py:data:`.tstr` elements.
    *  ``z_expr``, ``n_samples_expr``, ``weight_expr``, and
       ``ld_score_expr`` must be row-indexed fields.

    **If the arguments originate from a matrix table:**

    *  The dimensions of the matrix table must be variants
       (rows) by traits (columns).
    *  The rows of the matrix table must be keyed by fields
       ``locus`` of type :class:`.tlocus` and ``alleles``,
       a :py:data:`.tarray` of :py:data:`.tstr` elements.
    *  The columns of the matrix table must be keyed by a field
       of type :py:data:`.tstr` that uniquely identifies traits
       represented in the matrix table. The column key must be a
       single expression; compound keys are not accepted.
    *  ``weight_expr`` and ``ld_score_expr`` must be row-indexed
       fields.

    The function returns a :class:`.Table` with the following fields:

    *  **trait** (:py:data:`.tstr`) -- The name of the trait for which
       SNP-heritability is being estimated, defined by the column key of
       the originating matrix table. If the input expressions to the 
       function originate from a table, this field is omitted.
    *  **n_samples** (:py:data:`.tfloat`) -- The mean number of samples
       across variants for the given trait.
    *  **n_variants** (:py:data:`.tint`) -- The number of variants used
       to estimate heritability.
    *  **mean_chi_sq** (:py:data:`.tfloat64`) -- The mean chi-squared
       test statistic for the given trait.
    *  **intercept** (:py:data:`.tstruct`) -- Contains fields:

       -  **estimate** (:py:data:`.tfloat64`) -- A point estimate of the
          LD score regression intercept term :math:`1 + Na`.
       -  **standard_error**  (:py:data:`.tfloat64`) -- An estimate of
          the standard error of the point estimate.

    *  **snp_heritability** (:py:data:`.tstruct`) -- Contains fields:

       -  **estimate** (:py:data:`.tfloat64`) -- A point estimate of the
          SNP-heritability :math:`h_g^2`.
       -  **standard_error** (:py:data:`.tfloat64`) -- An estimate of
          the standard error of the point estimate.

    Warning
    -------
    :func:`.estimate_heritability` considers only rows for which the
    fields ``z_expr``, ``weight_expr`` and ``ld_score_expr`` are defined. 
    Rows with missing values in any of these fields are removed prior to
    fitting the LD score regression model.

    Parameters
    ----------
    z_expr : :class:`.NumericExpression`
            A row-indexed (if table) or entry-indexed (if matrix table)
            expression for Z statistics resulting from genome-wide
            association studies.
    n_samples_exprs: :class:`.NumericExpression`
                    A row-indexed (if table) or entry-indexed
                    (if matrix table) expression indicating the number of
                    samples used in the studies that generated the
                    ``z_expr`` test statistics.
    weight_expr : :class:`.NumericExpression`
                  Row-indexed expression for the LD scores used to derive
                  variant weights in the model.
    ld_score_expr : :class:`.NumericExpression`
                    Row-indexed expression for the LD scores used as covariates
                    in the model.
    n_blocks : :obj:`int`
               The number of blocks used in the jackknife approach to
               estimating standard errors.
    two_step_threshold : :obj:`int`, optional
                         If specified, variants with chi-squared statistics greater
                         than this value are excluded while estimating the intercept
                         term in the first step of the two-step procedure used to fit
                         the model. Default behavior is to estimate the intercept
                         and SNP-heritability terms in a single step.
    n_reference_panel_variants : :obj:`int`, optional
                                 Number of variants used to estimate the LD
                                 scores used as covariates in the model. Default
                                 is number of variants for which `ld_score_expr`
                                 is defined.

    Returns
    -------
    :class:`.Table`
        Table with fields described above."""

    ds = z_expr._indices.source

    analyze('ld_score_regression/weight_expr',
            weight_expr,
            ds._row_indices)
    for i, expr in ld_score_exprs:
        analyze(f'ld_score_regression/ld_score_expr{i}',
                expr,
                ds._row_indices)

    if not n_reference_panel_variants:
        M = ds.aggregate_rows(hl.agg.count_where(
            hl.is_defined(ld_score_expr)))
    else:
        M = n_reference_panel_variants

    if isinstance(ds, MatrixTable):
        if len(list(ds.col_key)) != 1:
            raise ValueError("""Matrix table must be keyed by a single
                trait field.""")

        analyze(f'estimate_heritability/z_expr',
                z_expr,
                ds._entry_indices)
        analyze(f'estimate_heritability/n_samples_expr',
                n_samples_expr,
                ds._entry_indices)

        mt = ds._select_all(row_exprs={'locus': ds.locus,
                                       'alleles': ds.alleles,
                                       '__w_initial': weight_expr,
                                       '__w_initial_floor': hl.max(weight_expr,
                                                                   1.0),
                                       '__x': ld_score_expr,
                                       '__x_floor': hl.max(ld_score_expr,
                                                           1.0)},
                            row_key=['locus', 'alleles'],
                            col_exprs={'__y_name': ds.col_key[0]},
                            col_key=['__y_name'],
                            entry_exprs={'__y': z_expr**2,
                                         '__n': n_samples_expr})
        mt = mt.annotate_entries(__w=mt.__w_initial)

    else:
        analyze(f'estimate_heritability/z_expr',
                z_expr,
                ds._row_indices)
        analyze(f'estimate_heritability/n_samples_expr',
                n_samples_expr,
                ds._row_indices)

        ds = ds.select(**{'locus': ds.locus,
                          'alleles': ds.alleles,
                          '__w_initial': weight_expr,
                          '__w_initial_floor': hl.max(weight_expr,
                                                      1.0),
                          '__x': ld_score_expr,
                          '__x_floor': hl.max(ld_score_expr,
                                              1.0),
                          '__entries': [hl.struct(
                              __y=z_expr,
                              __w=weight_expr,
                              __n=n_samples_expr)]})
        ds = ds.annotate_globals(__cols=[hl.struct(__y_name='trait')])
        ds = ds.key_by(ds.locus, ds.alleles)
        mt = ds._unlocalize_entries('__entries', '__cols', ['__y_name'])

    mt = mt.filter_rows(hl.is_defined(mt.locus) &
                        hl.is_defined(mt.alleles) &
                        hl.is_defined(mt.__w_initial) &
                        hl.is_defined(mt.__x))

    mt_tmp1 = new_temp_file()
    mt.write(mt_tmp1)
    mt = hl.read_matrix_table(mt_tmp1)

    if two_step_threshold:
        mt = mt.annotate_entries(__in_step1=(hl.is_defined(mt.__y) &
                                             (mt.__y < two_step_threshold)),
                                 __in_step2=hl.is_defined(mt.__y))
    else:
        mt = mt.annotate_entries(__in_step1=hl.is_defined(mt.__y))

    mt = mt.annotate_cols(__n_mean=hl.agg.mean(mt.__n),
                          __m_step1=hl.float(hl.agg.count_where(mt.__in_step1)))

    mt = _assign_blocks(mt, n_blocks, two_step_threshold)

    mt = mt.annotate_cols(__step1_betas=hl.array([
        1.0, (hl.agg.mean(mt.__y) - 1.0) / hl.agg.mean(mt.__x)]))
    for i in range(3):
        mt = mt.annotate_entries(__w_step1=hl.cond(
            mt.__in_step1,
            1.0/(mt.__w_initial_floor * 2.0 * (mt.__step1_betas[0] +
                                               mt.__step1_betas[1] *
                                               mt.__x_floor)**2),
            0.0))
        mt = mt.annotate_cols(__step1_betas=hl.agg.filter(
            mt.__in_step1,
            hl.agg.linreg(y=mt.__y,
                          x=[1.0, mt.__x],
                          weight=mt.__w_step1).beta))
        mt = mt.annotate_cols(__step1_h2=hl.max(hl.min(
            mt.__step1_betas[1] * M / mt.__n_mean, 1.0), 0.0))
        mt = mt.annotate_cols(__step1_betas=hl.array([
            mt.__step1_betas[0],
            mt.__step1_h2 * mt.__n_mean / M]))

    mt = mt.annotate_cols(__step1_block_betas=_block_betas(
        block_expr=mt.__step1_block,
        include_expr=mt.__in_step1,
        y_expr=mt.__y,
        covariates=[1.0, mt.__x],
        w_expr=mt.__w_step1,
        n_blocks=n_blocks))

    mt = mt.annotate_cols(__step1_jackknife_variances=hl.map(
        lambda i: hl.rbind(
            _pseudovalues(mt.__step1_betas[i],
                          hl.map(lambda block: block[i],
                                 mt.__step1_block_betas)),
            lambda pseudovalues: _variance(pseudovalues)),
        hl.range(0, hl.len(mt.__step1_betas))))

    if two_step_threshold:
        mt = mt.annotate_cols(__initial_betas=hl.array([
            1.0, (hl.agg.mean(mt.__y) - 1.0) / hl.agg.mean(mt.__x)]))
        mt = mt.annotate_cols(__step2_betas=mt.__initial_betas)
        for i in range(3):
            mt = mt.annotate_entries(__w_step2=hl.cond(
                mt.__in_step2,
                1.0/(mt.__w_initial_floor *
                     2.0 * (mt.__step2_betas[0] +
                            mt.__step2_betas[1] *
                            mt.__x_floor)**2),
                0.0))
            mt = mt.annotate_cols(__step2_betas=hl.array([
                mt.__step1_betas[0],
                hl.agg.filter(
                    mt.__in_step2,
                    hl.agg.linreg(y=mt.__y - mt.__step1_betas[0],
                                  x=[mt.__x],
                                  weight=mt.__w_step2).beta[0])]))
            mt = mt.annotate_cols(__step2_h2=hl.max(hl.min(
                mt.__step2_betas[1] * M / mt.__n_mean, 1.0), 0.0))
            mt = mt.annotate_cols(__step2_betas=hl.array([
                mt.__step1_betas[0],
                mt.__step2_h2 * mt.__n_mean / M]))
    
        mt = mt.annotate_cols(__step2_block_betas=_block_betas(
            block_expr=mt.__step2_block,
            include_expr=mt.__in_step2,
            y_expr=mt.__y - mt.__step1_betas[0],
            covariates=[mt.__x],
            w_expr=mt.__w_step2,
            n_blocks=n_blocks))

        mt = mt.annotate_cols(__step2_jackknife_variances=hl.map(
            lambda i: hl.rbind(
                _pseudovalues(mt.__step2_betas[i],
                              hl.map(lambda block: block[i],
                                     mt.__step2_block_betas)),
                lambda pseudovalues: _variance(pseudovalues)),
            hl.range(0, hl.len(mt.__step2_betas))))

        # combine step 1 and step 2 block jackknifes
        mt = mt.annotate_entries(
            __initial_w=1.0/(mt.__w_initial_floor *
                             2.0 * (mt.__initial_betas[0] +
                                    mt.__initial_betas[1] *
                                    mt.__x_floor)**2))

        mt = mt.annotate_cols(__final_block_betas=hl.rbind(
            (hl.agg.sum(mt.__initial_w * mt.__x) /
             hl.agg.sum(mt.__initial_w * mt.__x**2)),
            lambda c: hl.map(
                    lambda i: hl.rbind(
                        mt.__step2_block_betas[i] - c * (mt.__step1_block_betas[i][0] - mt.__step1_betas[0]),
                        lambda final_block_beta: n_blocks * mt.__step2_betas[1] - (n_blocks - 1) * final_block_beta),
                    hl.range(0, n_blocks))))

        mt = mt.annotate_cols(
            __final_betas=hl.array([
                mt.__step1_betas[0], mt.__step2_betas[1]]),
            __final_jackknife_variances=hl.array([
                mt.__step1_jackknife_variance[0],
                _variance(mt.__final_block_betas)]))

    else:
        mt = mt.annotate_cols(
            __final_betas=mt.__step1_betas,
            __final_jackknife_variances=mt.__step1_jackknife_variances,
            __final_block_betas=hl.map(lambda block: block[1], mt.__step1_block_betas))

    mt = mt.annotate_cols(
        trait=mt.__y_name,
        n_samples=mt.__n_mean,
        n_variants=M,
        mean_chi_sq=hl.agg.mean(mt.__y),
        intercept=hl.struct(
            estimate=mt.__final_betas[0],
            standard_error=hl.sqrt(mt.__final_jackknife_variances[0])),
        snp_heritability=hl.struct(
            estimate=(M / mt.__n_mean) * mt.__final_betas[1],
            standard_error=hl.sqrt((M / mt.__n_mean)**2 *
                                   mt.__final_jackknife_variances[1])))

    ht = mt.cols()
    ht = ht.key_by(ht.trait)
    if _return_block_estimates:
        ht = ht.select(ht.n_samples,
                       ht.n_variants,
                       ht.mean_chi_sq,
                       ht.intercept,
                       ht.snp_heritability,
                       ht.__final_block_betas)
    else:
        ht = ht.select(ht.n_samples,
                       ht.n_variants,
                       ht.mean_chi_sq,
                       ht.intercept,
                       ht.snp_heritability)

    ht_tmp_file = new_temp_file()
    ht.write(ht_tmp_file)
    ht = hl.read_table(ht_tmp_file)

    return ht


@typecheck(z_expr=expr_float64,
           n_samples_expr=expr_numeric,
           weight_expr=expr_float64,
           ld_score_expr=expr_numeric,
           n_blocks=int,
           two_step_threshold=nullable(int),
           n_reference_panel_variants=nullable(int))
def estimate_genetic_correlation(z_expr,
                                 n_samples_expr,
                                 weight_expr,
                                 ld_score_expr,
                                 n_blocks=200,
                                 two_step_threshold=None,
                                 n_reference_panel_variants=None) -> Table:
    r"""Estimate genetic correlation from GWAS summary statistics.

    Given a set or multiple sets of genome-wide association study (GWAS)
    summary statistics, :func:`.estimate_genetic_correlation` estimates the
    genetic correlation between pairs of traits using the LD score regression
    method. This leverages the model:

    .. math::

        \mathrm{E}[Z_{1j}Z_{2j}] = \frac{N_s\rho}{\sqrt{N_{1}N_{2}}} + \frac{\sqrt{N_{1}N_{2}}\rho_{g}}{M}l_j

    *   :math:`Z_{1j}` and :math:`Z_{2j}` are the statistics resulting from
        tests of association between variant :math:`j` and traits :math:`1`
        and :math:`2`, respectively.
    *   :math:`\rho` is the phenotypic correlation between traits :math:`1`
        and :math:`2`.
    *   :math:`N_1` and :math:`N_2` are the sample sizes of the association
        studies of traits :math:`1` and :math:`2`.
    *   :math:`N_s` is the number of overlap samples included in the association
        studies of both traits. Note that in the case where the same set of
        samples are used in the association studies of both trait :math:`1` and
        trait :math:`2`, the intercept term in the model above simplifies
        to :math:`\rho`.
    *   :math:`M` is the number of variants.
    *   :math:`l_j = \sum_{k} r_{jk}^2` is the LD score of variant
        :math:`j`, calculated as the sum of squared correlation coefficients
        between variant :math:`j` and nearby variants.
        See :func:`.calculate_ld_scores` for further details.
    *   :math:`\rho_g` is the genetic covariance between traits :math:`1` and
        :math:`2`.

    Normalizing the genetic covariance :math:`\rho_g` by the SNP heritiabilities
    :math:`h^2_1` and :math:`h^2_2` obtained using the method implemented in
    :func:`.estimate_heritability` yields the genetic correlation, :math:`r_g`,
    between traits :math:`1` and :math:`2`:

    .. math::

        r_g = \frac{\rho_g}{\sqrt{h^2_{1}h^2_{2}}}

    For more details on the method implemented in :func:`.estimate_genetic_correlation`,
    see:

    * `An atlas of genetic correlations across human diseases and traits (Bulik-Sullivan et al, 2015) <https://www.ncbi.nlm.nih.gov/pubmed/26414676>`__


    Examples
    --------

    Run the method on a matrix table of summary statistics, where the rows
    are variants and the columns are different traits:

    >>> mt_gwas = hl.read_matrix_table('data/ld_score.sumstats.mt')
    >>> ht_results = hl.experimental.ld_score.estimate_genetic_correlation(
    ...     z_expr=mt_gwas.Z,
    ...     n_samples_expr=mt_gwas.N,
    ...     weight_expr=mt_gwas.ld_score,
    ...     ld_score_expr=mt_gwas.ld_score)


    Notes
    -----

    The ``exprs`` provided as arguments to :func:`.estimate_genetic_correlation`
    must all originate from a :class:`MatrixTable`.

    *  The dimensions of the matrix table must be variants
       (rows) by traits (columns).
    *  The rows of the matrix table must be keyed by fields
       ``locus`` of type :class:`.tlocus` and ``alleles``,
       a :py:data:`.tarray` of :py:data:`.tstr` elements.
    *  The columns of the matrix table must be keyed by a field
       of type :py:data:`.tstr` that uniquely identifies traits
       represented in the matrix table. The column key must be a single
       expression; compound keys are not accepted.
    *  ``weight_expr`` and ``ld_score_expr`` must be row-indexed
       fields.
    *  ``z_expr`` and ``n_samples_expr`` must be entry-indexed fields.
       (not a list of fields).

    The function returns a :class:`Table` with one row per pair of traits (columns)
    in the matrix table from which the input expressions originate. The table
    contains the following fields:

    *  **traits** (:py:data:`.tarray` of :py:data:`.tstr`) -- An array whose two
       elements are the names of traits for which genetic correlation is being
       estimated, as defined by the column key values of the originating matrix
       table. The table returned is keyed by this field.
    *  **n_samples** (:py:data:`.tfloat64`) -- The mean value across variants of
       the term :math:`\sqrt{N_{1}N_{2}}`.
    *  **n_variants** (:py:data:`.tint`) -- The number of variants used to
       estimate genetic correlation.
    *  **mean_z1z2** (:py:data:`.tfloat64`) -- The mean across variants of the
       products of trait 1 and trait 2 GWAS association statistics.
    *  **intercept** (:py:data:`.tstruct`) -- Contains fields:

       -  **estimate** (:py:data:`.tfloat64`) -- A point estimate of the
          intercept :math:`\frac{N_s\rho}{\sqrt{N_{1}N_{2}}}`.
       -  **standard_error**  (:py:data:`.tfloat64`) -- An estimate of
          the standard error of the point estimate.
    *  **genetic_covariance** (:py:data:`.tstruct`) -- Contains fields:

       -  **estimate** (:py:data:`.tfloat64`) -- A point estimate of the
          genetic covariance :math:`\rho_g`.
       -  **standard_error** (:py:data:`.tfloat64`) -- An estimate of
          the standard error of the point estimate.
    *  **genetic_correlation** (:py:data:`.tstruct`) -- Contains fields:

       -  **estimate** (:py:data:`.tfloat64`) -- A point estimate of the
          genetic correlation :math:`r_g`.
       -  **standard_error** (:py:data:`.tfloat64`) -- An estimate of the
          standard error of the point estimate.
       -  **Z_score** (:py:data:`.tfloat64`) -- Z-score from the association
          test where the null hypothesis is :math:`r_g = 0.0` and the 
          alternative hypothesis is :math:`r_g \neq 0.0`.
       -  **p_value** (:py:data:`.tfloat64`) -- P-value for the :math:`r_g`
          association test.


    Warning
    -------

    For each pair of traits, :func:`.estimate_genetic_correlation` considers
    only the variants for which ``z_expr`` is defined for both traits and for
    which the row fields ``weight_expr`` and ``ld_score_expr`` are also defined.
    Rows with missing values in any of these fields are removed prior to fitting
    the LD score regression model.


    Parameters
    ----------
    z_expr : :class:`.Float64Expression`
        An entry-indexed expression for Z statistics
        resulting from association studies.
    n_samples_expr: :class:`.NumericExpression`
        An entry-indexed expression for the number of samples
        used in the underlying assosciation studies that generated
        the ``z_expr`` statistics.
    weight_expr : :class:`.Float64Expression`
        Row-indexed expression for the LD scores used to derive
        variant weights in the model.
    ld_score_expr : :class:`.Float64Expression`
        Row-indexed expression for the LD scores used as covariates
        in the model.
    n_blocks : :obj:`int`
        The number of blocks used in the jackknife approach to
        estimating standard errors.
    two_step_threshold : :obj:`int`
        Variants with chi-squared statistics greater than this
        value are excluded in the first step of the two-step
        procedure used to fit the model.
    n_reference_panel_variants : :obj:`int`, optional
        Number of variants used to estimate the
        genetic correlation.

    Returns
    -------
    :class:`.Table`
        Table with fields described above."""

    ds = z_expr._indices.source

    analyze('estimate_heritability/locus_expr',
            ds.locus,
            ds._row_indices)
    analyze('estimate_heritability/alleles_expr',
            ds.alleles,
            ds._row_indices)
    analyze('estimate_heritability/weight_expr',
            weight_expr,
            ds._row_indices)
    analyze('estimate_heritability/ld_score_expr',
            ld_score_expr,
            ds._row_indices)

    if not n_reference_panel_variants:
        M = mt.aggregate_rows(hl.agg.count_where(
            hl.is_defined(ds.ld_score_expr)))
    else:
        M = n_reference_panel_variants

    if len(list(ds.col_key)) != 1:
        raise ValueError("""Matrix table must be keyed by a single
            trait field.""")

    analyze(f'estimate_heritability/z_expr',
            z_expr,
            ds._entry_indices)
    analyze(f'estimate_heritability/n_samples_expr',
            n_samples_expr,
            ds._entry_indices)

    mt = ds._select_all(row_exprs={'locus': ds.locus,
                                   'alleles': ds.alleles,
                                   '__w_initial': weight_expr,
                                   '__w_initial_floor': hl.max(weight_expr,
                                                               1.0),
                                   '__x': ld_score_expr,
                                   '__x_floor': hl.max(ld_score_expr,
                                                       1.0)},
                        row_key=['locus', 'alleles'],
                        col_exprs={'__y_name': ds.col_key[0]},
                        col_key=['__y_name'],
                        entry_exprs={'__y': z_expr,
                                     '__n': n_samples_expr})
    mt = mt.annotate_entries(__w=mt.__w_initial)

    mt = mt.filter_rows(hl.is_defined(mt.locus) &
                        hl.is_defined(mt.alleles) &
                        hl.is_defined(mt.__w_initial) &
                        hl.is_defined(mt.__x))

    mt_tmp_file1 = new_temp_file()
    mt.write(mt_tmp_file1)
    mt = hl.read_matrix_table(mt_tmp_file1)

    ht_h2 = estimate_heritability(
        z_expr=mt.__y,
        n_samples_expr=mt.__n,
        weight_expr=mt.__x,
        ld_score_expr=mt.__x,
        n_blocks=n_blocks,
        two_step_threshold=two_step_threshold,
        n_reference_panel_variants=n_reference_panel_variants,
        _return_block_estimates=True)

    ht_h2_tmp = new_temp_file()
    ht_h2.write(ht_h2_tmp)
    ht_h2 = hl.read_table(ht_h2_tmp).cache()

    ht = mt.localize_entries(entries_array_field_name='__entries',
                             columns_array_field_name='__cols')

    n_ys = hl.eval(hl.len(ht.__cols))
    pairs = [(i, j) for i in range(n_ys) for j in range(i+1, n_ys)]

    ht = ht.annotate_globals(__cols=hl.map(
        lambda pair: hl.struct(
                __y_name=hl.struct(
                    __y0=ht.__cols[pair[0]].__y_name,
                    __y1=ht.__cols[pair[1]].__y_name)),
        hl.array(pairs)))

    ht = ht.annotate(__entries=hl.map(
        lambda pair: hl.rbind(
            ht.__entries[pair[0]].__y,
            ht.__entries[pair[1]].__y,
            ht.__entries[pair[0]].__n,
            ht.__entries[pair[1]].__n,
            ht.__entries[pair[0]].__w,
            lambda y0, y1, n0, n1, w: hl.struct(
                __y=(y0 * y1),
                __n=hl.sqrt(hl.int64(n0) * hl.int64(n1)),
                __w=w)),
        hl.array(pairs)))

    mt = ht._unlocalize_entries('__entries', '__cols', ['__y_name'])

    mt_tmp_file2 = new_temp_file()
    mt.write(mt_tmp_file2)
    mt = hl.read_matrix_table(mt_tmp_file2)

    if two_step_threshold:
        mt = mt.annotate_entries(__in_step1=(hl.is_defined(mt.__y) &
                                             (mt.__y < two_step_threshold)),
                                 __in_step2=hl.is_defined(mt.__y))
    else:
        mt = mt.annotate_entries(__in_step1=hl.is_defined(mt.__y))

    mt = mt.annotate_cols(__n_mean=hl.agg.mean(mt.__n),
                          __m_step1=hl.float(hl.agg.count_where(mt.__in_step1)))

    mt = _assign_blocks(mt, n_blocks, two_step_threshold)

    mt = mt.annotate_cols(__step1_betas=hl.array([
        0.0, hl.agg.mean(mt.__y) / hl.agg.mean(mt.__x)]))
    for i in range(3):
        mt = mt.annotate_entries(__w_step1=hl.cond(
            mt.__in_step1,
            hl.rbind(
                ht_h2[mt.__y_name[0]],
                ht_h2[mt.__y_name[1]],
                lambda y0, y1: hl.rbind(
                    y0.intercept.estimate + (y0.n_samples / M) * y0.snp_heritability.estimate * mt.__x_floor,
                    y1.intercept.estimate + (y1.n_samples / M) * y1.snp_heritability.estimate * mt.__x_floor,
                    mt.__step1_betas[0] + mt.__step1_betas[1] * mt.__x_floor,
                    lambda a, b, c: 1.0 / (mt.__w_initial_floor * (a * b + c**2)))),
            0.0))
        mt = mt.annotate_cols(__step1_betas=hl.agg.filter(
            mt.__in_step1,
            hl.agg.linreg(y=mt.__y,
                          x=[1.0, mt.__x],
                          weight=mt.__w_step1).beta))
        mt = mt.annotate_cols(__step1_rho_g=hl.max(hl.min(
            mt.__step1_betas[1] * M / mt.__n_mean, 1.0), -1.0))
        mt = mt.annotate_cols(__step1_betas=hl.array([
            mt.__step1_betas[0],
            mt.__step1_rho_g * mt.__n_mean / M]))

    mt = mt.annotate_cols(__step1_block_betas=_block_betas(
        block_expr=mt.__step1_block,
        include_expr=mt.__in_step1,
        y_expr=mt.__y,
        covariates=[1.0, mt.__x],
        w_expr=mt.__w_step1,
        n_blocks=n_blocks))

    mt = mt.annotate_cols(__step1_jackknife_variances=hl.map(
        lambda i: hl.rbind(
            _pseudovalues(mt.__step1_betas[i],
                          hl.map(lambda block: block[i],
                                 mt.__step1_block_betas)),
            lambda pseudovalues: _variance(pseudovalues)),
        hl.range(0, hl.len(mt.__step1_betas))))

    if two_step_threshold:
        mt = mt.annotate_cols(__step2_betas=hl.array([
            0.0, (hl.agg.mean(mt.__y) - 1.0) / hl.agg.mean(mt.__x)]))
        for i in range(3):
            mt = mt.annotate_entries(__w_step2=hl.cond(
                mt.__in_step2,
                hl.rbind(
                    ht_h2[mt.__y_name[0]],
                    ht_h2[mt.__y_name[1]],
                    lambda y0, y1: hl.rbind(
                        y0.intercept.estimate + (y0.n_samples / M) * y0.snp_heritability.estimate * mt.__x_floor,
                        y1.intercept.estimate + (y1.n_samples / M) * y1.snp_heritability.estimate * mt.__x_floor,
                        mt.__step2_betas[0] + mt.__step2_betas[1] * mt.__x_floor,
                        lambda a, b, c: 1.0 / (mt.__w_initial_floor * (a * b + c**2)))),
                0.0))
            mt = mt.annotate_cols(__step2_betas=hl.array([
                mt.__step1_betas[0],
                hl.agg.filter(
                    mt.__in_step2,
                    hl.agg.linreg(y=mt.__y - mt.__step1_betas[0],
                                  x=[mt.__x],
                                  weight=mt.__w_step2).beta[0])]))
            mt = mt.annotate_cols(__step2_rho_g=hl.max(hl.min(
                mt.__step2_betas[1] * M / mt.__n_mean, 1.0), -1.0))
            mt = mt.annotate_cols(__step2_betas=hl.array([
                mt.__step1_betas[0],
                mt.__step2_rho_g * mt.__n_mean / M]))

        mt = mt.annotate_cols(__step2_block_betas=_block_betas(
            block_expr=mt.__step2_block,
            include_expr=mt.__in_step2,
            y_expr=mt.__y - mt.__step1_betas[0],
            covariates=[mt.__x],
            w_expr=mt.__w_step2,
            n_blocks=n_blocks))

        mt = mt.annotate_cols(__step2_jackknife_variances=hl.map(
            lambda i: hl.rbind(
                _pseudovalues(mt.__step2_betas[i],
                              hl.map(lambda block: block[i],
                                     mt.__step2_block_betas)),
                lambda pseudovalues: _variance(pseudovalues)),
            hl.range(0, hl.len(mt.__step2_betas))))

        # combine step 1 and step 2 block jackknifes
        mt = mt.annotate_entries(
            __step2_initial_w=1.0/(mt.__w_initial_floor *
                                   2.0 * (mt.__initial_betas[0] +
                                          mt.__initial_betas[1] *
                                          mt.__x_floor)**2))

        mt = mt.annotate_cols(__final_block_betas=hl.rbind(
            (hl.agg.sum(mt.__step2_initial_w * mt.__x) /
             hl.agg.sum(mt.__step2_initial_w * mt.__x**2)),
            lambda c: hl.map(
                    lambda i: hl.rbind(
                        mt.__step2_block_betas[i] - c * (mt.__step1_block_betas[i][0] - mt.__step1_betas[0]),
                        lambda final_block_beta: n_blocks * mt.__step2_betas[1] - (n_blocks - 1) * final_block_beta),
                    hl.range(0, n_blocks))))

        mt = mt.annotate_cols(
            __final_betas=hl.array([
                mt.__step1_betas[0], mt.__step2_betas[1]]),
            __final_jackknife_variances=hl.array([
                mt.__step1_jackknife_variance[0],
                _variance(mt.__final_block_betas)]))

    else:
        mt = mt.annotate_cols(
            __final_betas=mt.__step1_betas,
            __final_jackknife_variances=mt.__step1_jackknife_variances,
            __final_block_betas=hl.map(lambda block: block[1], mt.__step1_block_betas))

    mt = mt.annotate_cols(
        traits=hl.array([
            mt.__y_name.__y0,
            mt.__y_name.__y1]),
        n_samples=mt.__n_mean,
        n_variants=M,
        mean_z1z2=hl.agg.mean(mt.__y),
        intercept=hl.struct(
            estimate=mt.__final_betas[0],
            standard_error=hl.sqrt(mt.__final_jackknife_variances[0])),
        genetic_covariance=hl.struct(
            estimate=(M / mt.__n_mean) * mt.__final_betas[1],
            standard_error=hl.sqrt((M / mt.__n_mean)**2 *
                                   mt.__final_jackknife_variances[1])))

    mt = mt.annotate_cols(genetic_correlation=hl.rbind(
        ht_h2[mt.__y_name.__y0].snp_heritability.estimate,
        ht_h2[mt.__y_name.__y1].snp_heritability.estimate,
        ht_h2[mt.__y_name.__y0].__final_block_betas,
        ht_h2[mt.__y_name.__y1].__final_block_betas,
        lambda h2_0, h2_1, blocks_0, blocks_1: hl.rbind(
            mt.genetic_covariance.estimate / hl.sqrt(h2_0 * h2_1),
            lambda rg: hl.struct(
                estimate=rg,
                standard_error=hl.rbind(
                    hl.map(lambda i: mt.__final_block_betas[i] / hl.sqrt(blocks_0[i] * blocks_1[i]),
                           hl.range(0, n_blocks)),
                    lambda block_values: hl.sqrt(_variance(_pseudovalues(rg, block_values))))))))

    mt = mt.annotate_cols(
        genetic_correlation=hl.rbind(
            mt.genetic_correlation.estimate / mt.genetic_correlation.standard_error,
            lambda Z: mt.genetic_correlation.annotate(
                Z_score=Z,
                p_value=hl.pchisqtail(x=Z**2, df=1))))

    ht = mt.cols()
    ht = ht.key_by(ht.traits)
    ht = ht.select(ht.n_samples,
                   ht.n_variants,
                   ht.mean_z1z2,
                   ht.intercept,
                   ht.genetic_covariance,
                   ht.genetic_correlation)

    ht_tmp_file = new_temp_file()
    ht.write(ht_tmp_file)
    ht = hl.read_table(ht_tmp_file)

    return ht

