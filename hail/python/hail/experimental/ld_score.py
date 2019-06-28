
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
    ht = mt.localize_entries(entries_array_field_name='_entries',
                             columns_array_field_name='_cols')

    if two_step_threshold:
        ht = ht.annotate(_entries=hl.rbind(
            hl.scan.array_agg(
                lambda entry: hl.scan.count_where(entry['_in_step1']),
                ht['_entries']),
            lambda step1_indices: hl.map(
                lambda i: hl.rbind(
                    hl.int(hl.or_else(step1_indices[i], 0)),
                    ht['_cols'][i]['_m_step1'],
                    ht['_entries'][i],
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
                                _step1_block=step1_block,
                                _step2_block=hl.cond(~entry['_in_step1'] & is_separator,
                                                      step1_block - 1,
                                                      step1_block))))),
                hl.range(0, hl.len(ht['_entries'])))))
    else:
        ht = ht.annotate(_entries=hl.rbind(
            hl.scan.array_agg(
                lambda entry: hl.scan.count_where(entry['_in_step1']),
                ht['_entries']),
            lambda step1_indices: hl.map(
                lambda i: hl.rbind(
                    hl.int(hl.or_else(step1_indices[i], 0)),
                    ht['_cols'][i]['_m_step1'],
                    ht['_entries'][i],
                    lambda step1_idx, m_step1, entry: hl.rbind(
                        hl.map(
                            lambda j: hl.int(hl.floor(j * (m_step1 / n_blocks))),
                            hl.range(0, n_blocks + 1)),
                        lambda step1_separators: entry.annotate(
                            _step1_block=hl.sum(
                                hl.map(
                                    lambda s1: step1_idx >= s1,
                                    step1_separators)) - 1))),
                hl.range(0, hl.len(ht['_entries'])))))

    mt = ht._unlocalize_entries('_entries', '_cols', col_keys)

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



@typecheck(ld_matrix=BlockMatrix,
           annotation_exprs=oneof(expr_numeric,
                                  sequenceof(expr_numeric)))
def compute_ld_scores(ld_matrix,
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
           n_reference_panel_variants_exprs=nullable(oneof(expr_numeric,
                                                           sequenceof(expr_numeric))),
           n_blocks=int,
           two_step_threshold=nullable(int),
           rg_pairs=nullable(sequenceof(sized_tupleof(expr_numeric, expr_numeric))),
           n_iterations=nullable(int),
           max_chi_sq=nullable(float))
def ld_score_regression(z_expr,
                        n_samples_expr,
                        ld_score_exprs,
                        weight_expr,
                        n_reference_panel_variants_exprs,
                        n_blocks=200,
                        two_step_threshold=None,
                        rg_pairs=None,
                        n_iterations=3,
                        max_chi_sq=None):
    r"""Estimate SNP-heritability, level of confounding biases, and
    genetic correlation from GWAS summary statistics.

    SNP Heritability
    ----------------

    Given genome-wide association study (GWAS) summary statistics,
    :func:`.ld_score_regression` estimates the heritability of a
    trait or set of traits and the level of confounding biases present in
    the underlying association studies using either a single LD score
    per variant (univariate LD score regression) or a set of annotation-specific
    LD scores per variant (partitioned or stratified LD score regression).

    In univariate LD score regression, this function fits the model:

    .. math::

        \mathrm{E}[\chi_j^2] = 1 + Na + \frac{Nh_g^2}{M}l_j

    *  :math:`\mathrm{E}[\chi_j^2]` is the expected chi-squared statistic
       for variant :math:`j` resulting from a test of association between
       variant :math:`j` and a trait.
    *  :math:`l_j = \sum_{k} r_{jk}^2` is the LD score of variant
       :math:`j`, calculated as the sum of squared correlation coefficients
       between variant :math:`j` and nearby variants.
    *  :math:`a` captures the contribution of confounding biases, such as
       cryptic relatedness and uncontrolled population structure, to the
       association test statistic.
    *  :math:`h_g^2` is the SNP-heritability, or the proportion of variation
       in the trait explained by the effects of variants included in the
       regression model above.
    *  :math:`M` is the number of variants used to estimate :math:`h_g^2`.
    *  :math:`N` is the number of samples in the underlying association study.

    In partitioned LD score regression, this function fits the model:

    .. math::

        \mathrm{E}[\chi_j^2] = 1 + Na + N\sum_C\tau_Cl(j,C)

    *  :math:`\mathrm{E}[\chi_j^2]` is the expected chi-squared statistic
       for variant :math:`j` resulting from a test of association between
       variant :math:`j` and a trait.


    Genetic Correlation
    -------------------

    By utilizing the ``rg_pairs`` argument, :func:`.ld_score_regression` can
    also estimate genetic correlation between pairs of traits.

    < to do: document>

    For more details on the methods implemented in this function, see:

    * `LD Score regression distinguishes confounding from polygenicity in genome-wide association studies (Bulik-Sullivan et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4495769/>`__
    * `Partitioning heritability by functional annotation using genome-wide association summary statistics (Finucane et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4626285/>`__
    * `An atlas of genetic correlations across human diseases and traits (Bulik-Sullivan et al, 2015) < https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4797329/>`__

    Examples
    --------

    Run univariate LD score regression on a matrix table of summary statistics,
    where the rows are variants and the columns are different traits:

    >>> mt_gwas = hl.read_matrix_table('data/ld_score_regression.sample.mt')
    >>> ht_results = hl.experimental.ld_score_regression(
    ...     z_expr=mt_gwas['Z'],
    ...     n_samples_exprsmt_gwas['N'],
    ...     weight_expr=mt_gwas['weight'],
    ...     ld_score_exprs=[mt_gwas[x] for x in list(mt_gwas['ld_scores'])][0],
    ...     n_reference_panel_variants_exprs=[mt_gwas[x] for x in list(mt_gwas['M_5_50'])][0],
    ...     two_step_threshold=30,
    ...     n_iterations=3)

    Run partitioned LD score regression on the same matrix table:

    >>> mt_gwas = hl.read_matrix_table('data/ld_score_regression.sample.mt')
    >>> ht_results = hl.experimental.ld_score_regression(
    ...     z_expr=mt_gwas['Z'],
    ...     n_samples_expr=mt_gwas['N'],
    ...     weight_expr=mt_gwas['weight'],
    ...     ld_score_exprs=[mt_gwas[x] for x in list(mt_gwas['ld_scores'])],
    ...     n_reference_panel_variants_exprs=[mt_gwas[x] for x in list(mt_gwas['M_5_50'])],
    ...     two_step_threshold=None,
    ...     n_iterations=1,
    ...     max_chi_sq=80)

    Run univariate LD score regression on a table with summary statistics
    for a single trait:

    >>> mt_gwas = hl.read_matrix_table('data/ld_score_regression.sample.mt')
    >>> ht_gwas = mt_gwas.filter_cols(mt['phenotype'] == '50_irnt').entries()
    >>> ht_results = hl.experimental.ld_score_regression(
    ...     z_expr=ht_gwas['Z_50_irnt'],
    ...     n_samples_expr=ht_gwas['N_50_irnt'],
    ...     weight_expr=ht_gwas['weight'],
    ...     ld_score_expr=[ht_gwas[x] for x in list(ht_gwas['ld_scores'])][0],
    ...     n_reference_panel_variants_exprs=[ht_gwas[x] for x in list(ht_gwas['M_5_50'])][0],
    ...     two_step_threshold=30,
    ...     n_iterations=3)


    Notes
    -----

    The ``exprs`` provided as arguments to :func:`.ld_score_regression`
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
    *  ``weight_expr`` and ``ld_score_exprs`` must be row-indexed
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
    :func:`.ld_score_regression` considers only rows for which the
    fields ``z_expr``, ``weight_expr`` and ``ld_score_exprs`` are defined. 
    Rows with missing values in any of these fields are removed prior to
    fitting the LD score regression model.

    Parameters
    ----------
    z_expr : :class:`.NumericExpression`
            A row-indexed (if table) or entry-indexed (if matrix table)
            expression for Z statistics resulting from genome-wide
            association studies.
    n_samples_expr: :class:`.NumericExpression`
                    A row-indexed (if table) or entry-indexed
                    (if matrix table) expression indicating the number of
                    samples used in the studies that generated the
                    ``z_expr`` test statistics.
    weight_expr : :class:`.NumericExpression`
                  Row-indexed expression for the LD scores used to derive
                  variant weights in the model.
    ld_score_exprs : :class:`.NumericExpression`
                    Row-indexed expression for the LD scores used as covariates
                    in the model.
    n_reference_panel_variants_exprs : :obj:`int`, optional
                                 Number of variants used to estimate the LD
                                 scores used as covariates in the model. Default
                                 is number of variants for which `ld_score_expr`
                                 is defined.
    n_blocks : :obj:`int`
               The number of blocks used in the jackknife approach to
               estimating standard errors.
    two_step_threshold : :obj:`int`, optional
                         If specified, variants with chi-squared statistics greater
                         than this value are excluded while estimating the intercept
                         term in the first step of the two-step procedure used to fit
                         the model. Default behavior is to estimate the intercept
                         and SNP-heritability terms in a single step.
    max_chi_sq : :obj:`int`, optional
                 Summary statistics with chi-squared test statistics greater than
                 this value are removed prior to fitting the model.

    Returns
    -------
    :class:`.Table`
        Table with fields described above."""

    ds = z_expr._indices.source

    analyze('ld_score_regression/weight_expr',
            weight_expr,
            ds._row_indices)

    ld_score_exprs = wrap_to_list(ld_score_exprs)
    n_reference_panel_variants_exprs = wrap_to_list(n_reference_panel_variants_exprs)

    k = len(ld_score_exprs)
    if k != len(n_reference_panel_variants_exprs):
        raise ValueError(
            """The same number of expressions must be passed to both the "ld_score_exprs"
               and "n_reference_panel_variants_exprs" arguments.""")

    for i, expr in enumerate(ld_score_exprs):
        analyze(f'ld_score_regression/ld_score_expr{i}',
                expr,
                ds._row_indices)

    if isinstance(ds, MatrixTable):
        analyze(f'estimate_heritability/z_expr',
                z_expr,
                ds._entry_indices)
        analyze('ld_score_regression/n_samples_expr',
                n_samples_expr,
                ds._entry_indices)

        mt = ds._select_all(
            row_exprs={'locus': ds['locus'],
                       'alleles': ds['alleles'],
                       '_w_initial': weight_expr,
                       '_w_initial_floor': hl.max(
                           weight_expr, 1.0),
                       '_x': hl.array(ld_score_exprs),
                       '_x_floor': hl.map(
                           lambda score: hl.max(score, 1.0),
                           ld_score_exprs)},
            row_key=['locus', 'alleles'],
            col_exprs={'_y_name': ds.col_key[0]},
            col_key=['_y_name'],
            entry_exprs={'_y': z_expr**2,
                         '_n': n_samples_expr},
            global_exprs={'_m': hl.array(n_reference_panel_variants_exprs)})
        mt = mt.annotate_entries(_w=mt['_w_initial'])

    else:
        if not isinstance(ds, Table):
            raise ValueError(
                '"z_expr" must originate from a Hail table or matrix table.')

        ds = ds.select(
            **{'locus': ds['locus'],
               'alleles': ds['alleles'],
               '_w_initial': weight_expr,
               '_w_initial_floor': hl.max(
                   weight_expr, 1.0),
               '_x': hl.array(ld_score_exprs),
               '_x_floor': hl.map(
                    lambda score: hl.max(score, 1.0),
                    ld_score_exprs),
               '_entries': [hl.struct(
                   _y=z_expr**2,
                   _w=weight_expr,
                   _n=n_samples_expr)]})
        ds = ds.annotate_globals(_cols=[hl.struct(_y_name='trait')])
        ds = ds.key_by('locus', 'alleles')
        mt = ds._unlocalize_entries('_entries', '_cols', ['_y_name'])
        mt = mt.annotate_globals(_m=hl.array(n_reference_panel_variants_exprs))

    mt = mt.filter_rows(
        hl.is_defined(mt['locus']) &
        hl.is_defined(mt['alleles']) &
        hl.is_defined(mt['_w_initial']) &
        hl.all(lambda x: hl.is_defined(x), mt['_x']))

    if k > 1:
        if max_chi_sq is None:
            mt = mt.annotate_cols(_max_chi_sq=hl.max(0.001*hl.agg.max(mt['_n']), 80))
        else:
            mt = mt.annotate_cols(_max_chi_sq=max_chi_sq)
    else:
        mt = mt.annotate_cols(_max_chi_sq=hl.agg.max(mt['_y']))

    if two_step_threshold:
        mt = mt.annotate_entries(
            _in_step1=(hl.is_defined(mt['_y']) & (mt['_y'] < two_step_threshold) & (mt['_y'] <= mt['_max_chi_sq'])),
            _in_step2=(hl.is_defined(mt['_y']) & (mt['_y'] <= mt['_max_chi_sq'])))
    else:
        mt = mt.annotate_entries(
            _in_step1=(hl.is_defined(mt['_y']) & (mt['_y'] <= mt['_max_chi_sq'])))

    mt = mt.annotate_cols(
        _n_mean=hl.agg.mean(mt['_n']),
        _m_step1=hl.float(hl.agg.count_where(mt['_in_step1'])))

    mt = _assign_blocks(mt, n_blocks, two_step_threshold)

    if k > 1:
        mt = mt.annotate_rows(_x_total=hl.sum(mt['_x']))
        mt = mt.annotate_globals(_m_total=hl.sum(mt['_m']))
        mt = mt.annotate_cols(
            _step1_h2=(
                mt['_m_total'] * (hl.agg.filter(mt['_y'] < mt['_max_chi_sq'], hl.agg.mean(mt['_y'])) - 1.0) / 
                hl.agg.filter(mt['_y'] < mt['_max_chi_sq'], hl.agg.mean(mt['_n'] * mt['_x_total']))))
        mt = mt.annotate_cols(
            _step1_h2=hl.max(hl.min(mt['_step1_h2'], 1.0), 0.0))
        mt = mt.annotate_entries(
            _w_step1=hl.cond(
                mt['_in_step1'],
                1.0 / (mt['_w_initial_floor'] * 2.0 * (1.0 + (mt['_n'] * mt['_step1_h2'] * mt['_x_total'] / mt['_m_total']))**2),
                0.0))
        mt = mt.annotate_cols(
            _step1_betas=hl.agg.filter(
                mt['_in_step1'],
                hl.agg.linreg(
                    y=mt['_y'],
                    x=[1.0] + [mt['_n'] * mt['_x'][i] / mt['_n_mean'] for i in range(k)],
                    weight=mt['_w_step1']).beta))

    else:
        mt = mt.annotate_cols(
            _step1_betas=hl.array([1.0]).extend(
                hl.agg.array_agg(
                    lambda x: (hl.agg.mean(mt['_y']) - 1.0) / hl.agg.mean(x), mt['_x'])))

        for iteration in range(n_iterations):
            mt = mt.annotate_entries(
                _w_step1=hl.cond(
                    mt['_in_step1'],
                    1.0 / (mt['_w_initial_floor'] * 2.0 * (
                        mt['_step1_betas'][0] + hl.sum(
                            hl.map(
                                lambda i: mt['_step1_betas'][i+1] * mt['_x_floor'][i],
                                hl.range(0, k))))**2),
                    0.0))
            mt = mt.annotate_cols(
                _step1_betas=hl.agg.filter(
                    mt['_in_step1'],
                    hl.agg.linreg(
                        y=mt['_y'],
                        x=[1.0] + [mt['_x'][i] for i in range(k)],
                        weight=mt['_w_step1']).beta))
            if (iteration + 1) != n_iterations:
                mt = mt.annotate_cols(
                    _step1_h2=hl.map(
                        lambda i: hl.max(hl.min(mt['_step1_betas'][i+1] * mt['_m'][i] / mt['_n_mean'], 1.0), 0.0),
                        hl.range(0, k)))
                mt = mt.annotate_cols(
                    _step1_betas=hl.array([
                        mt['_step1_betas'][0]]).extend(
                            hl.map(
                                lambda i: mt['_step1_h2'][i] * mt['_n_mean'] / mt['_m'][i],
                                hl.range(0, k))))

    mt = mt.annotate_cols(
        _step1_block_betas=_block_betas(
            block_expr=mt['_step1_block'],
            include_expr=mt['_in_step1'],
            y_expr=mt['_y'],
            covariates=[1.0] + [mt['_x'][i] for i in range(k)],
            w_expr=mt['_w_step1'],
            n_blocks=n_blocks))

    mt = mt.annotate_cols(
        _step1_jackknife_variances=hl.map(
            lambda i: hl.rbind(
                _pseudovalues(
                    estimate=mt['_step1_betas'][i],
                    block_estimates=hl.map(
                        lambda j: mt['_step1_block_betas'][j][i],
                        hl.range(0, n_blocks))),
                lambda pseudovalues: _variance(pseudovalues)),
            hl.range(0, k + 1)))

    if two_step_threshold:
        mt = mt.annotate_cols(
            _initial_betas=hl.array([1.0]).extend(
                hl.agg.array_agg(
                    lambda x: (hl.agg.mean(mt['_y']) - 1.0) / hl.agg.mean(x), mt['_x'])))
        mt = mt.annotate_cols(_step2_betas=mt['_initial_betas'])

        for iteration in range(n_iterations):
            mt = mt.annotate_entries(
                _w_step2=hl.cond(
                    mt['_in_step2'],
                    1.0 / (mt['_w_initial_floor'] * 2.0 * (
                        mt['_step2_betas'][0] + hl.sum(
                            hl.map(
                                lambda i: mt['_step2_betas'][i+1] * mt['_x_floor'][i],
                                hl.range(0, k))))**2),
                    0.0))
            mt = mt.annotate_cols(
                _step2_betas=hl.array([
                    mt['_step1_betas'][0]]).extend(
                        hl.agg.filter(
                            mt['_in_step2'],
                            hl.agg.linreg(
                                y=mt['_y'] - mt['_step1_betas'][0],
                                x=[mt['_x'][i] for i in range(k)],
                                weight=mt['_w_step2']).beta)))
            if (iteration + 1) != n_iterations:
                mt = mt.annotate_cols(
                    _step2_h2=hl.map(
                        lambda i: hl.max(hl.min(mt['_step2_betas'][i+1] * mt['_m'][i] / mt['_n_mean'], 1.0), 0.0),
                        hl.range(0, k)))
                mt = mt.annotate_cols(
                    _step2_betas=hl.array([
                        mt['_step1_betas'][0]]).extend(
                            hl.map(
                                lambda i: mt['_step2_h2'][i] * mt['_n_mean'] / mt['_m'][i],
                                hl.range(0, k))))

        mt = mt.annotate_cols(
            _step2_block_betas=_block_betas(
                block_expr=mt['_step2_block'],
                include_expr=mt['_in_step2'],
                y_expr=mt['_y'] - mt['_step1_betas'][0],
                covariates=[mt['_x'][i] for i in range(k)],
                w_expr=mt['_w_step2'],
                n_blocks=n_blocks))

        mt = mt.annotate_cols(
            _step2_jackknife_variances=hl.map(
                lambda i: hl.rbind(
                    _pseudovalues(
                        estimate=mt['_step2_betas'][i+1],
                        block_estimates=hl.map(
                            lambda j: mt['_step2_block_betas'][j][i],
                            hl.range(0, n_blocks))),
                    lambda pseudovalues: _variance(pseudovalues)),
                hl.range(0, k)))

        mt = mt.annotate_entries(
            _initial_w=1.0 / (mt['_w_initial_floor'] * 2.0 * (
                mt['_initial_betas'][0] + hl.sum(
                    hl.map(
                        lambda i: mt['_initial_betas'][i+1] * mt['_x_floor'][i],
                        hl.range(0, k))))**2))

        mt = mt.annotate_cols(
            _c=hl.agg.array_agg(
                lambda x: hl.agg.sum(mt['_initial_w'] * x) / hl.agg.sum(mt['_initial_w'] * x**2), mt['_x']))

        mt = mt.annotate_cols(
            _final_block_betas=hl.map(
                lambda i: hl.array([mt['_step1_block_betas'][i][0]]).extend(
                    hl.map(
                        lambda j: mt['_step2_block_betas'][i][j] - mt['_c'][j] * (mt['_step1_block_betas'][i][0] - mt['_step1_betas'][0]),
                        hl.range(0, k))),
                hl.range(0, n_blocks)))

        mt = mt.annotate_cols(
            _final_betas=hl.array([
                mt['_step1_betas'][0]]).extend(mt['_step2_betas'][1:]),
            _final_jackknife_variances=hl.array([
                mt['_step1_jackknife_variances'][0]]).extend(
                    hl.map(
                        lambda i: hl.rbind(
                            _pseudovalues(
                                mt['_step2_betas'][i+1],
                                hl.map(
                                    lambda j: mt['_final_block_betas'][j][i+1],
                                    hl.range(0, n_blocks))),
                            lambda pseudovalues: _variance(pseudovalues)),
                        hl.range(0, k))))

    else:
        mt = mt.annotate_cols(
            _final_betas=mt['_step1_betas'],
            _final_jackknife_variances=mt['_step1_jackknife_variances'],
            _final_block_betas=mt['_step1_block_betas'])

    mt = mt.annotate_cols(
        trait=mt['_y_name'],
        n_samples=mt['_n_mean'],
        n_reference_panel_variants=mt['_m'],
        mean_chi_sq=hl.agg.mean(mt['_y']),
        intercept=hl.struct(
            estimate=mt['_final_betas'][0],
            standard_error=hl.sqrt(
                mt['_final_jackknife_variances'][0])),
        snp_heritability=hl.struct(
            estimate=hl.sum(
                hl.map(
                    lambda i: mt['_final_betas'][i+1] * mt['_m'][i] / mt['_n_mean'],
                    hl.range(0, k))),
            standard_error=hl.sqrt(
                hl.sum(
                    hl.map(
                        lambda i: (mt['_m'][i] / mt['_n_mean'])**2 * mt['_final_jackknife_variances'][i+1],
                        hl.range(0, k))))))

    mt = mt.annotate_cols(
        ratio=hl.struct(
            estimate=hl.cond(
                mt['mean_chi_sq'] > 1.0,
                (mt['intercept']['estimate'] - 1.0) / (mt['mean_chi_sq'] - 1.0),
                hl.null(hl.tfloat)),
            standard_error=hl.cond(
                mt['mean_chi_sq'] > 1.0,
                mt['intercept']['standard_error'] / (mt['mean_chi_sq'] - 1.0),
                hl.null(hl.tfloat))))

    ht = mt.cols()
    ht = ht.key_by('trait')
    ht = ht.select('n_samples',
                   'n_reference_panel_variants',
                   'mean_chi_sq',
                   'intercept',
                   'snp_heritability',
                   'ratio')

    ht_tmp_file = new_temp_file()
    ht.write(ht_tmp_file)
    ht = hl.read_table(ht_tmp_file)

    return ht

