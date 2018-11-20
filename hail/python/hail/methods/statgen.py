import itertools
import math
import numpy as np
from typing import *

import hail as hl
import hail.expr.aggregators as agg
from hail.expr.expressions import *
from hail.expr.types import *
from hail.ir import *
from hail.genetics.reference_genome import reference_genome_type
from hail.linalg import BlockMatrix
from hail.matrixtable import MatrixTable
from hail.methods.misc import require_biallelic, require_row_key_variant
from hail.stats import LinearMixedModel
from hail.table import Table
from hail.typecheck import *
from hail.utils import wrap_to_list, new_temp_file
from hail.utils.java import *


@typecheck(dataset=MatrixTable,
           maf=nullable(expr_float64),
           bounded=bool,
           min=nullable(numeric),
           max=nullable(numeric))
def identity_by_descent(dataset, maf=None, bounded=True, min=None, max=None) -> Table:
    """Compute matrix of identity-by-descent estimates.

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    To calculate a full IBD matrix, using minor allele frequencies computed
    from the dataset itself:

    >>> hl.identity_by_descent(dataset)

    To calculate an IBD matrix containing only pairs of samples with
    ``PI_HAT`` in :math:`[0.2, 0.9]`, using minor allele frequencies stored in
    the row field `panel_maf`:

    >>> hl.identity_by_descent(dataset, maf=dataset['panel_maf'], min=0.2, max=0.9)

    Notes
    -----

    The implementation is based on the IBD algorithm described in the `PLINK
    paper <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1950838>`__.

    :func:`.identity_by_descent` requires the dataset to be biallelic and does
    not perform LD pruning. Linkage disequilibrium may bias the result so
    consider filtering variants first.

    The resulting :class:`.Table` entries have the type: *{ i: String,
    j: String, ibd: { Z0: Double, Z1: Double, Z2: Double, PI_HAT: Double },
    ibs0: Long, ibs1: Long, ibs2: Long }*. The key list is: `*i: String, j:
    String*`.

    Conceptually, the output is a symmetric, sample-by-sample matrix. The
    output table has the following form

    .. code-block:: text

        i		j	ibd.Z0	ibd.Z1	ibd.Z2	ibd.PI_HAT ibs0	ibs1	ibs2
        sample1	sample2	1.0000	0.0000	0.0000	0.0000 ...
        sample1	sample3	1.0000	0.0000	0.0000	0.0000 ...
        sample1	sample4	0.6807	0.0000	0.3193	0.3193 ...
        sample1	sample5	0.1966	0.0000	0.8034	0.8034 ...

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Variant-keyed :class:`.MatrixTable` containing genotype information.
    maf : :class:`.Float64Expression`, optional
        Row-indexed expression for the minor allele frequency.
    bounded : :obj:`bool`
        Forces the estimations for `Z0``, ``Z1``, ``Z2``, and ``PI_HAT`` to take
        on biologically meaningful values (in the range [0,1]).
    min : :obj:`float` or :obj:`None`
        Sample pairs with a ``PI_HAT`` below this value will
        not be included in the output. Must be in :math:`[0,1]`.
    max : :obj:`float` or :obj:`None`
        Sample pairs with a ``PI_HAT`` above this value will
        not be included in the output. Must be in :math:`[0,1]`.

    Returns
    -------
    :class:`.Table`
    """

    if maf is not None:
        analyze('identity_by_descent/maf', maf, dataset._row_indices)
        dataset = dataset.select_rows(__maf = maf)
    else:
        dataset = dataset.select_rows()
    dataset = dataset.select_cols().select_globals().select_entries('GT')
    return Table._from_java(Env.hail().methods.IBD.apply(require_biallelic(dataset, 'ibd')._jmt,
                                                         joption('__maf' if maf is not None else None),
                                                         bounded,
                                                         joption(min),
                                                         joption(max)))


@typecheck(call=expr_call,
           aaf_threshold=numeric,
           include_par=bool,
           female_threshold=numeric,
           male_threshold=numeric,
           aaf=nullable(str))
def impute_sex(call, aaf_threshold=0.0, include_par=False, female_threshold=0.2, male_threshold=0.8, aaf=None) -> Table:
    """Impute sex of samples by calculating inbreeding coefficient on the X
    chromosome.

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    Remove samples where imputed sex does not equal reported sex:

    >>> imputed_sex = hl.impute_sex(dataset.GT)
    >>> dataset_result = dataset.filter_cols(imputed_sex[dataset.s].is_female != dataset.pheno.is_female)

    Notes
    -----

    We have used the same implementation as `PLINK v1.7
    <http://pngu.mgh.harvard.edu/~purcell/plink/summary.shtml#sexcheck>`__.

    Let `gr` be the the reference genome of the type of the `locus` key (as
    given by :meth:`.TLocus.reference_genome`)

    1. Filter the dataset to loci on the X contig defined by `gr`.

    2. Calculate alternate allele frequency (AAF) for each row from the dataset.

    3. Filter to variants with AAF above `aaf_threshold`.

    4. Remove loci in the pseudoautosomal region, as defined by `gr`, if and
       only if `include_par` is ``True`` (it defaults to ``False``)

    5. For each row and column with a non-missing genotype call, :math:`E`, the
       expected number of homozygotes (from population AAF), is computed as
       :math:`1.0 - (2.0*maf*(1.0-maf))`.

    6. For each row and column with a non-missing genotype call, :math:`O`, the
       observed number of homozygotes, is computed interpreting ``0`` as
       heterozygote and ``1`` as homozygote`

    7. For each row and column with a non-missing genotype call, :math:`N` is
       incremented by 1

    8. For each column, :math:`E`, :math:`O`, and :math:`N` are combined across
       variants

    9. For each column, :math:`F` is calculated by :math:`(O - E) / (N - E)`

    10. A sex is assigned to each sample with the following criteria:
        - Female when ``F < 0.2``
        - Male when ``F > 0.8``
        Use `female_threshold` and `male_threshold` to change this behavior.

    **Annotations**

    The returned column-key indexed :class:`.Table` has the following fields in
    addition to the matrix table's column keys:

    - **is_female** (:py:data:`.tbool`) -- True if the imputed sex is female,
      false if male, missing if undetermined.
    - **f_stat** (:py:data:`.tfloat64`) -- Inbreeding coefficient.
    - **n_called**  (:py:data:`.tint64`) -- Number of variants with a genotype call.
    - **expected_homs** (:py:data:`.tfloat64`) -- Expected number of homozygotes.
    - **observed_homs** (:py:data:`.tint64`) -- Observed number of homozygotes.

    call : :class:`.CallExpression`
        A genotype call for each row and column. The source dataset's row keys
        must be [[locus], alleles] with types :class:`.tlocus` and
        :class:`.ArrayStringExpression`. Moreover, the alleles array must have
        exactly two elements (i.e. the variant must be biallelic).
    aaf_threshold : :obj:`float`
        Minimum alternate allele frequency threshold.
    include_par : :obj:`bool`
        Include pseudoautosomal regions.
    female_threshold : :obj:`float`
        Samples are called females if F < female_threshold.
    male_threshold : :obj:`float`
        Samples are called males if F > male_threshold.
    aaf : :obj:`str` or :obj:`None`
        A field defining the alternate allele frequency for each row. If
        ``None``, AAF will be computed from `call`.

    Return
    ------
    :class:`.Table`
        Sex imputation statistics per sample.
    """
    if aaf_threshold < 0.0 or aaf_threshold > 1.0:
        raise FatalError("Invalid argument for `aaf_threshold`. Must be in range [0, 1].")

    mt = call._indices.source
    mt, _ = mt._process_joins(call)
    mt = mt.annotate_entries(call=call)
    mt = require_biallelic(mt, 'impute_sex')
    if (aaf is None):
        mt = mt.annotate_rows(aaf=agg.call_stats(mt.call, mt.alleles).AF[1])
        aaf = 'aaf'

    rg = mt.locus.dtype.reference_genome
    mt = hl.filter_intervals(mt,
                             hl.map(lambda x_contig: hl.parse_locus_interval(x_contig, rg), rg.x_contigs),
                             keep=True)
    if not include_par:
        interval_type = hl.tarray(hl.tinterval(hl.tlocus(rg)))
        mt = hl.filter_intervals(mt,
                                 hl.literal(rg.par, interval_type),
                                 keep=False)

    mt = mt.filter_rows((mt[aaf] > aaf_threshold) & (mt[aaf] < (1 - aaf_threshold)))
    mt = mt.annotate_cols(ib=agg.inbreeding(mt.call, mt[aaf]))
    kt = mt.select_cols(
        is_female=hl.cond(mt.ib.f_stat < female_threshold,
                          True,
                          hl.cond(mt.ib.f_stat > male_threshold,
                                  False,
                                  hl.null(tbool))),
        **mt.ib).cols()

    return kt


def _get_regression_row_fields(mt, pass_through, method) -> Dict[str, str]:

    # include key as base
    row_fields = dict(mt.row_key)
    for f in pass_through:
        if isinstance(f, str):
            if f not in mt.row:
                raise ValueError(f"'{method}/pass_through': MatrixTable has no row field {repr(f)}")
            if f in row_fields:
                # allow silent pass through of key fields
                if f in mt.row_key:
                    pass
                else:
                    raise ValueError(f"'{method}/pass_through': found duplicated field {repr(f)}")
            row_fields[f] = mt[f]
        else:
            assert isinstance(f, Expression)
            if not f._ir.is_nested_field:
                raise ValueError(f"'{method}/pass_through': expect fields or nested fields, not complex expressions")
            if not f._indices == mt._row_indices:
                raise ExpressionException(f"'{method}/pass_through': require row-indexed fields, found indices {f._indices.axes}")
            name = f._ir.name
            if name in row_fields:
                # allow silent pass through of key fields
                if not (name in mt.row_key and f._ir == mt[name]._ir):
                    raise ValueError(f"'{method}/pass_through': found duplicated field {repr(name)}")
            row_fields[name] = f
    return row_fields


@typecheck(y=oneof(expr_float64, sequenceof(expr_float64), sequenceof(sequenceof(expr_float64))),
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           block_size=int,
           pass_through=sequenceof(oneof(str, Expression)))
def linear_regression_rows(y, x, covariates, block_size=16, pass_through=()) -> hail.Table:
    r"""For each row, test an input variable for association with
    response variables using linear regression.

    Examples
    --------

    >>> result_ht = hl.linear_regression_rows(
    ...     y=dataset.pheno.height,
    ...     x=dataset.GT.n_alt_alleles(),
    ...     covariates=[1, dataset.pheno.age, dataset.pheno.is_female])

    Warning
    -------
    As in the example, the intercept covariate ``1`` must be
    included **explicitly** if desired.

    Warning
    -------
    If `y` is a single value or a list, :func:`.linear_regression_rows`
    considers the same set of columns (i.e., samples, points) for every response
    variable and row, namely those columns for which **all** response variables
    and covariates are defined.

    If `y` is a list of lists, then each inner list is treated as an
    independent group, subsetting columns for missingness separately.

    Notes
    -----
    With the default root and `y` a single expression, the following row-indexed
    fields are added.

    - **<row key fields>** (Any) -- Row key fields.
    - **<pass_through fields>** (Any) -- Row fields in `pass_through`.
    - **n** (:py:data:`.tint32`) -- Number of columns used.
    - **sum_x** (:py:data:`.tfloat64`) -- Sum of input values `x`.
    - **y_transpose_x** (:py:data:`.tfloat64`) -- Dot product of response
      vector `y` with the input vector `x`.
    - **beta** (:py:data:`.tfloat64`) --
      Fit effect coefficient of `x`, :math:`\hat\beta_1` below.
    - **standard_error** (:py:data:`.tfloat64`) --
      Estimated standard error, :math:`\widehat{\mathrm{se}}_1`.
    - **t_stat** (:py:data:`.tfloat64`) -- :math:`t`-statistic, equal to
      :math:`\hat\beta_1 / \widehat{\mathrm{se}}_1`.
    - **p_value** (:py:data:`.tfloat64`) -- :math:`p`-value.

    If `y` is a list of expressions, then the last five fields instead have type
    :py:data:`.tarray` of :py:data:`.tfloat64`, with corresponding indexing of
    the list and each array.

    If `y` is a list of lists of expressions, then `n` and `sum_x` are of type
    ``array<float64>``, and the last five fields are of type
    ``array<array<float64>>``. Index into these arrays with
    ``a[index_in_outer_list, index_in_inner_list]``. For example, if
    ``y=[[a], [b, c]]`` then the p-value for ``b`` is ``p_value[1][0]``.


    In the statistical genetics example above, the input variable `x` encodes
    genotype as the number of alternate alleles (0, 1, or 2). For each variant
    (row), genotype is tested for association with height controlling for age
    and sex, by fitting the linear regression model:

    .. math::

        \mathrm{height} = \beta_0 + \beta_1 \, \mathrm{genotype}
            + \beta_2 \, \mathrm{age}
            + \beta_3 \, \mathrm{is\_female}
            + \varepsilon,
            \quad
            \varepsilon \sim \mathrm{N}(0, \sigma^2)

    Boolean covariates like :math:`\mathrm{is\_female}` are encoded as 1 for
    ``True`` and 0 for ``False``. The null model sets :math:`\beta_1 = 0`.

    The standard least-squares linear regression model is derived in Section
    3.2 of `The Elements of Statistical Learning, 2nd Edition
    <http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf>`__.
    See equation 3.12 for the t-statistic which follows the t-distribution with
    :math:`n - k - 1` degrees of freedom, under the null hypothesis of no
    effect, with :math:`n` samples and :math:`k` covariates in addition to
    ``x``.

    Note
    ----
    Use the `pass_through` parameter to include additional row fields from
    matrix table underlying ``x``. For example, to include an "rsid" field, set
    ``pass_through=['rsid']`` or ``pass_through=[mt.rsid]``.

    Parameters
    ----------
    y : :class:`.Float64Expression` or :obj:`list` of :class:`.Float64Expression`
        One or more column-indexed response expressions.
    x : :class:`.Float64Expression`
        Entry-indexed expression for input variable.
    covariates : :obj:`list` of :class:`.Float64Expression`
        List of column-indexed covariate expressions.
    block_size : :obj:`int`
        Number of row regressions to perform simultaneously per core. Larger blocks
        require more memory but may improve performance.
    pass_through : :obj:`list` of :obj:`str` or :class:`.Expression`
        Additional row fields to include in the resulting table.

    Returns
    -------
    :class:`.Table`
    """
    mt = matrix_table_source('linear_regression_rows/x', x)
    check_entry_indexed('linear_regression_rows/x', x)

    y_is_list = isinstance(y, list)
    if y_is_list and len(y) == 0:
        raise ValueError(f"'linear_regression_rows': found no values for 'y'")
    is_chained = y_is_list and isinstance(y[0], list)
    if is_chained and any(len(l) == 0 for l in y):
        raise ValueError(f"'linear_regression_rows': found empty inner list for 'y'")

    y = wrap_to_list(y)

    for e in (itertools.chain.from_iterable(y) if is_chained else y):
        analyze('linear_regression_rows/y', e, mt._col_indices)

    for e in covariates:
        analyze('linear_regression_rows/covariates', e, mt._col_indices)

    _warn_if_no_intercept('linear_regression_rows', covariates)

    x_field_name = Env.get_uid()
    if is_chained:
        y_field_names = [[f'__y_{i}_{j}' for j in range(len(y[i]))] for i in range(len(y))]
        y_dict = dict(zip(itertools.chain.from_iterable(y_field_names), itertools.chain.from_iterable(y)))
        func = Env.hail().methods.LinearRegression.chain

    else:
        y_field_names = list(f'__y_{i}' for i in range(len(y)))
        y_dict = dict(zip(y_field_names, y))
        func = Env.hail().methods.LinearRegression.single_group

    cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))

    row_fields = _get_regression_row_fields(mt, pass_through, 'linear_regression_rows')

    # FIXME: selecting an existing entry field should be emitted as a SelectFields
    mt = mt._select_all(col_exprs=dict(**y_dict,
                                       **dict(zip(cov_field_names, covariates))),
                        row_exprs=row_fields,
                        col_key=[],
                        entry_exprs={x_field_name: x})

    jt = func(
        mt._jmt,
        y_field_names,
        x_field_name,
        cov_field_names,
        block_size,
        list(row_fields)[len(mt.row_key):])

    ht_result = Table._from_java(jt)

    if not y_is_list:
        fields = ['y_transpose_x', 'beta', 'standard_error', 't_stat', 'p_value']
        ht_result = ht_result.annotate(**{f: ht_result[f][0] for f in fields})

    return ht_result


@typecheck(test=enumeration('wald', 'lrt', 'score', 'firth'),
           y=expr_float64,
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           pass_through=sequenceof(oneof(str, Expression)))
def logistic_regression_rows(test, y, x, covariates, pass_through=()) -> hail.Table:
    r"""For each row, test an input variable for association with a
    binary response variable using logistic regression.

    Examples
    --------
    Run the logistic regression Wald test per variant using a Boolean
    phenotype, intercept and two covariates stored in column-indexed
    fields:

    >>> result_ht = hl.logistic_regression_rows(
    ...     test='wald',
    ...     y=dataset.pheno.is_case,
    ...     x=dataset.GT.n_alt_alleles(),
    ...     covariates=[1, dataset.pheno.age, dataset.pheno.is_female])

    Warning
    -------
    :func:`.logistic_regression_rows` considers the same set of
    columns (i.e., samples, points) for every row, namely those columns for
    which **all** covariates are defined. For each row, missing values of
    `x` are mean-imputed over these columns. As in the example, the
    intercept covariate ``1`` must be included **explicitly** if desired.

    Notes
    -----
    This method performs, for each row, a significance test of the input
    variable in predicting a binary (case-control) response variable based
    on the logistic regression model. The response variable type must either
    be numeric (with all present values 0 or 1) or Boolean, in which case
    true and false are coded as 1 and 0, respectively.

    Hail supports the Wald test ('wald'), likelihood ratio test ('lrt'),
    Rao score test ('score'), and Firth test ('firth'). Hail only includes
    columns for which the response variable and all covariates are defined.
    For each row, Hail imputes missing input values as the mean of the
    non-missing values.

    The example above considers a model of the form

    .. math::

        \mathrm{Prob}(\mathrm{is_case}) =
            \mathrm{sigmoid}(\beta_0 + \beta_1 \, \mathrm{gt}
                            + \beta_2 \, \mathrm{age}
                            + \beta_3 \, \mathrm{is\_female} + \varepsilon),
        \quad
        \varepsilon \sim \mathrm{N}(0, \sigma^2)

    where :math:`\mathrm{sigmoid}` is the `sigmoid function`_, the genotype
    :math:`\mathrm{gt}` is coded as 0 for HomRef, 1 for Het, and 2 for
    HomVar, and the Boolean covariate :math:`\mathrm{is\_female}` is coded as
    for ``True`` (female) and 0 for ``False`` (male). The null model sets
    :math:`\beta_1 = 0`.

    .. _sigmoid function: https://en.wikipedia.org/wiki/Sigmoid_function

    The structure of the emitted row field depends on the test statistic as
    shown in the tables below.

    ========== ================== ======= ============================================
    Test       Field              Type    Value
    ========== ================== ======= ============================================
    Wald       `beta`             float64 fit effect coefficient,
                                          :math:`\hat\beta_1`
    Wald       `standard_error`   float64 estimated standard error,
                                          :math:`\widehat{\mathrm{se}}`
    Wald       `z_stat`           float64 Wald :math:`z`-statistic, equal to
                                          :math:`\hat\beta_1 / \widehat{\mathrm{se}}`
    Wald       `p_value`          float64 Wald p-value testing :math:`\beta_1 = 0`
    LRT, Firth `beta`             float64 fit effect coefficient,
                                          :math:`\hat\beta_1`
    LRT, Firth `chi_sq_stat`      float64 deviance statistic
    LRT, Firth `p_value`          float64 LRT / Firth p-value testing
                                          :math:`\beta_1 = 0`
    Score      `chi_sq_stat`      float64 score statistic
    Score      `p_value`          float64 score p-value testing :math:`\beta_1 = 0`
    ========== ================== ======= ============================================

    For the Wald and likelihood ratio tests, Hail fits the logistic model for
    each row using Newton iteration and only emits the above fields
    when the maximum likelihood estimate of the coefficients converges. The
    Firth test uses a modified form of Newton iteration. To help diagnose
    convergence issues, Hail also emits three fields which summarize the
    iterative fitting process:

    ================ =================== ======= ===============================
    Test             Field               Type    Value
    ================ =================== ======= ===============================
    Wald, LRT, Firth `fit.n_iterations`  int32   number of iterations until
                                                 convergence, explosion, or
                                                 reaching the max (25 for
                                                 Wald, LRT; 100 for Firth)
    Wald, LRT, Firth `fit.converged`      bool    ``True`` if iteration converged
    Wald, LRT, Firth `fit.exploded`       bool    ``True`` if iteration exploded
    ================ =================== ======= ===============================

    We consider iteration to have converged when every coordinate of
    :math:`\beta` changes by less than :math:`10^{-6}`. For Wald and LRT,
    up to 25 iterations are attempted; in testing we find 4 or 5 iterations
    nearly always suffice. Convergence may also fail due to explosion,
    which refers to low-level numerical linear algebra exceptions caused by
    manipulating ill-conditioned matrices. Explosion may result from (nearly)
    linearly dependent covariates or complete separation_.

    .. _separation: https://en.wikipedia.org/wiki/Separation_(statistics)

    A more common situation in genetics is quasi-complete seperation, e.g.
    variants that are observed only in cases (or controls). Such variants
    inevitably arise when testing millions of variants with very low minor
    allele count. The maximum likelihood estimate of :math:`\beta` under
    logistic regression is then undefined but convergence may still occur
    after a large number of iterations due to a very flat likelihood
    surface. In testing, we find that such variants produce a secondary bump
    from 10 to 15 iterations in the histogram of number of iterations per
    variant. We also find that this faux convergence produces large standard
    errors and large (insignificant) p-values. To not miss such variants,
    consider using Firth logistic regression, linear regression, or
    group-based tests.

    Here's a concrete illustration of quasi-complete seperation in R. Suppose
    we have 2010 samples distributed as follows for a particular variant:

    ======= ====== === ======
    Status  HomRef Het HomVar
    ======= ====== === ======
    Case    1000   10  0
    Control 1000   0   0
    ======= ====== === ======

    The following R code fits the (standard) logistic, Firth logistic,
    and linear regression models to this data, where ``x`` is genotype,
    ``y`` is phenotype, and ``logistf`` is from the logistf package:

    .. code-block:: R

        x <- c(rep(0,1000), rep(1,1000), rep(1,10)
        y <- c(rep(0,1000), rep(0,1000), rep(1,10))
        logfit <- glm(y ~ x, family=binomial())
        firthfit <- logistf(y ~ x)
        linfit <- lm(y ~ x)

    The resulting p-values for the genotype coefficient are 0.991, 0.00085,
    and 0.0016, respectively. The erroneous value 0.991 is due to
    quasi-complete separation. Moving one of the 10 hets from case to control
    eliminates this quasi-complete separation; the p-values from R are then
    0.0373, 0.0111, and 0.0116, respectively, as expected for a less
    significant association.

    The Firth test reduces bias from small counts and resolves the issue of
    separation by penalizing maximum likelihood estimation by the `Jeffrey's
    invariant prior <https://en.wikipedia.org/wiki/Jeffreys_prior>`__. This
    test is slower, as both the null and full model must be fit per variant,
    and convergence of the modified Newton method is linear rather than
    quadratic. For Firth, 100 iterations are attempted for the null model
    and, if that is successful, for the full model as well. In testing we
    find 20 iterations nearly always suffices. If the null model fails to
    converge, then the `logreg.fit` fields reflect the null model;
    otherwise, they reflect the full model.

    See
    `Recommended joint and meta-analysis strategies for case-control association testing of single low-count variants <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4049324/>`__
    for an empirical comparison of the logistic Wald, LRT, score, and Firth
    tests. The theoretical foundations of the Wald, likelihood ratio, and score
    tests may be found in Chapter 3 of Gesine Reinert's notes
    `Statistical Theory <http://www.stats.ox.ac.uk/~reinert/stattheory/theoryshort09.pdf>`__.
    Firth introduced his approach in
    `Bias reduction of maximum likelihood estimates, 1993 <http://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/GibbsFieldEst/BiasReductionMLE.pdf>`__.
    Heinze and Schemper further analyze Firth's approach in
    `A solution to the problem of separation in logistic regression, 2002 <https://cemsiis.meduniwien.ac.at/fileadmin/msi_akim/CeMSIIS/KB/volltexte/Heinze_Schemper_2002_Statistics_in_Medicine.pdf>`__.

    Hail's logistic regression tests correspond to the ``b.wald``,
    ``b.lrt``, and ``b.score`` tests in `EPACTS`_. For each variant, Hail
    imputes missing input values as the mean of non-missing input values,
    whereas EPACTS subsets to those samples with called genotypes. Hence,
    Hail and EPACTS results will currently only agree for variants with no
    missing genotypes.

    .. _EPACTS: http://genome.sph.umich.edu/wiki/EPACTS#Single_Variant_Tests

    Note
    ----
    Use the `pass_through` parameter to include additional row fields from
    matrix table underlying ``x``. For example, to include an "rsid" field, set
    ``pass_through=['rsid']`` or ``pass_through=[mt.rsid]``.

    Parameters
    ----------
    test : {'wald', 'lrt', 'score', 'firth'}
        Statistical test.
    y : :class:`.Float64Expression`
        Column-indexed response expression.
        All non-missing values must evaluate to 0 or 1.
        Note that a :class:`.BooleanExpression` will be implicitly converted to
        a :class:`.Float64Expression` with this property.
    x : :class:`.Float64Expression`
        Entry-indexed expression for input variable.
    covariates : :obj:`list` of :class:`.Float64Expression`
        Non-empty list of column-indexed covariate expressions.
    pass_through : :obj:`list` of :obj:`str` or :class:`.Expression`
        Additional row fields to include in the resulting table.

    Returns
    -------
    :class:`.Table`
    """
    if len(covariates) == 0:
        raise ValueError('logistic regression requires at least one covariate expression')

    mt = matrix_table_source('logistic_regresion_rows/x', x)
    check_entry_indexed('logistic_regresion_rows/x', x)

    analyze('logistic_regresion_rows/y', y, mt._col_indices)

    all_exprs = [y]
    for e in covariates:
        all_exprs.append(e)
        analyze('logistic_regression/covariates', e, mt._col_indices)

    _warn_if_no_intercept('logistic_regresion_rows', covariates)

    x_field_name = Env.get_uid()
    y_field_name = '__y'
    cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))
    row_fields = _get_regression_row_fields(mt, pass_through, 'logistic_regression_rows')

# FIXME: selecting an existing entry field should be emitted as a SelectFields
    mt = mt._select_all(col_exprs=dict(**{y_field_name: y},
                                       **dict(zip(cov_field_names, covariates))),
                        row_exprs=row_fields,
                        col_key=[],
                        entry_exprs={x_field_name: x})

    jt = Env.hail().methods.LogisticRegression.apply(
        mt._jmt,
        test,
        y_field_name,
        x_field_name,
        cov_field_names,
        list(row_fields)[len(mt.row_key):])
    return Table._from_java(jt)


@typecheck(test=enumeration('wald', 'lrt', 'score'),
           y=expr_float64,
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           pass_through=sequenceof(oneof(str, Expression)))
def poisson_regression_rows(test, y, x, covariates, pass_through=()) -> Table:
    r"""For each row, test an input variable for association with a
    count response variable using `Poisson regression <https://en.wikipedia.org/wiki/Poisson_regression>`__.

    Notes
    -----
    See :func:`.logistic_regression_rows` for more info on statistical tests
    of general linear models.

    Note
    ----
    Use the `pass_through` parameter to include additional row fields from
    matrix table underlying ``x``. For example, to include an "rsid" field, set
    ``pass_through=['rsid']`` or ``pass_through=[mt.rsid]``.

    Parameters
    ----------
    y : :class:`.Float64Expression`
        Column-indexed response expression.
        All non-missing values must evaluate to a non-negative integer.
    x : :class:`.Float64Expression`
        Entry-indexed expression for input variable.
    covariates : :obj:`list` of :class:`.Float64Expression`
        Non-empty list of column-indexed covariate expressions.
    pass_through : :obj:`list` of :obj:`str` or :class:`.Expression`
        Additional row fields to include in the resulting table.

    Returns
    -------
    :class:`.Table`
    """
    if len(covariates) == 0:
        raise ValueError('Poisson regression requires at least one covariate expression')

    mt = matrix_table_source('poisson_regression_rows/x', x)
    check_entry_indexed('poisson_regression_rows/x', x)

    analyze('poisson_regression_rows/y', y, mt._col_indices)

    all_exprs = [y]
    for e in covariates:
        all_exprs.append(e)
        analyze('poisson_regression_rows/covariates', e, mt._col_indices)

    _warn_if_no_intercept('poisson_regression_rows', covariates)

    x_field_name = Env.get_uid()
    y_field_name = '__y'
    cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))
    row_fields = _get_regression_row_fields(mt, pass_through, 'poisson_regression_rows')

    # FIXME: selecting an existing entry field should be emitted as a SelectFields
    mt = mt._select_all(col_exprs=dict(**{y_field_name: y},
                                       **dict(zip(cov_field_names, covariates))),
                        row_exprs=row_fields,
                        col_key=[],
                        entry_exprs={x_field_name: x})

    jt = Env.hail().methods.PoissonRegression.apply(
        mt._jmt,
        test,
        y_field_name,
        x_field_name,
        cov_field_names,
        list(row_fields)[len(mt.row_key):])
    return Table._from_java(jt)


@typecheck(y=expr_float64,
           x=sequenceof(expr_float64),
           z_t=nullable(expr_float64),
           k=nullable(np.ndarray),
           p_path=nullable(str),
           overwrite=bool,
           standardize=bool,
           mean_impute=bool)
def linear_mixed_model(y,
                       x,
                       z_t=None,
                       k=None,
                       p_path=None,
                       overwrite=False,
                       standardize=True,
                       mean_impute=True):
    r"""Initialize a linear mixed model from a matrix table.

    Examples
    --------
    Initialize a model using three fixed effects (including intercept) and
    genetic marker random effects:

    >>> marker_ds = dataset.filter_rows(dataset.use_as_marker)
    >>> model, _ = hl.linear_mixed_model(
    ...     y=marker_ds.pheno.height,
    ...     x=[1, marker_ds.pheno.age, marker_ds.pheno.is_female],
    ...     z_t=marker_ds.GT.n_alt_alleles(),
    ...     p_path='output/p.bm')

    Fit the model and examine :math:`h^2`:

    >>> model.fit()
    >>> model.h_sq

    Sanity-check the normalized likelihood of :math:`h^2` over the percentile
    grid:

    >>> import matplotlib.pyplot as plt                     # doctest: +SKIP
    >>> plt.plot(range(101), model.h_sq_normalized_lkhd())  # doctest: +SKIP

    For this value of :math:`h^2`, test each variant for association:

    >>> result_table = hl.linear_mixed_regression_rows(dataset.GT.n_alt_alleles(), model)

    Alternatively, one can define a full-rank model using a pre-computed kinship
    matrix :math:`K` in ndarray form. When :math:`K` is the realized
    relationship matrix defined by the genetic markers, we obtain the same model
    as above with :math:`P` written as a block matrix but returned as an
    ndarray:

    >>> rrm = hl.realized_relationship_matrix(marker_ds.GT).to_numpy()
    >>> model, p = hl.linear_mixed_model(
    ...     y=dataset.pheno.height,
    ...     x=[1, dataset.pheno.age, dataset.pheno.is_female],
    ...     k=rrm,
    ...     p_path='output/p.bm',
    ...     overwrite=True)

    Notes
    -----
    See :class:`.LinearMixedModel` for details on the model and notation.

    Exactly one of `z_t` and `k` must be set.

    If `z_t` is set, the model is low-rank if the number of samples :math:`n` exceeds
    the number of random effects :math:`m`. At least one dimension must be less
    than or equal to 46300. If `standardize` is true, each random effect is first
    standardized to have mean 0 and variance :math:`\frac{1}{m}`, so that the
    diagonal values of the kinship matrix `K = ZZ^T` are 1.0 in expectation.
    This kinship matrix corresponds to the :meth:`realized_relationship_matrix`
    in genetics. See :meth:`.LinearMixedModel.from_random_effects`
    and :meth:`.BlockMatrix.svd` for more details.

    If `k` is set, the model is full-rank. For correct results, the indices of
    `k` **must be aligned** with columns of the source of `y`.
    Set `p_path` if you plan to use the model in :meth:`.linear_mixed_regression_rows`.
    `k` must be positive semi-definite; symmetry is not checked as only the
    lower triangle is used. See :meth:`.LinearMixedModel.from_kinship` for more
    details.

    Missing, nan, or infinite values in `y` or `x` will raise an error.
    If set, `z_t` may only have missing values if `mean_impute` is true, in
    which case missing values of are set to the row mean. We recommend setting
    `mean_impute` to false if you expect no missing values, both for performance
    and as a sanity check.

    Warning
    -------
    If the rows of the matrix table have been filtered to a small fraction,
    then :meth:`.MatrixTable.repartition` before this method to improve
    performance.

    Parameters
    ----------
    y: :class:`.Float64Expression`
        Column-indexed expression for the observations (rows of :math:`y`).
        Must have no missing values.
    x: :obj:`list` of :class:`.Float64Expression`
        Non-empty list of column-indexed expressions for the fixed effects (rows of :math:`X`).
        Each expression must have the same source as `y` or no source
        (e.g., the intercept ``1.0``).
        Must have no missing values.
    z_t: :class:`.Float64Expression`, optional
        Entry-indexed expression for each mixed effect. These values are
        row-standardized to variance :math:`1 / m` to form the entries of
        :math:`Z^T`. If `mean_impute` is false, must have no missing values.
        Exactly one of `z_t` and `k` must be set.
    k: :class:`ndarray`, optional
        Kinship matrix :math:`K`.
        Exactly one of `z_t` and `k` must be set.
    p_path: :obj:`str`, optional
        Path at which to write the projection :math:`P` as a block matrix.
        Required if `z_t` is set.
    overwrite: :obj:`bool`
        If ``True``, overwrite an existing file at `p_path`.
    standardize: :obj:`bool`
        If ``True``, standardize `z_t` by row to mean 0 and variance
        :math:`\frac{1}{m}`.
    mean_impute: :obj:`bool`
        If ``True``, mean-impute missing values of `z_t` by row.

    Returns
    -------
    model: :class:`.LinearMixedModel`
        Linear mixed model ready to be fit.
    p: :class:`ndarray` or :class:`.BlockMatrix`
        Matrix :math:`P` whose rows are the eigenvectors of :math:`K`.
        The type is block matrix if the model is low rank (i.e., if `z_t` is set
        and :math:`n > m`).
    """
    source = matrix_table_source('linear_mixed_model/y', y)

    if ((z_t is None and k is None) or
            (z_t is not None and k is not None)):
        raise ValueError("linear_mixed_model: set exactly one of 'z_t' and 'k'")

    if len(x) == 0:
        raise ValueError("linear_mixed_model: 'x' must include at least one fixed effect")

    _warn_if_no_intercept('linear_mixed_model', x)

    # collect x and y in one pass
    mt = source.select_cols(xy=hl.array(x + [y])).key_cols_by()
    xy = np.array(mt.xy.collect(), dtype=np.float64)
    xy = xy.reshape(xy.size // (len(x) + 1), len(x) + 1)
    x_nd = np.copy(xy[:, :-1])
    y_nd = np.copy(xy[:, -1])
    n = y_nd.size
    del xy

    if not np.all(np.isfinite(y_nd)):
        raise ValueError("linear_mixed_model: 'y' has missing, nan, or infinite values")
    if not np.all(np.isfinite(x_nd)):
        raise ValueError("linear_mixed_model: 'x' has missing, nan, or infinite values")

    if z_t is None:
        model, p = LinearMixedModel.from_kinship(y_nd, x_nd, k, p_path, overwrite)
    else:
        check_entry_indexed('from_matrix_table: z_t', z_t)
        if matrix_table_source('linear_mixed_model/z_t', z_t) != source:
            raise ValueError("linear_mixed_model: 'y' and 'z_t' must "
                             "have the same source")
        z_bm = BlockMatrix.from_entry_expr(z_t,
                                           mean_impute=mean_impute,
                                           center=standardize,
                                           normalize=standardize).T  # variance is 1 / n
        m = z_bm.shape[1]
        model, p = LinearMixedModel.from_random_effects(y_nd, x_nd, z_bm, p_path, overwrite)
        if standardize:
            model.s = model.s * (n / m)  # now variance is 1 / m
        if model.low_rank and isinstance(p, np.ndarray):
            assert n > m
            p = BlockMatrix.read(p_path)
    return model, p


@typecheck(entry_expr=expr_float64,
           model=LinearMixedModel,
           pa_t_path=nullable(str),
           a_t_path=nullable(str),
           mean_impute=bool,
           partition_size=nullable(int),
           pass_through=sequenceof(oneof(str, Expression)))
def linear_mixed_regression_rows(entry_expr,
                                 model,
                                 pa_t_path=None,
                                 a_t_path=None,
                                 mean_impute=True,
                                 partition_size=None,
                                 pass_through=()):
    """For each row, test an input variable for association using a linear
    mixed model.

    Examples
    --------
    See the example in :meth:`linear_mixed_model` and section below on
    efficiently testing multiple responses or sets of fixed effects.

    Notes
    -----
    See :class:`.LinearMixedModel` for details on the model and notation.

    This method packages up several steps for convenience:

    1. Read the transformation :math:`P` from ``model.p_path``.

    2. Write `entry_expr` at `a_t_path` as the block matrix :math:`A^T` with
       block size that of :math:`P`. The parallelism is ``n_rows / block_size``.

    3. Multiply and write :math:`A^T P^T` at `pa_t_path`. The parallelism is the
       number of blocks in :math:`(PA)^T`, which equals
       ``(n_rows / block_size) * (model.r / block_size)``.

    4. Compute regression results per row with
       :meth:`.LinearMixedModel.fit_alternatives`.
       The parallelism is ``n_rows / partition_size``.

    If `pa_t_path` and `a_t_path` are not set, temporary files are used.

    `entry_expr` may only have missing values if `mean_impute` is true, in
    which case missing values of are set to the row mean. We recommend setting
    `mean_impute` to false if you expect no missing values, both for performance
    and as a sanity check.

    **Efficiently varying the response or set of fixed effects**

    Computing :math:`K`, :math:`P`, :math:`S`, :math:`A^T`, and especially the
    product :math:`(PA)^T` may require significant compute when :math:`n` and/or
    :math:`m` is large. However these quantities are all independent of the
    response :math:`y` or fixed effects :math:`X`! And with the model
    diagonalized, Step 4 above is fast and scalable.

    So having run linear mixed regression once, we can
    compute :math:`h^2` and regression statistics for another response or set of
    fixed effects on the **same samples** at the roughly the speed of
    :func:`.linear_regression_rows`.

    For example, having collected another `y` and `x` as ndarrays, one can
    construct a new linear mixed model directly.

    Supposing the model is full-rank and `p` is an ndarray:

    >>> model = hl.stats.LinearMixedModel(p @ y, p @ x, s)      # doctest: +SKIP
    >>> model.fit()                                    # doctest: +SKIP
    >>> result_ht = model.fit_alternatives(pa_t_path)  # doctest: +SKIP

    Supposing the model is low-rank and `p` is a block matrix:

    >>> p = BlockMatrix.read(p_path)                             # doctest: +SKIP
    >>> py, px = (p @ y).to_numpy(), (p @ x).to_numpy()          # doctest: +SKIP
    >>> model = LinearMixedModel(py, px, s, y, x)                # doctest: +SKIP
    >>> model.fit()                                              # doctest: +SKIP
    >>> result_ht = model.fit_alternatives(pa_t_path, a_t_path)  # doctest: +SKIP

    In either case, one can easily loop through many responses or conditional
    analyses. To join results back to the matrix table:

    >>> dataset = dataset.add_row_index()                                    # doctest: +SKIP
    >>> dataset = dataset.annotate_rows(lmmreg=result_ht[dataset.row_idx]])  # doctest: +SKIP

    Warning
    -------
    For correct results, the column-index of `entry_expr` must correspond to the
    sample index of the model. This will be true, for example, if `model`
    was created with :func:`.linear_mixed_model` using (a possibly row-filtered
    version of) the source of `entry_expr`, or if `y` and `x` were collected to
    arrays from this source. Hail will raise an error if the number of columns
    does not match ``model.n``, but will not detect, for example, permuted
    samples.

    The warning on :meth:`.BlockMatrix.write_from_entry_expr` applies to this
    method when the number of samples is large.

    Note
    ----
    Use the `pass_through` parameter to include additional row fields from
    matrix table underlying ``entry_expr``. For example, to include an "rsid"
    field, set` pass_through=['rsid']`` or ``pass_through=[mt.rsid]``.

    Parameters
    ----------
    entry_expr: :class:`.Float64Expression`
        Entry-indexed expression for input variable.
        If mean_impute is false, must have no missing values.
    model: :class:`.LinearMixedModel`
        Fit linear mixed model with ``path_p`` set.
    pa_t_path: :obj:`str`, optional
        Path at which to store the transpose of :math:`PA`.
        If not set, a temporary file is used.
    a_t_path: :obj:`str`, optional
        Path at which to store the transpose of :math:`A`.
        If not set, a temporary file is used.
    mean_impute: :obj:`bool`
        Mean-impute missing values of `entry_expr` by row.
    partition_size: :obj:`int`
        Number of rows to process per partition.
        Default given by block size of :math:`P`.
    pass_through : :obj:`list` of :obj:`str` or :class:`.Expression`
        Additional row fields to include in the resulting table.

    Returns
    -------
    :class:`.Table`
    """
    mt = matrix_table_source('linear_mixed_regression_rows', entry_expr)
    n = mt.count_cols()

    check_entry_indexed('linear_mixed_regression_rows', entry_expr)
    if not model._fitted:
        raise ValueError("linear_mixed_regression_rows: 'model' has not been fit "
                         "using 'fit()'")
    if model.p_path is None:
        raise ValueError("linear_mixed_regression_rows: 'model' property 'p_path' "
                         "was not set at initialization")

    if model.n != n:
        raise ValueError(f"linear_mixed_regression_rows: linear mixed model expects {model.n} samples, "
                         f"\n    but 'entry_expr' source has {n} columns.")

    pa_t_path = new_temp_file() if pa_t_path is None else pa_t_path
    a_t_path = new_temp_file() if a_t_path is None else a_t_path
    p = BlockMatrix.read(model.p_path)

    BlockMatrix.write_from_entry_expr(entry_expr,
                                      a_t_path,
                                      mean_impute=mean_impute,
                                      block_size=p.block_size)
    a_t = BlockMatrix.read(a_t_path)
    (a_t @ p.T).write(pa_t_path, force_row_major=True)

    ht = model.fit_alternatives(pa_t_path,
                                a_t_path if model.low_rank else None,
                                partition_size)
    row_fields = _get_regression_row_fields(mt, pass_through, 'linear_mixed_regression_rows')
    for k in mt.row_key:
        del row_fields[k]

    mt_keys = mt.select_rows(**row_fields).add_row_index('__row_idx').rows().add_index('__row_idx').key_by('__row_idx')
    return mt_keys.annotate(**ht[mt_keys['__row_idx']]).key_by(*mt.row_key).drop('__row_idx')


@typecheck(key_expr=expr_any,
           weight_expr=expr_float64,
           y=expr_float64,
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           logistic=bool,
           max_size=int,
           accuracy=numeric,
           iterations=int)
def skat(key_expr, weight_expr, y, x, covariates, logistic=False,
         max_size=46340, accuracy=1e-6, iterations=10000) -> Table:
    r"""Test each keyed group of rows for association by linear or logistic
    SKAT test.

    Examples
    --------

    Test each gene for association using the linear sequence kernel association
    test:

    >>> burden_ds = hl.read_matrix_table('data/example_burden.vds')
    >>> skat_table = hl.skat(key_expr=burden_ds.gene,
    ...                      weight_expr=burden_ds.weight,
    ...                      y=burden_ds.burden.pheno,
    ...                      x=burden_ds.GT.n_alt_alleles(),
    ...                      covariates=[1, burden_ds.burden.cov1, burden_ds.burden.cov2])

    .. caution::

       By default, the Davies algorithm iterates up to 10k times until an
       accuracy of 1e-6 is achieved. Hence a reported p-value of zero with no
       issues may truly be as large as 1e-6. The accuracy and maximum number of
       iterations may be controlled by the corresponding function parameters.
       In general, higher accuracy requires more iterations.

    .. caution::

       To process a group with :math:`m` rows, several copies of an
       :math:`m \times m` matrix of doubles must fit in worker memory. Groups
       with tens of thousands of rows may exhaust worker memory causing the
       entire job to fail. In this case, use the `max_size` parameter to skip
       groups larger than `max_size`.

    Warning
    -------
    :func:`.skat` considers the same set of columns (i.e., samples, points) for
    every group, namely those columns for which **all** covariates are defined.
    For each row, missing values of `x` are mean-imputed over these columns.
    As in the example, the intercept covariate ``1`` must be included
    **explicitly** if desired.

    Notes
    -----

    This method provides a scalable implementation of the score-based
    variance-component test originally described in
    `Rare-Variant Association Testing for Sequencing Data with the Sequence Kernel Association Test
    <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3135811/>`__.

    Row weights must be non-negative. Rows with missing weights are ignored. In
    the R package ``skat``---which assumes rows are variants---default weights
    are given by evaluating the Beta(1, 25) density at the minor allele
    frequency. To replicate these weights in Hail using alternate allele
    frequencies stored in a row-indexed field `AF`, one can use the expression:

    >>> hl.dbeta(hl.min(ds2.AF), 1.0, 25.0) ** 2

    In the logistic case, the response `y` must either be numeric (with all
    present values 0 or 1) or Boolean, in which case true and false are coded
    as 1 and 0, respectively.

    The resulting :class:`.Table` provides the group's key (`id`), thenumber of
    rows in the group (`size`), the variance component score `q_stat`, the SKAT
    `p-value`, and a `fault` flag. For the toy example above, the table has the
    form:

    +-------+------+--------+---------+-------+
    |  id   | size | q_stat | p_value | fault |
    +=======+======+========+=========+=======+
    | geneA |   2  | 4.136  | 0.205   |   0   |
    +-------+------+--------+---------+-------+
    | geneB |   1  | 5.659  | 0.195   |   0   |
    +-------+------+--------+---------+-------+
    | geneC |   3  | 4.122  | 0.192   |   0   |
    +-------+------+--------+---------+-------+

    Groups larger than `max_size` appear with missing `q_stat`, `p_value`, and
    `fault`. The hard limit on the number of rows in a group is 46340.

    Note that the variance component score `q_stat` agrees with ``Q`` in the R
    package ``skat``, but both differ from :math:`Q` in the paper by the factor
    :math:`\frac{1}{2\sigma^2}` in the linear case and :math:`\frac{1}{2}` in
    the logistic case, where :math:`\sigma^2` is the unbiased estimator of
    residual variance for the linear null model. The R package also applies a
    "small-sample adjustment" to the null distribution in the logistic case
    when the sample size is less than 2000. Hail does not apply this
    adjustment.

    The fault flag is an integer indicating whether any issues occurred when
    running the Davies algorithm to compute the p-value as the right tail of a
    weighted sum of :math:`\chi^2(1)` distributions.

    +-------------+-----------------------------------------+
    | fault value | Description                             |
    +=============+=========================================+
    |      0      | no issues                               |
    +------+------+-----------------------------------------+
    |      1      | accuracy NOT achieved                   |
    +------+------+-----------------------------------------+
    |      2      | round-off error possibly significant    |
    +------+------+-----------------------------------------+
    |      3      | invalid parameters                      |
    +------+------+-----------------------------------------+
    |      4      | unable to locate integration parameters |
    +------+------+-----------------------------------------+
    |      5      | out of memory                           |
    +------+------+-----------------------------------------+

    Parameters
    ----------
    key_expr : :class:`.Expression`
        Row-indexed expression for key associated to each row.
    weight_expr : :class:`.Float64Expression`
        Row-indexed expression for row weights.
    y : :class:`.Float64Expression`
        Column-indexed response expression.
        If `logistic` is ``True``, all non-missing values must evaluate to 0 or
        1. Note that a :class:`.BooleanExpression` will be implicitly converted
        to a :class:`.Float64Expression` with this property.
    x : :class:`.Float64Expression`
        Entry-indexed expression for input variable.
    covariates : :obj:`list` of :class:`.Float64Expression`
        List of column-indexed covariate expressions.
    logistic : :obj:`bool`
        If true, use the logistic test rather than the linear test.
    max_size : :obj:`int`
        Maximum size of group on which to run the test.
    accuracy : :obj:`float`
        Accuracy achieved by the Davies algorithm if fault value is zero.
    iterations : :obj:`int`
        Maximum number of iterations attempted by the Davies algorithm.

    Returns
    -------
    :class:`.Table`
        Table of SKAT results.
    """
    mt = matrix_table_source('skat/x', x)
    check_entry_indexed('skat/x', x)

    analyze('skat/key_expr', key_expr, mt._row_indices)
    analyze('skat/weight_expr', weight_expr, mt._row_indices)
    analyze('skat/y', y, mt._col_indices)

    all_exprs = [key_expr, weight_expr, y]
    for e in covariates:
        all_exprs.append(e)
        analyze('skat/covariates', e, mt._col_indices)

    _warn_if_no_intercept('skat', covariates)

    # FIXME: remove this logic when annotation is better optimized
    if x in mt._fields_inverse:
        x_field_name = mt._fields_inverse[x]
        entry_expr = {}
    else:
        x_field_name = Env.get_uid()
        entry_expr = {x_field_name: x}

    y_field_name = '__y'
    weight_field_name = '__weight'
    key_field_name = '__key'
    cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))

    mt = mt._annotate_all(col_exprs=dict(**{y_field_name: y},
                                         **dict(zip(cov_field_names, covariates))),
                          row_exprs={weight_field_name: weight_expr,
                                     key_field_name: key_expr},
                          entry_exprs=entry_expr)

    jt = Env.hail().methods.Skat.apply(
        mt._jmt,
        key_field_name,
        weight_field_name,
        y_field_name,
        x_field_name,
        jarray(Env.jvm().java.lang.String, cov_field_names),
        logistic,
        max_size,
        accuracy,
        iterations)

    return Table._from_java(jt)


@typecheck(call_expr=expr_call,
           k=int,
           compute_loadings=bool)
def hwe_normalized_pca(call_expr, k=10, compute_loadings=False) -> Tuple[List[float], Table, Table]:
    r"""Run principal component analysis (PCA) on the Hardy-Weinberg-normalized
    genotype call matrix.

    Examples
    --------

    >>> eigenvalues, scores, loadings = hl.hwe_normalized_pca(dataset.GT, k=5)

    Notes
    -----
    This method specializes :func:`.pca` for the common use case
    of PCA in statistical genetics, that of projecting samples to a small
    number of ancestry coordinates. Variants that are all homozygous reference
    or all homozygous alternate are unnormalizable and removed before
    evaluation. See :func:`.pca` for more details.

    Users of PLINK/GCTA should be aware that Hail computes the GRM slightly
    differently with regard to missing data. In Hail, the
    :math:`ij` entry of the GRM :math:`MM^T` is simply the dot product of rows
    :math:`i` and :math:`j` of :math:`M`; in terms of :math:`C` it is

    .. math::

      \frac{1}{m}\sum_{l\in\mathcal{C}_i\cap\mathcal{C}_j}\frac{(C_{il}-2p_l)(C_{jl} - 2p_l)}{2p_l(1-p_l)}

    where :math:`\mathcal{C}_i = \{l \mid C_{il} \text{ is non-missing}\}`. In
    PLINK/GCTA the denominator :math:`m` is replaced with the number of terms in
    the sum :math:`\lvert\mathcal{C}_i\cap\mathcal{C}_j\rvert`, i.e. the
    number of variants where both samples have non-missing genotypes. While this
    is arguably a better estimator of the true GRM (trading shrinkage for
    noise), it has the drawback that one loses the clean interpretation of the
    loadings and scores as features and projections

    Separately, for the PCs PLINK/GCTA output the eigenvectors of the GRM, i.e.
    the left singular vectors :math:`U_k` instead of the component scores
    :math:`U_k S_k`. The scores have the advantage of representing true
    projections of the data onto features with the variance of a score
    reflecting the variance explained by the corresponding feature. In PC
    bi-plots this amounts to a change in aspect ratio; for use of PCs as
    covariates in regression it is immaterial.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression.
    k : :obj:`int`
        Number of principal components.
    compute_loadings : :obj:`bool`
        If ``True``, compute row loadings.

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """
    mt = matrix_table_source('hwe_normalized_pca/call_expr', call_expr)
    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())
    mt = mt.annotate_rows(__AC=agg.sum(mt.__gt),
                          __n_called=agg.count_where(hl.is_defined(mt.__gt)))
    mt = mt.filter_rows((mt.__AC > 0) & (mt.__AC < 2 * mt.__n_called))

    n_variants = mt.count_rows()
    if n_variants == 0:
        raise FatalError("hwe_normalized_pca: found 0 variants after filtering out monomorphic sites.")
    info("hwe_normalized_pca: running PCA using {} variants.".format(n_variants))

    mt = mt.annotate_rows(__mean_gt=mt.__AC / mt.__n_called)
    mt = mt.annotate_rows(
        __hwe_scaled_std_dev=hl.sqrt(mt.__mean_gt * (2 - mt.__mean_gt) * n_variants / 2))

    normalized_gt = hl.or_else((mt.__gt - mt.__mean_gt) / mt.__hwe_scaled_std_dev, 0.0)

    return pca(normalized_gt,
               k,
               compute_loadings)


@typecheck(entry_expr=expr_float64,
           k=int,
           compute_loadings=bool)
def pca(entry_expr, k=10, compute_loadings=False) -> Tuple[List[float], Table, Table]:
    r"""Run principal component analysis (PCA) on numeric columns derived from a
    matrix table.

    Examples
    --------

    For a matrix table with variant rows, sample columns, and genotype entries,
    compute the top 2 PC sample scores and eigenvalues of the matrix of 0s and
    1s encoding missingness of genotype calls.

    >>> eigenvalues, scores, _ = hl.pca(hl.int(hl.is_defined(dataset.GT)),
    ...                                 k=2)

    Warning
    -------
      This method does **not** automatically mean-center or normalize each column.
      If desired, such transformations should be incorporated in `entry_expr`.

      Hail will return an error if `entry_expr` evaluates to missing, nan, or
      infinity on any entry.

    Notes
    -----

    PCA is run on the columns of the numeric matrix obtained by evaluating
    `entry_expr` on each entry of the matrix table, or equivalently on the rows
    of the **transposed** numeric matrix :math:`M` referenced below.

    PCA computes the SVD

    .. math::

      M = USV^T

    where columns of :math:`U` are left singular vectors (orthonormal in
    :math:`\mathbb{R}^n`), columns of :math:`V` are right singular vectors
    (orthonormal in :math:`\mathbb{R}^m`), and :math:`S=\mathrm{diag}(s_1, s_2,
    \ldots)` with ordered singular values :math:`s_1 \ge s_2 \ge \cdots \ge 0`.
    Typically one computes only the first :math:`k` singular vectors and values,
    yielding the best rank :math:`k` approximation :math:`U_k S_k V_k^T` of
    :math:`M`; the truncations :math:`U_k`, :math:`S_k` and :math:`V_k` are
    :math:`n \times k`, :math:`k \times k` and :math:`m \times k`
    respectively.

    From the perspective of the rows of :math:`M` as samples (data points),
    :math:`V_k` contains the loadings for the first :math:`k` PCs while
    :math:`MV_k = U_k S_k` contains the first :math:`k` PC scores of each
    sample. The loadings represent a new basis of features while the scores
    represent the projected data on those features. The eigenvalues of the Gramian
    :math:`MM^T` are the squares of the singular values :math:`s_1^2, s_2^2,
    \ldots`, which represent the variances carried by the respective PCs. By
    default, Hail only computes the loadings if the ``loadings`` parameter is
    specified.

    Scores are stored in a :class:`.Table` with the column key of the matrix
    table as key and a field `scores` of type ``array<float64>`` containing
    the principal component scores.

    Loadings are stored in a :class:`.Table` with the row key of the matrix
    table as key and a field `loadings` of type ``array<float64>`` containing
    the principal component loadings.

    The eigenvalues are returned in descending order, with scores and loadings
    given the corresponding array order.

    Parameters
    ----------
    entry_expr : :class:`.Expression`
        Numeric expression for matrix entries.
    k : :obj:`int`
        Number of principal components.
    compute_loadings : :obj:`bool`
        If ``True``, compute row loadings.

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """
    check_entry_indexed('pca/entry_expr', entry_expr)

    mt = matrix_table_source('pca/entry_expr', entry_expr)

    #  FIXME: remove once select_entries on a field is free
    if entry_expr in mt._fields_inverse:
        field = mt._fields_inverse[entry_expr]
    else:
        field = Env.get_uid()
        mt = mt.select_entries(**{field: entry_expr})
    mt = mt.select_cols().select_rows().select_globals()

    r = Env.hail().methods.PCA.apply(mt._jmt, field, k, compute_loadings)
    scores = Table._from_java(Env.hail().methods.PCA.scoresTable(mt._jmt, r._2()))
    loadings = from_option(r._3())
    if loadings:
        loadings = Table._from_java(loadings)
    return jiterable_to_list(r._1()), scores, loadings


@typecheck(call_expr=expr_call,
           min_individual_maf=numeric,
           k=nullable(int),
           scores_expr=nullable(expr_array(expr_float64)),
           min_kinship=numeric,
           statistics=enumeration('kin', 'kin2', 'kin20', 'all'),
           block_size=nullable(int))
def pc_relate(call_expr, min_individual_maf, *, k=None, scores_expr=None,
              min_kinship=-float("inf"), statistics="all", block_size=None) -> Table:
    r"""Compute relatedness estimates between individuals using a variant of the
    PC-Relate method.

    .. include:: ../_templates/req_diploid_gt.rst

    Examples
    --------
    Estimate kinship, identity-by-descent two, identity-by-descent one, and
    identity-by-descent zero for every pair of samples, using a minimum minor
    allele frequency filter of 0.01 and 10 principal components to control
    for population structure.

    >>> rel = hl.pc_relate(dataset.GT, 0.01, k=10)

    Only compute the kinship statistic. This is more efficient than
    computing all statistics.

    >>> rel = hl.pc_relate(dataset.GT, 0.01, k=10, statistics='kin')

    Compute all statistics, excluding sample-pairs with kinship less
    than 0.1. This is more efficient than producing the full table and
    then filtering using :meth:`.Table.filter`.

    >>> rel = hl.pc_relate(dataset.GT, 0.01, k=10, min_kinship=0.1)

    One can also pass in pre-computed principal component scores.
    To produce the same results as in the previous example:

    >>> _, scores_table, _ = hl.hwe_normalized_pca(dataset.GT,
    ...                                      k=10,
    ...                                      compute_loadings=False)
    >>> rel = hl.pc_relate(dataset.GT,
    ...                    0.01,
    ...                    scores_expr=scores_table[dataset.col_key].scores,
    ...                    min_kinship=0.1)

    Notes
    -----
    The traditional estimator for kinship between a pair of individuals
    :math:`i` and :math:`j`, sharing the set :math:`S_{ij}` of
    single-nucleotide variants, from a population with allele frequencies
    :math:`p_s`, is given by:

    .. math::

      \widehat{\phi_{ij}} :=
        \frac{1}{|S_{ij}|}
        \sum_{s \in S_{ij}}
          \frac{(g_{is} - 2 p_s) (g_{js} - 2 p_s)}
                {4 \sum_{s \in S_{ij} p_s (1 - p_s)}}

    This estimator is true under the model that the sharing of common
    (relative to the population) alleles is not very informative to
    relatedness (because they're common) and the sharing of rare alleles
    suggests a recent common ancestor from which the allele was inherited by
    descent.

    When multiple ancestry groups are mixed in a sample, this model breaks
    down. Alleles that are rare in all but one ancestry group are treated as
    very informative to relatedness. However, these alleles are simply
    markers of the ancestry group. The PC-Relate method corrects for this
    situation and the related situation of admixed individuals.

    PC-Relate slightly modifies the usual estimator for relatedness:
    occurrences of population allele frequency are replaced with an
    "individual-specific allele frequency". This modification allows the
    method to correctly weight an allele according to an individual's unique
    ancestry profile.

    The "individual-specific allele frequency" at a given genetic locus is
    modeled by PC-Relate as a linear function of a sample's first ``k``
    principal component coordinates. As such, the efficacy of this method
    rests on two assumptions:

     - an individual's first ``k`` principal component coordinates fully
       describe their allele-frequency-relevant ancestry, and

     - the relationship between ancestry (as described by principal
       component coordinates) and population allele frequency is linear

    The estimators for kinship, and identity-by-descent zero, one, and two
    follow. Let:

     - :math:`S_{ij}` be the set of genetic loci at which both individuals
       :math:`i` and :math:`j` have a defined genotype

     - :math:`g_{is} \in {0, 1, 2}` be the number of alternate alleles that
       individual :math:`i` has at genetic locus :math:`s`

     - :math:`\widehat{\mu_{is}} \in [0, 1]` be the individual-specific allele
       frequency for individual :math:`i` at genetic locus :math:`s`

     - :math:`{\widehat{\sigma^2_{is}}} := \widehat{\mu_{is}} (1 - \widehat{\mu_{is}})`,
       the binomial variance of :math:`\widehat{\mu_{is}}`

     - :math:`\widehat{\sigma_{is}} := \sqrt{\widehat{\sigma^2_{is}}}`,
       the binomial standard deviation of :math:`\widehat{\mu_{is}}`

     - :math:`\text{IBS}^{(0)}_{ij} := \sum_{s \in S_{ij}} \mathbb{1}_{||g_{is} - g_{js} = 2||}`,
       the number of genetic loci at which individuals :math:`i` and :math:`j`
       share no alleles

     - :math:`\widehat{f_i} := 2 \widehat{\phi_{ii}} - 1`, the inbreeding
       coefficient for individual :math:`i`

     - :math:`g^D_{is}` be a dominance encoding of the genotype matrix, and
       :math:`X_{is}` be a normalized dominance-coded genotype matrix

    .. math::

        g^D_{is} :=
          \begin{cases}
            \widehat{\mu_{is}}     & g_{is} = 0 \\
            0                        & g_{is} = 1 \\
            1 - \widehat{\mu_{is}} & g_{is} = 2
          \end{cases}

        X_{is} := g^D_{is} - \widehat{\sigma^2_{is}} (1 - \widehat{f_i})

    The estimator for kinship is given by:

    .. math::

      \widehat{\phi_{ij}} :=
        \frac{\sum_{s \in S_{ij}}(g - 2 \mu)_{is} (g - 2 \mu)_{js}}
              {4 * \sum_{s \in S_{ij}}
                            \widehat{\sigma_{is}} \widehat{\sigma_{js}}}

    The estimator for identity-by-descent two is given by:

    .. math::

      \widehat{k^{(2)}_{ij}} :=
        \frac{\sum_{s \in S_{ij}}X_{is} X_{js}}{\sum_{s \in S_{ij}}
          \widehat{\sigma^2_{is}} \widehat{\sigma^2_{js}}}

    The estimator for identity-by-descent zero is given by:

    .. math::

      \widehat{k^{(0)}_{ij}} :=
        \begin{cases}
          \frac{\text{IBS}^{(0)}_{ij}}
                {\sum_{s \in S_{ij}}
                       \widehat{\mu_{is}}^2(1 - \widehat{\mu_{js}})^2
                       + (1 - \widehat{\mu_{is}})^2\widehat{\mu_{js}}^2}
            & \widehat{\phi_{ij}} > 2^{-5/2} \\
          1 - 4 \widehat{\phi_{ij}} + k^{(2)}_{ij}
            & \widehat{\phi_{ij}} \le 2^{-5/2}
        \end{cases}

    The estimator for identity-by-descent one is given by:

    .. math::

      \widehat{k^{(1)}_{ij}} :=
        1 - \widehat{k^{(2)}_{ij}} - \widehat{k^{(0)}_{ij}}

    Note that, even if present, phase information is ignored by this method.

    The PC-Relate method is described in "Model-free Estimation of Recent
    Genetic Relatedness". Conomos MP, Reiner AP, Weir BS, Thornton TA. in
    American Journal of Human Genetics. 2016 Jan 7. The reference
    implementation is available in the `GENESIS Bioconductor package
    <https://bioconductor.org/packages/release/bioc/html/GENESIS.html>`_ .

    :func:`.pc_relate` differs from the reference implementation in a few
    ways:

     - if `k` is supplied, samples scores are computed via PCA on all samples,
       not a specified subset of genetically unrelated samples. The latter
       can be achieved by filtering samples, computing PCA variant loadings,
       and using these loadings to compute and pass in scores for all samples.

     - the estimators do not perform small sample correction

     - the algorithm does not provide an option to use population-wide
       allele frequency estimates

     - the algorithm does not provide an option to not use "overall
       standardization" (see R ``pcrelate`` documentation)

    Under the PC-Relate model, kinship, :math:`\phi_{ij}`, ranges from 0 to
    0.5, and is precisely half of the
    fraction-of-genetic-material-shared. Listed below are the statistics for
    a few pairings:

     - Monozygotic twins share all their genetic material so their kinship
       statistic is 0.5 in expection.

     - Parent-child and sibling pairs both have kinship 0.25 in expectation
       and are separated by the identity-by-descent-zero, :math:`k^{(2)}_{ij}`,
       statistic which is zero for parent-child pairs and 0.25 for sibling
       pairs.

     - Avuncular pairs and grand-parent/-child pairs both have kinship 0.125
       in expectation and both have identity-by-descent-zero 0.5 in expectation

     - "Third degree relatives" are those pairs sharing
       :math:`2^{-3} = 12.5 %` of their genetic material, the results of
       PCRelate are often too noisy to reliably distinguish these pairs from
       higher-degree-relative-pairs or unrelated pairs.

    Note that :math:`g_{is}` is the number of alternate alleles. Hence, for
    multi-allelic variants, a value of 2 may indicate two distinct alternative
    alleles rather than a homozygous variant genotype. To enforce the latter,
    either filter or split multi-allelic variants first.

    The resulting table has the first 3, 4, 5, or 6 fields below, depending on
    the `statistics` parameter:

     - `i` (``col_key.dtype``) -- First sample. (key field)
     - `j` (``col_key.dtype``) -- Second sample. (key field)
     - `kin` (:py:data:`.tfloat64`) -- Kinship estimate, :math:`\widehat{\phi_{ij}}`.
     - `ibd2` (:py:data:`.tfloat64`) -- IBD2 estimate, :math:`\widehat{k^{(2)}_{ij}}`.
     - `ibd0` (:py:data:`.tfloat64`) -- IBD0 estimate, :math:`\widehat{k^{(0)}_{ij}}`.
     - `ibd1` (:py:data:`.tfloat64`) -- IBD1 estimate, :math:`\widehat{k^{(1)}_{ij}}`.

    Here ``col_key`` refers to the column key of the source matrix table,
    and ``col_key.dtype`` is a struct containing the column key fields.

    There is one row for each pair of distinct samples (columns), where `i`
    corresponds to the column of smaller column index. In particular, if the
    same column key value exists for :math:`n` columns, then the resulting
    table will have :math:`\binom{n-1}{2}` rows with both key fields equal to
    that column key value. This may result in unexpected behavior in downstream
    processing.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression.
    min_individual_maf : :obj:`float`
        The minimum individual-specific minor allele frequency.
        If either individual-specific minor allele frequency for a pair of
        individuals is below this threshold, then the variant will not
        be used to estimate relatedness for the pair.
    k : :obj:`int`, optional
        If set, `k` principal component scores are computed and used.
        Exactly one of `k` and `scores_expr` must be specified.
    scores_expr : :class:`.ArrayNumericExpression`, optional
        Column-indexed expression of principal component scores, with the same
        source as `call_expr`. All array values must have the same positive length,
        corresponding to the number of principal components, and all scores must
        be non-missing. Exactly one of `k` and `scores_expr` must be specified.
    min_kinship : :obj:`float`
        Pairs of samples with kinship lower than ``min_kinship`` are excluded
        from the results.
    statistics : :obj:`str`
        Set of statistics to compute.
        If ``'kin'``, only estimate the kinship statistic.
        If ``'kin2'``, estimate the above and IBD2.
        If ``'kin20'``, estimate the above and IBD0.
        If ``'all'``, estimate the above and IBD1.
    block_size : :obj:`int`, optional
        Block size of block matrices used in the algorithm.
        Default given by :meth:`.BlockMatrix.default_block_size`.

    Returns
    -------
    :class:`.Table`
        A :class:`.Table` mapping pairs of samples to their pair-wise statistics.
    """
    mt = matrix_table_source('pc_relate/call_expr', call_expr)

    if k and scores_expr is None:
        _, scores, _ = hwe_normalized_pca(mt.GT, k, compute_loadings=False)
        scores_expr = scores[mt.col_key].scores
    elif not k and scores_expr is not None:
        analyze('pc_relate/scores_expr', scores_expr, mt._col_indices)
    elif k and scores_expr is not None:
        raise ValueError("pc_relate: exactly one of 'k' and 'scores_expr' must be set, found both")
    else:
        raise ValueError("pc_relate: exactly one of 'k' and 'scores_expr' must be set, found neither")

    scores_table = mt.select_cols(__scores=scores_expr)\
        .key_cols_by().select_cols('__scores').cols()

    n_missing = scores_table.aggregate(agg.count_where(hl.is_missing(scores_table.__scores)))
    if n_missing > 0:
        raise ValueError(f'Found {n_missing} columns with missing scores array.')

    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())
    mt = mt.annotate_rows(__mean_gt=agg.mean(mt.__gt))
    mean_imputed_gt = hl.or_else(hl.float64(mt.__gt), mt.__mean_gt)

    if not block_size:
        block_size = BlockMatrix.default_block_size()

    g = BlockMatrix.from_entry_expr(mean_imputed_gt,
                                    block_size=block_size)

    int_statistics = {'kin': 0, 'kin2': 1, 'kin20': 2, 'all': 3}[statistics]

    ht = Table._from_java(scala_object(Env.hail().methods, 'PCRelate')
                          .apply(Env.hc()._jhc,
                                 g._jbm,
                                 scores_table._jt,
                                 min_individual_maf,
                                 block_size,
                                 min_kinship,
                                 int_statistics))

    if statistics == 'kin':
        ht = ht.drop('ibd0', 'ibd1', 'ibd2')
    elif statistics == 'kin2':
        ht = ht.drop('ibd0', 'ibd1')
    elif statistics == 'kin20':
        ht = ht.drop('ibd1')

    col_keys = hl.literal(mt.select_cols().key_cols_by().cols().collect(), dtype=tarray(mt.col_key.dtype))
    return ht.key_by(i=col_keys[ht.i], j=col_keys[ht.j])


@typecheck(ds=oneof(Table, MatrixTable),
           keep_star=bool,
           left_aligned=bool)
def split_multi(ds, keep_star=False, left_aligned=False):
    """Split multiallelic variants.

    The resulting dataset will be keyed by the split locus and alleles.

    :func:`.split_multi` adds the following fields:

     - `was_split` (*bool*) -- ``True`` if this variant was originally
       multiallelic, otherwise ``False``.

     - `a_index` (*int*) -- The original index of this alternate allele in the
       multiallelic representation (NB: 1 is the first alternate allele or the
       only alternate allele in a biallelic variant). For example, 1:100:A:T,C
       splits into two variants: 1:100:A:T with ``a_index = 1`` and 1:100:A:C
       with ``a_index = 2``.

     - `old_locus` (*locus*) -- The original, unsplit locus.

     - `old_alleles` (*array<str>*) -- The original, unsplit alleles.

     All other fields are left unchanged.

    Example
    -------

    :func:`.split_multi_hts`, which splits multiallelic variants for the HTS
    genotype schema and updates the entry fields by downcoding the genotype, is
    implemented as:

    >>> sm = hl.split_multi(ds)
    >>> pl = hl.or_missing(
    ...      hl.is_defined(sm.PL),
    ...      (hl.range(0, 3).map(lambda i: hl.min(hl.range(0, hl.len(sm.PL))
    ...                     .filter(lambda j: hl.downcode(hl.unphased_diploid_gt_index_call(j), sm.a_index) == hl.unphased_diploid_gt_index_call(i))
    ...                     .map(lambda j: sm.PL[j])))))
    >>> split_ds = sm.annotate_entries(
    ...     GT=hl.downcode(sm.GT, sm.a_index),
    ...     AD=hl.or_missing(hl.is_defined(sm.AD),
    ...                     [hl.sum(sm.AD) - sm.AD[sm.a_index], sm.AD[sm.a_index]]),
    ...     DP=sm.DP,
    ...     PL=pl,
    ...     GQ=hl.gq_from_pl(pl)).drop('old_locus', 'old_alleles')

    Warning
    -------
    In order to support a wide variety of data types, this function splits only
    the variants on a :class:`.MatrixTable`, but **not the genotypes**. Use
    :func:`.split_multi_hts` if possible, or split the genotypes yourself using
    one of the entry modification methods: :meth:`.MatrixTable.annotate_entries`,
    :meth:`.MatrixTable.select_entries`, :meth:`.MatrixTable.transmute_entries`.

    See Also
    --------
    :func:`.split_multi_hts`

    Parameters
    ----------
    ds : :class:`.MatrixTable` or :class:`.Table`
        An unsplit dataset.
    keep_star : :obj:`bool`
        Do not filter out * alleles.
    left_aligned : :obj:`bool`
        If ``True``, variants are assumed to be left aligned and have unique
        loci. This avoids a shuffle. If the assumption is violated, an error
        is generated.

    Returns
    -------
    :class:`.MatrixTable` or :class:`.Table`
    """

    require_row_key_variant(ds, "split_multi")
    new_id = Env.get_uid()
    is_table = isinstance(ds, Table)

    old_row = ds.row if is_table else ds._rvrow
    kept_alleles = hl.range(1, hl.len(old_row.alleles))
    if not keep_star:
        kept_alleles = kept_alleles.filter(lambda i: old_row.alleles[i] != "*")

    def new_struct(variant, i):
        return hl.struct(alleles=variant.alleles,
                         locus=variant.locus,
                         a_index=i,
                         was_split=hl.len(old_row.alleles) > 2)

    def split_rows(expr, rekey):
        if isinstance(ds, MatrixTable):
            mt = (ds.annotate_rows(**{new_id: expr})
                  .explode_rows(new_id))
            if rekey:
                mt = mt.key_rows_by()
            else:
                mt = mt.key_rows_by('locus')
            new_row_expr = mt._rvrow.annotate(locus=mt[new_id]['locus'],
                                              alleles=mt[new_id]['alleles'],
                                              a_index=mt[new_id]['a_index'],
                                              was_split=mt[new_id]['was_split'],
                                              old_locus=mt.locus,
                                              old_alleles=mt.alleles).drop(new_id)

            mt = mt._select_rows('split_multi', new_row_expr)
            if rekey:
                return mt.key_rows_by('locus', 'alleles')
            else:
                return MatrixTable(MatrixKeyRowsBy(mt._mir, ['locus', 'alleles'], is_sorted=True))
        else:
            assert isinstance(ds, Table)
            ht = (ds.annotate(**{new_id: expr})
                  .explode(new_id))
            if rekey:
                ht = ht.key_by()
            else:
                ht = ht.key_by('locus')
            new_row_expr = ht.row.annotate(locus=ht[new_id]['locus'],
                                           alleles=ht[new_id]['alleles'],
                                           a_index=ht[new_id]['a_index'],
                                           was_split=ht[new_id]['was_split'],
                                           old_locus=ht.locus,
                                           old_alleles=ht.alleles).drop(new_id)

            ht = ht._select('split_multi', new_row_expr)
            if rekey:
                return ht.key_by('locus', 'alleles')
            else:
                return Table(TableKeyBy(ht._tir, ['locus', 'alleles'], is_sorted=True))

    if left_aligned:
        def make_struct(i):
            def error_on_moved(v):
                return (hl.case()
                        .when(v.locus == old_row.locus, new_struct(v, i))
                        .or_error("Found non-left-aligned variant in split_multi"))
            return hl.bind(error_on_moved,
                           hl.min_rep(old_row.locus, [old_row.alleles[0], old_row.alleles[i]]))
        return split_rows(hl.sorted(kept_alleles.map(make_struct)), False)
    else:
        def make_struct(i, cond):
            def struct_or_empty(v):
                return (hl.case()
                        .when(cond(v.locus), hl.array([new_struct(v, i)]))
                        .or_missing())
            return hl.bind(struct_or_empty,
                           hl.min_rep(old_row.locus, [old_row.alleles[0], old_row.alleles[i]]))

        def make_array(cond):
            return hl.sorted(kept_alleles.flatmap(lambda i: make_struct(i, cond)))

        left = split_rows(make_array(lambda locus: locus == ds['locus']), False)
        moved = split_rows(make_array(lambda locus: locus != ds['locus']), True)
    return left.union(moved) if is_table else left.union_rows(moved)


@typecheck(ds=oneof(Table, MatrixTable),
           keep_star=bool,
           left_aligned=bool,
           vep_root=str)
def split_multi_hts(ds, keep_star=False, left_aligned=False, vep_root='vep'):
    """Split multiallelic variants for datasets that contain one or more fields
    from a standard high-throughput sequencing entry schema.

    .. code-block:: text

      struct {
        GT: call,
        AD: array<int32>,
        DP: int32,
        GQ: int32,
        PL: array<int32>,
        PGT: call,
        PID: str
      }

    For other entry fields, write your own splitting logic using
    :meth:`.MatrixTable.annotate_entries`.

    Examples
    --------

    >>> hl.split_multi_hts(dataset).write('output/split.vds')

    Notes
    -----

    We will explain by example. Consider a hypothetical 3-allelic
    variant:

    .. code-block:: text

      A   C,T 0/2:7,2,6:15:45:99,50,99,0,45,99

    :func:`.split_multi_hts` will create two biallelic variants (one for each
    alternate allele) at the same position

    .. code-block:: text

      A   C   0/0:13,2:15:45:0,45,99
      A   T   0/1:9,6:15:50:50,0,99

    Each multiallelic `GT` or `PGT` field is downcoded once for each alternate allele. A
    call for an alternate allele maps to 1 in the biallelic variant
    corresponding to itself and 0 otherwise. For example, in the example above,
    0/2 maps to 0/0 and 0/1. The genotype 1/2 maps to 0/1 and 0/1.

    The biallelic alt `AD` entry is just the multiallelic `AD` entry
    corresponding to the alternate allele. The ref AD entry is the sum of the
    other multiallelic entries.

    The biallelic `DP` is the same as the multiallelic `DP`.

    The biallelic `PL` entry for a genotype g is the minimum over `PL` entries
    for multiallelic genotypes that downcode to g. For example, the `PL` for (A,
    T) at 0/1 is the minimum of the PLs for 0/1 (50) and 1/2 (45), and thus 45.

    Fixing an alternate allele and biallelic variant, downcoding gives a map
    from multiallelic to biallelic alleles and genotypes. The biallelic `AD` entry
    for an allele is just the sum of the multiallelic `AD` entries for alleles
    that map to that allele. Similarly, the biallelic `PL` entry for a genotype is
    the minimum over multiallelic `PL` entries for genotypes that map to that
    genotype.

    `GQ` is recomputed from `PL` if `PL` is provided. If not, it is copied from the
    original GQ.

    Here is a second example for a het non-ref

    .. code-block:: text

      A   C,T 1/2:2,8,6:16:45:99,50,99,45,0,99

    splits as

    .. code-block:: text

      A   C   0/1:8,8:16:45:45,0,99
      A   T   0/1:10,6:16:50:50,0,99

    **VCF Info Fields**

    Hail does not split fields in the info field. This means that if a
    multiallelic site with `info.AC` value ``[10, 2]`` is split, each split
    site will contain the same array ``[10, 2]``. The provided allele index
    field `a_index` can be used to select the value corresponding to the split
    allele's position:

    >>> split_ds = hl.split_multi_hts(dataset)
    >>> split_ds = split_ds.filter_rows(split_ds.info.AC[split_ds.a_index - 1] < 10,
    ...                                 keep = False)

    VCFs split by Hail and exported to new VCFs may be
    incompatible with other tools, if action is not taken
    first. Since the "Number" of the arrays in split multiallelic
    sites no longer matches the structure on import ("A" for 1 per
    allele, for example), Hail will export these fields with
    number ".".

    If the desired output is one value per site, then it is
    possible to use annotate_variants_expr to remap these
    values. Here is an example:

    >>> split_ds = hl.split_multi_hts(dataset)
    >>> split_ds = split_ds.annotate_rows(info = Struct(AC=split_ds.info.AC[split_ds.a_index - 1],
    ...                                   **split_ds.info)) # doctest: +SKIP
    >>> hl.export_vcf(split_ds, 'output/export.vcf') # doctest: +SKIP

    The info field AC in *data/export.vcf* will have ``Number=1``.

    **New Fields**

    :func:`.split_multi_hts` adds the following fields:

     - `was_split` (*bool*) -- ``True`` if this variant was originally
       multiallelic, otherwise ``False``.

     - `a_index` (*int*) -- The original index of this alternate allele in the
       multiallelic representation (NB: 1 is the first alternate allele or the
       only alternate allele in a biallelic variant). For example, 1:100:A:T,C
       splits into two variants: 1:100:A:T with ``a_index = 1`` and 1:100:A:C
       with ``a_index = 2``.

    See Also
    --------
    :func:`.split_multi`

    Parameters
    ----------
    ds : :class:`.MatrixTable` or :class:`.Table`
        An unsplit dataset.
    keep_star : :obj:`bool`
        Do not filter out * alleles.
    left_aligned : :obj:`bool`
        If ``True``, variants are assumed to be left
        aligned and have unique loci. This avoids a shuffle. If the assumption
        is violated, an error is generated.
    vep_root : :obj:`str`
        Top-level location of vep data. All variable-length VEP fields
        (intergenic_consequences, motif_feature_consequences,
        regulatory_feature_consequences, and transcript_consequences)
        will be split properly (i.e. a_index corresponding to the VEP allele_num).

    Returns
    -------
    :class:`.MatrixTable` or :class:`.Table`
        A biallelic variant dataset.
    """

    split = split_multi(ds, keep_star=keep_star, left_aligned=left_aligned)

    row_fields = set(ds.row)
    update_rows_expression = {}
    if vep_root in row_fields:
        update_rows_expression[vep_root] = split[vep_root].annotate(**{
            x: split[vep_root][x].filter(lambda csq: csq.allele_num == split.a_index)
            for x in ('intergenic_consequences', 'motif_feature_consequences',
                      'regulatory_feature_consequences', 'transcript_consequences')})

    if isinstance(ds, Table):
        return split.annotate(**update_rows_expression).drop('old_locus', 'old_alleles')

    entry_fields = ds.entry

    expected_field_types = {
        'GT': hl.tcall,
        'AD': hl.tarray(hl.tint),
        'DP': hl.tint,
        'GQ': hl.tint,
        'PL': hl.tarray(hl.tint),
        'PGT': hl.tcall,
        'PID': hl.tstr
    }

    bad_fields = []
    for field in entry_fields:
        if field in expected_field_types and entry_fields[field].dtype != expected_field_types[field]:
            bad_fields.append((field, entry_fields[field].dtype, expected_field_types[field]))

    if bad_fields:
        msg = '\n  '.join([f"'{x[0]}'\tfound: {x[1]}\texpected: {x[2]}" for x in bad_fields])
        raise TypeError("'split_multi_hts': Found invalid types for the following fields:\n  " + msg)

    update_entries_expression = {}
    if 'GT' in entry_fields:
        update_entries_expression['GT'] = hl.downcode(split.GT, split.a_index)
    if 'DP' in entry_fields:
        update_entries_expression['DP'] = split.DP
    if 'AD' in entry_fields:
        update_entries_expression['AD'] = hl.or_missing(hl.is_defined(split.AD),
                                                        [hl.sum(split.AD) - split.AD[split.a_index], split.AD[split.a_index]])
    if 'PL' in entry_fields:
        pl = hl.or_missing(
            hl.is_defined(split.PL),
            (hl.range(0, 3).map(lambda i:
                                hl.min((hl.range(0, hl.triangle(split.old_alleles.length()))
                                        .filter(lambda j: hl.downcode(hl.unphased_diploid_gt_index_call(j),
                                                                      split.a_index) == hl.unphased_diploid_gt_index_call(i)
                                                ).map(lambda j: split.PL[j]))))))
        update_entries_expression['PL'] = pl
        if 'GQ' in entry_fields:
            update_entries_expression['GQ'] = hl.gq_from_pl(pl)
    else:
        if 'GQ' in entry_fields:
            update_entries_expression['GQ'] = split.GQ

    if 'PGT' in entry_fields:
        update_entries_expression['PGT'] = hl.downcode(split.PGT, split.a_index)
    if 'PID' in entry_fields:
        update_entries_expression['PID'] = split.PID
    return split._annotate_all(
        row_exprs=update_rows_expression,
        entry_exprs=update_entries_expression).drop('old_locus', 'old_alleles')


@typecheck(call_expr=expr_call)
def genetic_relatedness_matrix(call_expr) -> BlockMatrix:
    r"""Compute the genetic relatedness matrix (GRM).

    Examples
    --------

    >>> grm = hl.genetic_relatedness_matrix(dataset.GT)

    Notes
    -----
    The genetic relationship matrix (GRM) :math:`G` encodes genetic correlation
    between each pair of samples. It is defined by :math:`G = MM^T` where
    :math:`M` is a standardized version of the genotype matrix, computed as
    follows. Let :math:`C` be the :math:`n \times m` matrix of raw genotypes
    in the variant dataset, with rows indexed by :math:`n` samples and columns
    indexed by :math:`m` bialellic autosomal variants; :math:`C_{ij}` is the
    number of alternate alleles of variant :math:`j` carried by sample
    :math:`i`, which can be 0, 1, 2, or missing. For each variant :math:`j`,
    the sample alternate allele frequency :math:`p_j` is computed as half the
    mean of the non-missing entries of column :math:`j`. Entries of :math:`M`
    are then mean-centered and variance-normalized as

    .. math::

        M_{ij} = \frac{C_{ij}-2p_j}{\sqrt{2p_j(1-p_j)m}},

    with :math:`M_{ij} = 0` for :math:`C_{ij}` missing (i.e. mean genotype
    imputation). This scaling normalizes genotype variances to a common value
    :math:`1/m` for variants in Hardy-Weinberg equilibrium and is further
    motivated in the paper `Patterson, Price and Reich, 2006
    <http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190>`__.
    (The resulting amplification of signal from the low end of the allele
    frequency spectrum will also introduce noise for rare variants; common
    practice is to filter out variants with minor allele frequency below some
    cutoff.) The factor :math:`1/m` gives each sample row approximately unit
    total variance (assuming linkage equilibrium) so that the diagonal entries
    of the GRM are approximately 1. Equivalently,

    .. math::

        G_{ik} = \frac{1}{m} \sum_{j=1}^m \frac{(C_{ij}-2p_j)(C_{kj}-2p_j)}{2 p_j (1-p_j)}

    This method drops variants with :math:`p_j = 0` or math:`p_j = 1` before
    computing kinship.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression with columns corresponding
        to samples.

    Returns
    -------
    :class:`.BlockMatrix`
        Genetic relatedness matrix for all samples. Row and column indices
        correspond to matrix table column index.
    """
    mt = matrix_table_source('genetic_relatedness_matrix/call_expr', call_expr)
    check_entry_indexed('genetic_relatedness_matrix/call_expr', call_expr)

    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())
    mt = mt.select_rows(__AC=agg.sum(mt.__gt),
                        __n_called=agg.count_where(hl.is_defined(mt.__gt)))
    mt = mt.filter_rows((mt.__AC > 0) & (mt.__AC < 2 * mt.__n_called))

    mt = mt.select_rows(__mean_gt=mt.__AC / mt.__n_called)
    mt = mt.annotate_rows(__hwe_scaled_std_dev=hl.sqrt(mt.__mean_gt * (2 - mt.__mean_gt)))

    normalized_gt = hl.or_else((mt.__gt - mt.__mean_gt) / mt.__hwe_scaled_std_dev, 0.0)
    bm = BlockMatrix.from_entry_expr(normalized_gt)

    return (bm.T @ bm) / (bm.n_rows / 2.0)


@typecheck(call_expr=expr_call)
def realized_relationship_matrix(call_expr) -> BlockMatrix:
    r"""Computes the realized relationship matrix (RRM).

    Examples
    --------

    >>> rrm = hl.realized_relationship_matrix(dataset.GT)

    Notes
    -----
    The realized relationship matrix (RRM) is defined as follows. Consider the
    :math:`n \times m` matrix :math:`C` of raw genotypes, with rows indexed by
    :math:`n` samples and columns indexed by the :math:`m` bialellic autosomal
    variants; :math:`C_{ij}` is the number of alternate alleles of variant
    :math:`j` carried by sample :math:`i`, which can be 0, 1, 2, or missing. For
    each variant :math:`j`, the sample alternate allele frequency :math:`p_j` is
    computed as half the mean of the non-missing entries of column :math:`j`.
    Entries of :math:`M` are then mean-centered and variance-normalized as

    .. math::

        M_{ij} =
          \frac{C_{ij}-2p_j}
                {\sqrt{\frac{m}{n} \sum_{k=1}^n (C_{ij}-2p_j)^2}},

    with :math:`M_{ij} = 0` for :math:`C_{ij}` missing (i.e. mean genotype
    imputation). This scaling normalizes each variant column to have empirical
    variance :math:`1/m`, which gives each sample row approximately unit total
    variance (assuming linkage equilibrium) and yields the :math:`n \times n`
    sample correlation or realized relationship matrix (RRM) :math:`K` as simply

    .. math::

        K = MM^T

    Note that the only difference between the realized relationship matrix and
    the genetic relatedness matrix (GRM) used in
    :func:`.realized_relationship_matrix` is the variant (column) normalization:
    where RRM uses empirical variance, GRM uses expected variance under
    Hardy-Weinberg Equilibrium.

    This method drops variants with zero variance before computing kinship.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression on matrix table with columns corresponding
        to samples.

    Returns
    -------
    :class:`.BlockMatrix`
        Realized relationship matrix for all samples. Row and column indices
        correspond to matrix table column index.
    """
    mt = matrix_table_source('realized_relationship_matrix/call_expr', call_expr)
    check_entry_indexed('realized_relationship_matrix/call_expr', call_expr)

    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())
    mt = mt.select_rows(__AC=agg.sum(mt.__gt),
                        __ACsq=agg.sum(mt.__gt * mt.__gt),
                        __n_called=agg.count_where(hl.is_defined(mt.__gt)))
    mt = mt.select_rows(__mean_gt=mt.__AC / mt.__n_called,
                        __centered_length=hl.sqrt(mt.__ACsq - (mt.__AC ** 2) / mt.__n_called))
    mt = mt.filter_rows(mt.__centered_length > 0.1)  # truly non-zero values are at least sqrt(0.5)

    normalized_gt = hl.or_else((mt.__gt - mt.__mean_gt) / mt.__centered_length, 0.0)
    bm = BlockMatrix.from_entry_expr(normalized_gt)

    return (bm.T @ bm) / (bm.n_rows / bm.n_cols)


@typecheck(entry_expr=expr_float64, block_size=nullable(int))
def row_correlation(entry_expr, block_size=None) -> BlockMatrix:
    """Computes the correlation matrix between row vectors.

    Examples
    --------
    Consider the following dataset with three variants and four samples:

    >>> data = [{'v': '1:1:A:C', 's': 'a', 'GT': hl.Call([0, 0])},
    ...         {'v': '1:1:A:C', 's': 'b', 'GT': hl.Call([0, 0])},
    ...         {'v': '1:1:A:C', 's': 'c', 'GT': hl.Call([0, 1])},
    ...         {'v': '1:1:A:C', 's': 'd', 'GT': hl.Call([1, 1])},
    ...         {'v': '1:2:G:T', 's': 'a', 'GT': hl.Call([0, 1])},
    ...         {'v': '1:2:G:T', 's': 'b', 'GT': hl.Call([1, 1])},
    ...         {'v': '1:2:G:T', 's': 'c', 'GT': hl.Call([0, 1])},
    ...         {'v': '1:2:G:T', 's': 'd', 'GT': hl.Call([0, 0])},
    ...         {'v': '1:3:C:G', 's': 'a', 'GT': hl.Call([0, 1])},
    ...         {'v': '1:3:C:G', 's': 'b', 'GT': hl.Call([0, 0])},
    ...         {'v': '1:3:C:G', 's': 'c', 'GT': hl.Call([1, 1])},
    ...         {'v': '1:3:C:G', 's': 'd', 'GT': hl.null(hl.tcall)}]
    >>> ht = hl.Table.parallelize(data, hl.dtype('struct{v: str, s: str, GT: call}'))
    >>> mt = ht.to_matrix_table(row_key=['v'], col_key=['s'])

    Compute genotype correlation between all pairs of variants:

    >>> ld = hl.row_correlation(mt.GT.n_alt_alleles())
    >>> ld.to_numpy()
    array([[ 1.        , -0.85280287,  0.42640143],
           [-0.85280287,  1.        , -0.5       ],
           [ 0.42640143, -0.5       ,  1.        ]])

    Compute genotype correlation between consecutively-indexed variants:

    >>> ld.sparsify_band(lower=0, upper=1).to_numpy()
    array([[ 1.        , -0.85280287,  0.        ],
           [ 0.        ,  1.        , -0.5       ],
           [ 0.        ,  0.        ,  1.        ]])

    Warning
    -------
    Rows with a constant value (i.e., zero variance) will result `nan`
    correlation values. To avoid this, first check that all rows vary or filter
    out constant rows (for example, with the help of :func:`.aggregators.stats`).

    Notes
    -----
    In this method, each row of entries is regarded as a vector with elements
    defined by `entry_expr` and missing values mean-imputed per row.
    The ``(i, j)`` element of the resulting block matrix is the correlation
    between rows ``i`` and ``j`` (as 0-indexed by order in the matrix table;
    see :meth:`add_row_index`).

    The correlation of two vectors is defined as the
    `Pearson correlation coeffecient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__
    between the corresponding empirical distributions of elements,
    or equivalently as the cosine of the angle between the vectors.

    This method has two stages:

    - writing the row-normalized block matrix to a temporary file on persistent
      disk with :meth:`BlockMatrix.from_entry_expr`. The parallelism is
      ``n_rows / block_size``.

    - reading and multiplying this block matrix by its transpose. The
      parallelism is ``(n_rows / block_size)^2`` if all blocks are computed.

    Warning
    -------
    See all warnings on :meth:`BlockMatrix.from_entry_expr`. In particular,
    for large matrices, it may be preferable to run the two stages separately,
    saving the row-normalized block matrix to a file on external storage with
    :meth:`BlockMatrix.write_from_entry_expr`.

    The resulting number of matrix elements is the square of the number of rows
    in the matrix table, so computing the full matrix may be infeasible. For
    example, ten million rows would produce 800TB of float64 values. The
    block-sparse representation on BlockMatrix may be used to work efficiently
    with regions of such matrices, as in the second example above and
    :meth:`ld_matrix`.

    To prevent excessive re-computation, be sure to write and read the (possibly
    block-sparsified) result before multiplication by another matrix.

    Parameters
    ----------
    entry_expr : :class:`.Float64Expression`
        Entry-indexed numeric expression on matrix table.
    block_size : :obj:`int`, optional
        Block size. Default given by :meth:`.BlockMatrix.default_block_size`.

    Returns
    -------
    :class:`.BlockMatrix`
        Correlation matrix between row vectors. Row and column indices
        correspond to matrix table row index.
    """
    bm = BlockMatrix.from_entry_expr(entry_expr, mean_impute=True, center=True, normalize=True, block_size=block_size)
    return bm @ bm.T


@typecheck(entry_expr=expr_float64,
           locus_expr=expr_locus(),
           radius=oneof(int, float),
           coord_expr=nullable(expr_float64),
           block_size=nullable(int))
def ld_matrix(entry_expr, locus_expr, radius, coord_expr=None, block_size=None) -> BlockMatrix:
    """Computes the windowed correlation (linkage disequilibrium) matrix between
    variants.

    Examples
    --------
    Consider the following dataset consisting of three variants with centimorgan
    coordinates and four samples:

    >>> data = [{'v': '1:1:A:C',       'cm': 0.1, 's': 'a', 'GT': hl.Call([0, 0])},
    ...         {'v': '1:1:A:C',       'cm': 0.1, 's': 'b', 'GT': hl.Call([0, 0])},
    ...         {'v': '1:1:A:C',       'cm': 0.1, 's': 'c', 'GT': hl.Call([0, 1])},
    ...         {'v': '1:1:A:C',       'cm': 0.1, 's': 'd', 'GT': hl.Call([1, 1])},
    ...         {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'a', 'GT': hl.Call([0, 1])},
    ...         {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'b', 'GT': hl.Call([1, 1])},
    ...         {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'c', 'GT': hl.Call([0, 1])},
    ...         {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'd', 'GT': hl.Call([0, 0])},
    ...         {'v': '2:1:C:G',       'cm': 0.2, 's': 'a', 'GT': hl.Call([0, 1])},
    ...         {'v': '2:1:C:G',       'cm': 0.2, 's': 'b', 'GT': hl.Call([0, 0])},
    ...         {'v': '2:1:C:G',       'cm': 0.2, 's': 'c', 'GT': hl.Call([1, 1])},
    ...         {'v': '2:1:C:G',       'cm': 0.2, 's': 'd', 'GT': hl.null(hl.tcall)}]
    >>> ht = hl.Table.parallelize(data, hl.dtype('struct{v: str, s: str, cm: float64, GT: call}'))
    >>> ht = ht.transmute(**hl.parse_variant(ht.v))
    >>> mt = ht.to_matrix_table(row_key=['locus', 'alleles'], col_key=['s'], row_fields=['cm'])

    Compute linkage disequilibrium between all pairs of variants on the same
    contig and within two megabases:

    >>> ld = hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=2e6)
    >>> ld.to_numpy()
    array([[ 1.        , -0.85280287,  0.        ],
           [-0.85280287,  1.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])

    Within one megabases:

    >>> ld = hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=1e6)
    >>> ld.to_numpy()
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Within one centimorgan:

    >>> ld = hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=1.0, coord_expr=mt.cm)
    >>> ld.to_numpy()
    array([[ 1.        , -0.85280287,  0.        ],
           [-0.85280287,  1.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])

    Within one centimorgan, and only calculate the upper triangle:

    >>> ld = hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=1.0, coord_expr=mt.cm)
    >>> ld = ld.sparsify_triangle()
    >>> ld.to_numpy()
    array([[ 1.        , -0.85280287,  0.        ],
           [ 0.        ,  1.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])

    Notes
    -----
    This method sparsifies the result of :meth:`row_correlation` using
    :func:`.linalg.utils.locus_windows` and
    :meth:`.BlockMatrix.sparsify_row_intervals`
    in order to only compute linkage disequilibrium between nearby
    variants. Use :meth:`row_correlation` directly to calculate correlation
    without windowing.

    More precisely, variants are 0-indexed by their order in the matrix table
    (see :meth:`add_row_index`). Each variant is regarded as a vector of
    elements defined by `entry_expr`, typically the number of alternate alleles
    or genotype dosage. Missing values are mean-imputed within variant.

    The method produces a symmetric block-sparse matrix supported in a
    neighborhood of the diagonal. If variants ``i`` and ``j`` are on the same
    contig and within `radius` base pairs (inclusive) then the ``(i, j)``
    element is their
    `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__.
    Otherwise, the ``(i, j)`` element is ``0.0``.

    Rows with a constant value (i.e., zero variance) will result in `nan`
    correlation values. To avoid this, first check that all variants vary or
    filter out constant variants (for example, with the help of
    :func:`.aggregators.stats`).

    If the :meth:`.global_position` on `locus_expr` is not in ascending order,
    this method will fail. Ascending order should hold for a matrix table keyed
    by locus or variant (and the associated row table), or for a table that's
    been ordered by `locus_expr`.

    Set `coord_expr` to use a value other than position to define the windows.
    This row-indexed numeric expression must be non-missing, non-``nan``, on the
    same source as `locus_expr`, and ascending with respect to locus
    position for each contig; otherwise the method will raise an error.

    Warning
    -------
    See the warnings in :meth:`row_correlation`. In particular, for large
    matrices it may be preferable to run its stages separately.

    `entry_expr` and `locus_expr` are implicitly aligned by row-index, though
    they need not be on the same source. If their sources differ in the number
    of rows, an error will be raised; otherwise, unintended misalignment may
    silently produce unexpected results.

    Parameters
    ----------
    entry_expr : :class:`.Float64Expression`
        Entry-indexed numeric expression on matrix table.
    locus_expr : :class:`.LocusExpression`
        Row-indexed locus expression on a table or matrix table that is
        row-aligned with the matrix table of `entry_expr`.
    radius: :obj:`int` or :obj:`float`
        Radius of window for row values.
    coord_expr: :class:`.Float64Expression`, optional
        Row-indexed numeric expression for the row value on the same table or
        matrix table as `locus_expr`.
        By default, the row value is given by the locus position.
    block_size : :obj:`int`, optional
        Block size. Default given by :meth:`.BlockMatrix.default_block_size`.

    Returns
    -------
    :class:`.BlockMatrix`
        Windowed correlation matrix between variants.
        Row and column indices correspond to matrix table variant index.
    """
    starts, stops = hl.linalg.utils.locus_windows(locus_expr, radius, coord_expr)
    ld = hl.row_correlation(entry_expr, block_size)
    return ld.sparsify_row_intervals(starts, stops)


@typecheck(n_populations=int,
           n_samples=int,
           n_variants=int,
           n_partitions=nullable(int),
           pop_dist=nullable(sequenceof(numeric)),
           fst=nullable(sequenceof(numeric)),
           af_dist=expr_any,
           reference_genome=reference_genome_type,
           mixture=bool)
def balding_nichols_model(n_populations, n_samples, n_variants, n_partitions=None,
                          pop_dist=None, fst=None, af_dist=hl.rand_unif(0.1, 0.9, seed=0),
                          reference_genome='default', mixture=False) -> MatrixTable:
    r"""Generate a matrix table of variants, samples, and genotypes using the
    Balding-Nichols model.

    Examples
    --------
    Generate a matrix table of genotypes with 1000 variants and 100 samples
    across 3 populations:

    >>> bn_ds = hl.balding_nichols_model(3, 100, 1000)

    Generate a matrix table using 4 populations, 40 samples, 150 variants, 3
    partitions, population distribution ``[0.1, 0.2, 0.3, 0.4]``,
    :math:`F_{ST}` values ``[.02, .06, .04, .12]``, ancestral allele
    frequencies drawn from a truncated beta distribution with ``a = 0.01`` and
    ``b = 0.05`` over the interval ``[0.05, 1]``, and random seed 1:

    >>> hl.set_global_seed(1)
    >>> bn_ds = hl.balding_nichols_model(4, 40, 150, 3,
    ...          pop_dist=[0.1, 0.2, 0.3, 0.4],
    ...          fst=[.02, .06, .04, .12],
    ...          af_dist=hl.rand_beta(a=0.01, b=2.0, lower=0.05, upper=1.0))

    Note that in order to guarantee reproducibility, the hail global seed is set
    with :func:`.set_global_seed` immediately prior to generating the dataset.

    Notes
    -----
    This method simulates a matrix table of variants, samples, and genotypes
    using the Balding-Nichols model, which we now define.

    - :math:`K` populations are labeled by integers 0, 1, ..., K - 1.
    - :math:`N` samples are labeled by strings 0, 1, ..., N - 1.
    - :math:`M` variants are defined as ``1:1:A:C``, ``1:2:A:C``, ...,
      ``1:M:A:C``.
    - The default distribution for population assignment :math:`\pi` is uniform.
    - The default ancestral frequency distribution :math:`P_0` is uniform on
      ``[0.1, 0.9]``.
      All three classes are located in ``hail.stats``.
    - The default :math:`F_{ST}` values are all 0.1.

    The Balding-Nichols model models genotypes of individuals from a structured
    population comprising :math:`K` homogeneous modern populations that have
    each diverged from a single ancestral population (a `star phylogeny`). Each
    sample is assigned a population by sampling from the categorical
    distribution :math:`\pi`. Note that the actual size of each population is
    random.

    Variants are modeled as biallelic and unlinked. Ancestral allele
    frequencies are drawn independently for each variant from a frequency
    spectrum :math:`P_0`. The extent of genetic drift of each modern population
    from the ancestral population is defined by the corresponding :math:`F_{ST}`
    parameter :math:`F_k` (here and below, lowercase indices run over a range
    bounded by the corresponding uppercase parameter, e.g. :math:`k = 1, \ldots,
    K`). For each variant and population, allele frequencies are drawn from a
    `beta distribution <https://en.wikipedia.org/wiki/Beta_distribution>`__
    whose parameters are determined by the ancestral allele frequency and
    :math:`F_{ST}` parameter. The beta distribution gives a continuous
    approximation of the effect of genetic drift. We denote sample population
    assignments by :math:`k_n`, ancestral allele frequencies by :math:`p_m`,
    population allele frequencies by :math:`p_{k, m}`, and diploid, unphased
    genotype calls by :math:`g_{n, m}` (0, 1, and 2 correspond to homozygous
    reference, heterozygous, and homozygous variant, respectively).

    The generative model is then given by:

    .. math::
        k_n \,&\sim\, \pi

        p_m \,&\sim\, P_0

        p_{k,m} \mid p_m\,&\sim\, \mathrm{Beta}(\mu = p_m,\, \sigma^2 = F_k p_m (1 - p_m))

        g_{n,m} \mid k_n, p_{k, m} \,&\sim\, \mathrm{Binomial}(2, p_{k_n, m})

    The beta distribution by its mean and variance above; the usual parameters
    are :math:`a = (1 - p) \frac{1 - F}{F}` and :math:`b = p \frac{1 - F}{F}` with
    :math:`F = F_k` and :math:`p = p_m`.

    The resulting dataset has the following fields.

    Global fields:

    - `bn.n_populations` (:py:data:`.tint32`) -- Number of populations.
    - `bn.n_samples` (:py:data:`.tint32`) -- Number of samples.
    - `bn.n_variants` (:py:data:`.tint32`) -- Number of variants.
    - `bn.n_partitions` (:py:data:`.tint32`) -- Number of partitions.
    - `bn.pop_dist` (:class:`.tarray` of :py:data:`.tfloat64`) -- Population distribution indexed by
      population.
    - `bn.fst` (:class:`.tarray` of :py:data:`.tfloat64`) -- :math:`F_{ST}` values indexed by
      population.
    - `bn.seed` (:py:data:`.tint32`) -- Random seed.
    - `bn.mixture` (:py:data:`.tbool`) -- Value of `mixture` parameter.

    Row fields:

    - `locus` (:class:`.tlocus`) -- Variant locus (key field).
    - `alleles` (:class:`.tarray` of :py:data:`.tstr`) -- Variant alleles (key field).
    - `ancestral_af` (:py:data:`.tfloat64`) -- Ancestral allele frequency.
    - `af` (:class:`.tarray` of :py:data:`.tfloat64`) -- Modern allele frequencies indexed by
      population.

    Column fields:

    - `sample_idx` (:py:data:`.tint32`) - Sample index (key field).
    - `pop` (:py:data:`.tint32`) -- Population of sample.

    Entry fields:

    - `GT` (:py:data:`.tcall`) -- Genotype call (diploid, unphased).

    Parameters
    ----------
    n_populations : :obj:`int`
        Number of modern populations.
    n_samples : :obj:`int`
        Total number of samples.
    n_variants : :obj:`int`
        Number of variants.
    n_partitions : :obj:`int`, optional
        Number of partitions.
        Default is 1 partition per million entries or 8, whichever is larger.
    pop_dist : :obj:`list` of :obj:`float`, optional
        Unnormalized population distribution, a list of length
        ``n_populations`` with non-negative values.
        Default is ``[1, ..., 1]``.
    fst : :obj:`list` of :obj:`float`, optional
        :math:`F_{ST}` values, a list of length ``n_populations`` with values
        in (0, 1). Default is ``[0.1, ..., 0.1]``.
    af_dist : :class:`.Float64Expression` representing a random function.
        Ancestral allele frequency distribution.
        Default is :func:`.rand_unif` over the range `[0.1, 0.9]` with seed 0.
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`
        Reference genome to use.
    mixture : :obj:`bool`
        Treat `pop_dist` as the parameters of a Dirichlet distribution,
        as in the Prichard-Stevens-Donnelly model. This feature is
        EXPERIMENTAL and currently undocumented and untested.
        If ``True``, the type of `pop` is :class:`.tarray` of
        :py:data:`.tfloat64` and the value is the mixture proportions.

    Returns
    -------
    :class:`.MatrixTable`
        Simulated matrix table of variants, samples, and genotypes.
    """
    if pop_dist is None:
        pop_dist = [1 for _ in range(n_populations)]

    if fst is None:
        fst = [0.1 for _ in range(n_populations)]

    if n_partitions is None:
        n_partitions = max(8, int(n_samples * n_variants / 1000000))

    # verify args
    for name, var in {"populations": n_populations,
                      "samples": n_samples,
                      "variants": n_variants,
                      "partitions": n_partitions}.items():
        if var < 1:
            raise ValueError("n_{} must be positive, got {}".format(name, var))

    for name, var in {"pop_dist": pop_dist, "fst": fst}.items():
        if len(var) != n_populations:
            raise ValueError("{} must be of length n_populations={}, got length {}"
                             .format(name, n_populations, len(var)))

    if any(x < 0 for x in pop_dist):
        raise ValueError("pop_dist must be non-negative, got {}"
                         .format(pop_dist))

    if any(x <= 0 or x >= 1 for x in fst):
        raise ValueError("elements of fst must satisfy 0 < x < 1, got {}"
                         .format(fst))

    # verify af_dist
    if not af_dist._is_scalar:
        raise ExpressionException('balding_nichols_model expects af_dist to ' +
                                  'have scalar arguments: found expression ' +
                                  'from source {}'
                                  .format(af_dist._indices.source))

    if af_dist.dtype != tfloat64:
        raise ValueError("af_dist must be a hail function with return type tfloat64.")

    info("balding_nichols_model: generating genotypes for {} populations, {} samples, and {} variants..."
         .format(n_populations, n_samples, n_variants))

    # generate matrix table

    bn = hl.utils.range_matrix_table(n_variants, n_samples, n_partitions)
    bn = bn.annotate_globals(
        bn=hl.struct(n_populations=n_populations,
                     n_samples=n_samples,
                     n_variants=n_variants,
                     n_partitions=n_partitions,
                     pop_dist=pop_dist,
                     fst=fst,
                     mixture=mixture))
    # col info
    pop_f = hl.rand_dirichlet if mixture else hl.rand_cat
    bn = bn.key_cols_by(sample_idx=bn.col_idx)
    bn = bn.select_cols(pop=pop_f(pop_dist))

    # row info
    bn = bn.key_rows_by(locus=hl.locus_from_global_position(bn.row_idx, reference_genome=reference_genome),
                        alleles=['A', 'C'])
    bn = bn.select_rows(ancestral_af=af_dist,
                        af=hl.bind(lambda ancestral:
                                   hl.array([(1 - x) / x for x in fst])
                                   .map(lambda x:
                                        hl.rand_beta(ancestral * x,
                                                     (1 - ancestral) * x)),
                                   af_dist))
    # entry info
    p = hl.sum(bn.pop * bn.af) if mixture else bn.af[bn.pop]
    idx = hl.rand_cat([(1 - p) ** 2, 2 * p * (1-p), p ** 2])
    return bn.select_entries(GT=hl.unphased_diploid_gt_index_call(idx))


@typecheck(mt=MatrixTable, f=anytype)
def filter_alleles(mt: MatrixTable,
                   f: Callable) -> MatrixTable:
    """Filter alternate alleles.

    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------
    Keep SNPs:

    >>> ds_result = hl.filter_alleles(ds, lambda allele, i: hl.is_snp(ds.alleles[0], allele))

    Keep alleles with AC > 0:

    >>> ds_result = hl.filter_alleles(ds, lambda a, allele_index: ds.info.AC[allele_index - 1] > 0)

    Update the AC field of the resulting dataset:

    >>> updated_info = ds_result.info.annotate(AC = ds_result.new_to_old.map(lambda i: ds_result.info.AC[i-1]))
    >>> ds_result = ds_result.annotate_rows(info = updated_info)

    Notes
    -----
    The following new fields are generated:

     - `old_locus` (``locus``) -- The old locus, before filtering and computing
       the minimal representation.
     - `old_alleles` (``array<str>``) -- The old alleles, before filtering and
       computing the minimal representation.
     - old_to_new (``array<int32>``) -- An array that maps old allele index to
       new allele index. Its length is the same as `old_alleles`. Alleles that
       are filtered are missing.
     - new_to_old (``array<int32>``) -- An array that maps new allele index to
       the old allele index. Its length is the same as the modified `alleles`
       field.

    If all alternate alleles of a variant are filtered out, the variant itself
    is filtered out.

    **Using _f_**

    The `f` argument is a function or lambda evaluated per alternate allele to
    determine whether that allele is kept. If `f` evaluates to ``True``, the
    allele is kept. If `f` evaluates to ``False`` or missing, the allele is
    removed.

    `f` is a function that takes two arguments: the allele string (of type
    :class:`.StringExpression`) and the allele index (of type
    :class:`.Int32Expression`), and returns a boolean expression. This can
    be either a defined function or a lambda. For example, these two usages
    are equivalent:

    (with a lambda)

    >>> ds_result = hl.filter_alleles(ds, lambda allele, i: hl.is_snp(ds.alleles[0], allele))

    (with a defined function)

    >>> def filter_f(allele, allele_index):
    ...     return hl.is_snp(ds.alleles[0], allele)
    >>> ds_result = hl.filter_alleles(ds, filter_f)

    Warning
    -------
    :func:`.filter_alleles` does not update any fields other than `locus` and
    `alleles`. This means that row fields like allele count (AC) and entry
    fields like allele depth (AD) can become meaningless unless they are also
    updated. You can update them with :meth:`.annotate_rows` and
    :meth:`.annotate_entries`.

    See Also
    --------
    :func:`.filter_alleles_hts`

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        Dataset.
    f : callable
        Function from (allele: :class:`StringExpression`, allele_index:
        :class:`.Int32Expression`) to :class:`.BooleanExpression`

    Returns
    -------
    :class:`.MatrixTable`
    """
    require_row_key_variant(mt, 'filter_alleles')
    inclusion = hl.range(0, hl.len(mt.alleles)).map(lambda i: (i == 0) | hl.bind(lambda ii: f(mt.alleles[ii], ii), i))

    # old locus, old alleles, new to old, old to new
    mt = mt.annotate_rows(__allele_inclusion=inclusion,
                          old_locus=mt.locus,
                          old_alleles=mt.alleles)
    new_to_old = (hl.zip_with_index(mt.__allele_inclusion)
                  .filter(lambda elt: elt[1])
                  .map(lambda elt: elt[0]))
    old_to_new_dict = (hl.dict(hl.zip_with_index(hl.zip_with_index(mt.alleles)
                                                 .filter(lambda elt: mt.__allele_inclusion[elt[0]]))
                               .map(lambda elt: (elt[1][1], elt[0]))))

    old_to_new = hl.bind(lambda d: mt.alleles.map(lambda a: d.get(a)), old_to_new_dict)
    mt = mt.annotate_rows(old_to_new=old_to_new, new_to_old=new_to_old)
    new_locus_alleles = hl.min_rep(mt.locus, mt.new_to_old.map(lambda i: mt.alleles[i]))
    mt = mt.annotate_rows(__new_locus=new_locus_alleles.locus, __new_alleles=new_locus_alleles.alleles)
    mt = mt.filter_rows(hl.len(mt.__new_alleles) > 1)
    left = mt.filter_rows((mt.locus == mt.__new_locus) & (mt.alleles == mt.__new_alleles))

    right = mt.filter_rows((mt.locus != mt.__new_locus) | (mt.alleles != mt.__new_alleles))
    right = right.key_rows_by(locus=right.__new_locus, alleles=right.__new_alleles)
    return left.union_rows(right).drop('__allele_inclusion', '__new_locus', '__new_alleles')


@typecheck(mt=MatrixTable, f=anytype, subset=bool)
def filter_alleles_hts(mt: MatrixTable,
                       f: Callable,
                       subset: bool = False) -> MatrixTable:
    """Filter alternate alleles and update standard GATK entry fields.

    Examples
    --------
    Filter to SNP alleles using the subset strategy:

    >>> ds_result = hl.filter_alleles_hts(
    ...     ds,
    ...     lambda allele, _: hl.is_snp(ds.alleles[0], allele),
    ...     subset=True)

    Update the AC field of the resulting dataset:

    >>> updated_info = ds_result.info.annotate(AC = ds_result.new_to_old.map(lambda i: ds_result.info.AC[i-1]))
    >>> ds_result = ds_result.annotate_rows(info = updated_info)

    Notes
    -----
    For usage of the _f_ argument, see the :func:`.filter_alleles`
    documentation.

    :func:`.filter_alleles_hts` requires the dataset have the GATK VCF schema,
    namely the following entry fields in this order:

    .. code-block:: text

        GT: call
        AD: array<int32>
        DP: int32
        GQ: int32
        PL: array<int32>

    Use :meth:`.MatrixTable.select_entries` to rearrange these fields if
    necessary.

    The following new fields are generated:

     - `old_locus` (``locus``) -- The old locus, before filtering and computing
       the minimal representation.
     - `old_alleles` (``array<str>``) -- The old alleles, before filtering and
       computing the minimal representation.
     - old_to_new (``array<int32>``) -- An array that maps old allele index to
       new allele index. Its length is the same as `old_alleles`. Alleles that
       are filtered are missing.
     - new_to_old (``array<int32>``) -- An array that maps new allele index to
       the old allele index. Its length is the same as the modified `alleles`
       field.

    **Downcode algorithm**

    We will illustrate the behavior on the example genotype below
    when filtering the first alternate allele (allele 1) at a site
    with 1 reference allele and 2 alternate alleles.

    .. code-block:: text

      GT: 1/2
      GQ: 10
      AD: 0,50,35

      0 | 1000
      1 | 1000   10
      2 | 1000   0     20
        +-----------------
           0     1     2

    The downcode algorithm recodes occurances of filtered alleles
    to occurances of the reference allele (e.g. 1 -> 0 in our
    example). So the depths of filtered alleles in the AD field
    are added to the depth of the reference allele. Where
    downcoding filtered alleles merges distinct genotypes, the
    minimum PL is used (since PL is on a log scale, this roughly
    corresponds to adding probabilities). The PLs are then
    re-normalized (shifted) so that the most likely genotype has a
    PL of 0, and GT is set to this genotype.  If an allele is
    filtered, this algorithm acts similarly to
    :func:`.split_multi_hts`.

    The downcode algorithm would produce the following:

    .. code-block:: text

      GT: 0/1
      GQ: 10
      AD: 35,50

      0 | 20
      1 | 0    10
        +-----------
          0    1

    In summary:

     - GT: Downcode filtered alleles to reference.
     - AD: Columns of filtered alleles are eliminated and their
       values are added to the reference column, e.g., filtering
       alleles 1 and 2 transforms ``25,5,10,20`` to ``40,20``.
     - DP: No change.
     - PL: Downcode filtered alleles to reference, combine PLs
       using minimum for each overloaded genotype, and shift so
       the overall minimum PL is 0.
     - GQ: The second-lowest PL (after shifting).

    **Subset algorithm**

    We will illustrate the behavior on the example genotype below
    when filtering the first alternate allele (allele 1) at a site
    with 1 reference allele and 2 alternate alleles.

    .. code-block:: text

      GT: 1/2
      GQ: 10
      AD: 0,50,35

      0 | 1000
      1 | 1000   10
      2 | 1000   0     20
        +-----------------
           0     1     2

    The subset algorithm subsets the AD and PL arrays
    (i.e. removes entries corresponding to filtered alleles) and
    then sets GT to the genotype with the minimum PL.  Note that
    if the genotype changes (as in the example), the PLs are
    re-normalized (shifted) so that the most likely genotype has a
    PL of 0.  Qualitatively, subsetting corresponds to the belief
    that the filtered alleles are not real so we should discard
    any probability mass associated with them.

    The subset algorithm would produce the following:

    .. code-block:: text

      GT: 1/1
      GQ: 980
      AD: 0,50

      0 | 980
      1 | 980    0
        +-----------
           0      1

    In summary:

     - GT: Set to most likely genotype based on the PLs ignoring
       the filtered allele(s).
     - AD: The filtered alleles' columns are eliminated, e.g.,
       filtering alleles 1 and 2 transforms ``25,5,10,20`` to
       ``25,20``.
     - DP: Unchanged.
     - PL: Columns involving filtered alleles are eliminated and
       the remaining columns' values are shifted so the minimum
       value is 0.
     - GQ: The second-lowest PL (after shifting).

    Warning
    -------
    :func:`.filter_alleles_hts` does not update any row fields other than
    `locus` and `alleles`. This means that row fields like allele count (AC) can
    become meaningless unless they are also updated. You can update them with
    :meth:`.annotate_rows`.

    See Also
    --------
    :func:`.filter_alleles`

    Parameters
    ----------
    mt : :class:`.MatrixTable`
    f : callable
        Function from (allele: :class:`StringExpression`, allele_index:
        :class:`.Int32Expression`) to :class:`.BooleanExpression`
    subset : :obj:`.bool`
        Subset PL field if ``True``, otherwise downcode PL field. The
        calculation of GT and GQ also depend on whether one subsets or
        downcodes the PL.

    Returns
    -------
    :class:`.MatrixTable`
    """
    if mt.entry.dtype != hl.hts_entry_schema:
        raise FatalError("'filter_alleles_hts': entry schema must be the HTS entry schema:\n"
                         "  found: {}\n"
                         "  expected: {}\n"
                         "  Use 'hl.filter_alleles' to split entries with non-HTS entry fields.".format(
            mt.entry.dtype, hl.hts_entry_schema
        ))

    mt = filter_alleles(mt, f)

    if subset:
        newPL = hl.cond(
            hl.is_defined(mt.PL),
            hl.bind(
                lambda unnorm: unnorm - hl.min(unnorm),
                hl.range(0, hl.triangle(mt.alleles.length())).map(
                    lambda newi: hl.bind(
                        lambda newc: mt.PL[hl.call(mt.new_to_old[newc[0]],
                                                   mt.new_to_old[newc[1]]).unphased_diploid_gt_index()],
                        hl.unphased_diploid_gt_index_call(newi)))),
            hl.null(tarray(tint32)))
        return mt.annotate_entries(
            GT=hl.unphased_diploid_gt_index_call(hl.argmin(newPL, unique=True)),
            AD=hl.cond(
                hl.is_defined(mt.AD),
                hl.range(0, mt.alleles.length()).map(
                    lambda newi: mt.AD[mt.new_to_old[newi]]),
                hl.null(tarray(tint32))),
            # DP unchanged
            GQ=hl.gq_from_pl(newPL),
            PL=newPL)
    # otherwise downcode
    else:
        mt = mt.annotate_rows(__old_to_new_no_na = mt.old_to_new.map(lambda x: hl.or_else(x, 0)))
        newPL = hl.cond(
            hl.is_defined(mt.PL),
            (hl.range(0, hl.triangle(hl.len(mt.alleles)))
             .map(lambda newi: hl.min(hl.range(0, hl.triangle(hl.len(mt.old_alleles)))
                                      .filter(lambda oldi: hl.bind(
                lambda oldc: hl.call(mt.__old_to_new_no_na[oldc[0]],
                                     mt.__old_to_new_no_na[oldc[1]]) == hl.unphased_diploid_gt_index_call(newi),
                hl.unphased_diploid_gt_index_call(oldi)))
                                      .map(lambda oldi: mt.PL[oldi])))),
            hl.null(tarray(tint32)))
        return mt.annotate_entries(
            GT=hl.call(mt.__old_to_new_no_na[mt.GT[0]],
                       mt.__old_to_new_no_na[mt.GT[1]]),
            AD=hl.cond(
                hl.is_defined(mt.AD),
                (hl.range(0, hl.len(mt.alleles))
                 .map(lambda newi: hl.sum(hl.range(0, hl.len(mt.old_alleles))
                                          .filter(lambda oldi: mt.__old_to_new_no_na[oldi] == newi)
                                          .map(lambda oldi: mt.AD[oldi])))),
                hl.null(tarray(tint32))),
            # DP unchanged
            GQ=hl.gq_from_pl(newPL),
            PL=newPL).drop('__old_to_new_no_na')


@typecheck(mt=MatrixTable,
           call_field=str,
           r2=numeric,
           bp_window_size=int,
           memory_per_core=int)
def _local_ld_prune(mt, call_field, r2=0.2, bp_window_size=1000000, memory_per_core=256):
    bytes_per_core = memory_per_core * 1024 * 1024
    fraction_memory_to_use = 0.25
    variant_byte_overhead = 50
    genotypes_per_pack = 32
    n_samples = mt.count_cols()
    min_bytes_per_core = math.ceil((1 / fraction_memory_to_use) * 8 * n_samples + variant_byte_overhead)
    if bytes_per_core < min_bytes_per_core:
        raise ValueError("memory_per_core must be greater than {} MB".format(min_bytes_per_core // (1024 * 1024)))
    bytes_per_variant = math.ceil(8 * n_samples / genotypes_per_pack) + variant_byte_overhead
    bytes_available_per_core = bytes_per_core * fraction_memory_to_use
    max_queue_size = int(max(1.0, math.ceil(bytes_available_per_core / bytes_per_variant)))

    info(f'ld_prune: running local pruning stage with max queue size of {max_queue_size} variants')

    sites_only_table = Table._from_java(Env.hail().methods.LocalLDPrune.apply(
        mt._jmt, call_field, float(r2), bp_window_size, max_queue_size))

    return sites_only_table


@typecheck(call_expr=expr_call,
           r2=numeric,
           bp_window_size=int,
           memory_per_core=int,
           keep_higher_maf=bool,
           block_size=nullable(int))
def ld_prune(call_expr, r2=0.2, bp_window_size=1000000, memory_per_core=256, keep_higher_maf=True, block_size=None):
    """Returns a maximal subset of variants that are nearly uncorrelated within each window.

    .. include:: ../_templates/req_diploid_gt.rst

    .. include:: ../_templates/req_biallelic.rst

    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------
    Prune variants in linkage disequilibrium by filtering a dataset to those variants returned
    by :func:`.ld_prune`. If the dataset contains multiallelic variants, the multiallelic variants
    must be filtered out or split before being passed to :func:`.ld_prune`.

    >>> biallelic_dataset = dataset.filter_rows(hl.len(dataset.alleles) == 2)
    >>> pruned_variant_table = hl.ld_prune(biallelic_dataset.GT, r2=0.2, bp_window_size=500000)
    >>> filtered_ds = dataset.filter_rows(hl.is_defined(pruned_variant_table[dataset.row_key]))

    Notes
    -----
    This method finds a maximal subset of variants such that the squared Pearson
    correlation coefficient :math:`r^2` of any pair at most `bp_window_size`
    base pairs apart is strictly less than `r2`. Each variant is represented as
    a vector over samples with elements given by the (mean-imputed) number of
    alternate alleles. In particular, even if present, **phase information is
    ignored**. Variants that do not vary across samples are dropped.

    The method prunes variants in linkage disequilibrium in three stages.

    - The first, "local pruning" stage prunes correlated variants within each
      partition, using a local variant queue whose size is determined by
      `memory_per_core`. A larger queue may facilitate more local pruning in
      this stage. Minor allele frequency is not taken into account. The
      parallelism is the number of matrix table partitions.

    - The second, "global correlation" stage uses block-sparse matrix
      multiplication to compute correlation between each pair of remaining
      variants within `bp_window_size` base pairs, and then forms a graph of
      correlated variants. The parallelism of writing the locally-pruned matrix
      table as a block matrix is ``n_locally_pruned_variants / block_size``.

    - The third, "global pruning" stage applies :func:`.maximal_independent_set`
      to prune variants from this graph until no edges remain. This algorithm
      iteratively removes the variant with the highest vertex degree. If
      `keep_higher_maf` is true, then in the case of a tie for highest degree,
      the variant with lowest minor allele frequency is removed.

    Warning
    -------
    The locally-pruned matrix table and block matrix are stored as temporary files
    on persistent disk. See the warnings on `BlockMatrix.from_entry_expr` with
    regard to memory and Hadoop replication errors.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression on a matrix table with row-indexed
        variants and column-indexed samples.
    r2 : :obj:`float`
        Squared correlation threshold (exclusive upper bound).
        Must be in the range [0.0, 1.0].
    bp_window_size: :obj:`int`
        Window size in base pairs (inclusive upper bound).
    memory_per_core : :obj:`int`
        Memory in MB per core for local pruning queue.
    keep_higher_maf: :obj:`int`
        If ``True``, break ties at each step of the global pruning stage by
        preferring to keep variants with higher minor allele frequency.
    block_size: :obj:`int`, optional
        Block size for block matrices in the second stage.
        Default given by :meth:`.BlockMatrix.default_block_size`.

    Returns
    -------
    :class:`.Table`
        Table of a maximal independent set of variants.
    """
    if block_size is None:
        block_size = BlockMatrix.default_block_size()

    if not 0.0 <= r2 <= 1:
      raise ValueError(f'r2 must be in the range [0.0, 1.0], found {r2}')

    if bp_window_size < 0:
      raise ValueError(f'bp_window_size must be non-negative, found {bp_window_size}')

    check_entry_indexed('ld_prune/call_expr', call_expr)
    mt = matrix_table_source('ld_prune/call_expr', call_expr)

    require_row_key_variant(mt, 'ld_prune')

    #  FIXME: remove once select_entries on a field is free
    if call_expr in mt._fields_inverse:
        field = mt._fields_inverse[call_expr]
    else:
        field = Env.get_uid()
        mt = mt.select_entries(**{field: call_expr})
    mt = mt.select_rows().select_cols()
    mt = mt.distinct_by_row()
    locally_pruned_table_path = new_temp_file()
    (_local_ld_prune(require_biallelic(mt, 'ld_prune'), field, r2, bp_window_size, memory_per_core)
        .write(locally_pruned_table_path, overwrite=True))
    locally_pruned_table = hl.read_table(locally_pruned_table_path).add_index()

    locally_pruned_ds_path = new_temp_file()
    mt = mt.annotate_rows(info=locally_pruned_table[mt.row_key])
    (mt.filter_rows(hl.is_defined(mt.info))
        .write(locally_pruned_ds_path, overwrite=True))
    locally_pruned_ds = hl.read_matrix_table(locally_pruned_ds_path)

    n_locally_pruned_variants = locally_pruned_ds.count_rows()
    info(f'ld_prune: local pruning stage retained {n_locally_pruned_variants} variants')

    standardized_mean_imputed_gt_expr = hl.or_else(
        (locally_pruned_ds[field].n_alt_alleles() - locally_pruned_ds.info.mean) * locally_pruned_ds.info.centered_length_rec,
        0.0)

    std_gt_bm = BlockMatrix.from_entry_expr(standardized_mean_imputed_gt_expr, block_size=block_size)
    r2_bm = (std_gt_bm @ std_gt_bm.T) ** 2

    _, stops = hl.linalg.utils.locus_windows(locally_pruned_table.locus, bp_window_size)

    entries = r2_bm.sparsify_row_intervals(range(stops.size), stops, blocks_only=True).entries()
    entries = entries.filter((entries.entry >= r2) & (entries.i < entries.j))

    locally_pruned_info = locally_pruned_table.key_by('idx').select('locus', 'mean')

    entries = entries.select(info_i=locally_pruned_info[entries.i],
                             info_j=locally_pruned_info[entries.j])

    entries = entries.filter((entries.info_i.locus.contig == entries.info_j.locus.contig)
                             & (entries.info_j.locus.position - entries.info_i.locus.position <= bp_window_size))

    entries_path = new_temp_file()
    entries.write(entries_path, overwrite=True)
    entries = hl.read_table(entries_path)

    n_edges = entries.count()
    info(f'ld_prune: correlation graph of locally-pruned variants has {n_edges} edges,'
         f'\n    finding maximal independent set...')

    if keep_higher_maf:
        entries = entries.key_by(
            i=hl.struct(idx=entries.i,
                        twice_maf=hl.min(entries.info_i.mean, 2.0 - entries.info_i.mean)),
            j=hl.struct(idx=entries.j,
                        twice_maf=hl.min(entries.info_j.mean, 2.0 - entries.info_j.mean)))

        def tie_breaker(l, r):
            return hl.cond(l.twice_maf > r.twice_maf,
                           -1,
                           hl.cond(l.twice_maf < r.twice_maf,
                                   1,
                                   0))

        variants_to_remove = hl.maximal_independent_set(entries.i, entries.j, keep=False, tie_breaker=tie_breaker)
        variants_to_remove = variants_to_remove.key_by(variants_to_remove.node.idx)
    else:
        variants_to_remove = hl.maximal_independent_set(entries.i, entries.j, keep=False)

    return locally_pruned_table.filter(
        hl.is_defined(variants_to_remove[locally_pruned_table.idx]), keep=False).select().persist()


def _warn_if_no_intercept(caller, covariates):
    if all([e._indices.axes for e in covariates]):
        warn(f'{caller}: model appears to have no intercept covariate.'
             '\n    To include an intercept, add 1.0 to the list of covariates.')
        return True
    return False
