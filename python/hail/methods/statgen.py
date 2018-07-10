import itertools
import math
from typing import *

import hail as hl
import hail.expr.aggregators as agg
from hail.expr.expressions import *
from hail.expr.types import *
from hail.genetics import KinshipMatrix
from hail.genetics.reference_genome import reference_genome_type
from hail.linalg import BlockMatrix
from hail.matrixtable import MatrixTable
from hail.methods.misc import require_biallelic, require_row_key_variant, require_partition_key_locus, require_col_key_str
from hail.stats import UniformDist, BetaDist, TruncatedBetaDist
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
        dataset, _ = dataset._process_joins(maf)
        maf = maf._ast.to_hql()

    return Table(Env.hail().methods.IBD.apply(require_biallelic(dataset, 'ibd')._jvds,
                                              joption(maf),
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


@typecheck(y=oneof(expr_float64, sequenceof(expr_float64)),
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           root=str,
           block_size=int)
def linear_regression(y, x, covariates=(), root='linreg', block_size=16) -> MatrixTable:
    """For each row, test an input variable for association with
    response variables using linear regression.

    Examples
    --------

    >>> dataset_result = hl.linear_regression(y=dataset.pheno.height,
    ...                                       x=dataset.GT.n_alt_alleles(),
    ...                                       covariates=[dataset.pheno.age, dataset.pheno.is_female])

    Warning
    -------
    :func:`.linear_regression` considers the same set of columns (i.e., samples, points)
    for every response variable and row, namely those columns for which **all**
    response variables and covariates are defined. For each row, missing values
    of `x` are mean-imputed over these columns.

    Notes
    -----
    With the default root and `y` a single expression, the following row-indexed
    fields are added.

    - **linreg.n** (:py:data:`.tint32`) -- Number of columns used.
    - **linreg.sum_x** (:py:data:`.tfloat64`) -- Sum of input values `x`.
    - **linreg.y_transpose_x** (:py:data:`.tfloat64`) -- Dot product of response
      vector `y` with the input vector `x`.
    - **linreg.beta** (:py:data:`.tfloat64`) --
      Fit effect coefficient of `x`, :math:`\hat\\beta_1` below.
    - **linreg.standard_error** (:py:data:`.tfloat64`) --
      Estimated standard error, :math:`\widehat{\mathrm{se}}_1`.
    - **linreg.t_stat** (:py:data:`.tfloat64`) -- :math:`t`-statistic, equal to
      :math:`\hat\\beta_1 / \widehat{\mathrm{se}}_1`.
    - **linreg.p_value** (:py:data:`.tfloat64`) -- :math:`p`-value.

    If `y` is a list of expressions, then the last five fields instead have type
    :py:data:`.tarray` of :py:data:`.tfloat64`, with corresponding indexing of
    the list and each array.

    In the statistical genetics example above, the input variable `x` encodes
    genotype as the number of alternate alleles (0, 1, or 2). For each variant
    (row), genotype is tested for association with height controlling for age
    and sex, by fitting the linear regression model:

    .. math::

        \mathrm{height} = \\beta_0 + \\beta_1 \, \mathrm{genotype}
                          + \\beta_2 \, \mathrm{age}
                          + \\beta_3 \, \mathrm{is\_female}
                          + \\varepsilon, \quad \\varepsilon
                        \sim \mathrm{N}(0, \sigma^2)

    Boolean covariates like :math:`\mathrm{is\_female}` are encoded as 1 for
    ``True`` and 0 for ``False``. The null model sets :math:`\\beta_1 = 0`.

    The standard least-squares linear regression model is derived in Section
    3.2 of `The Elements of Statistical Learning, 2nd Edition
    <http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf>`__.
    See equation 3.12 for the t-statistic which follows the t-distribution with
    :math:`n - k - 2` degrees of freedom, under the null hypothesis of no
    effect, with :math:`n` samples and :math:`k` covariates in addition to
    ``x`` and the intercept.

    Parameters
    ----------
    y : :class:`.Float64Expression` or :obj:`list` of :class:`.Float64Expression`
        One or more column-indexed response expressions.
    x : :class:`.Float64Expression`
        Entry-indexed expression for input variable.
    covariates : :obj:`list` of :class:`.Float64Expression`
        List of column-indexed covariate expressions.
    root : :obj:`str`
        Name of resulting row-indexed field.
    block_size : :obj:`int`
        Number of row regressions to perform simultaneously per core. Larger blocks
        require more memory but may improve performance.

    Returns
    -------
    :class:`.MatrixTable`
        Matrix table with regression results in a new row-indexed field.
    """
    mt = matrix_table_source('linear_regression/x', x)
    check_entry_indexed('linear_regression/x', x)

    y_is_list = isinstance(y, list)

    all_exprs = []
    y = wrap_to_list(y)
    for e in y:
        all_exprs.append(e)
        analyze('linear_regression/y', e, mt._col_indices)
    for e in covariates:
        all_exprs.append(e)
        analyze('linear_regression/covariates', e, mt._col_indices)

    # FIXME: remove this logic when annotation is better optimized
    if x in mt._fields_inverse:
        x_field_name = mt._fields_inverse[x]
        fields_to_drop = []
        entry_expr = {}
    else:
        x_field_name = Env.get_uid()
        fields_to_drop = [x_field_name]
        entry_expr = {x_field_name: x}

    y_field_names = list(f'__y{i}' for i in range(len(y)))
    cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))

    fields_to_drop.extend(y_field_names)
    fields_to_drop.extend(cov_field_names)

    mt = mt._annotate_all(col_exprs=dict(**dict(zip(y_field_names, y)),
                                         **dict(zip(cov_field_names, covariates))),
                          entry_exprs=entry_expr)


    jm = Env.hail().methods.LinearRegression.apply(
        mt._jvds,
        jarray(Env.jvm().java.lang.String, y_field_names),
        x_field_name,
        jarray(Env.jvm().java.lang.String, cov_field_names),
        root,
        block_size)

    mt_result = MatrixTable(jm).drop(*fields_to_drop)

    if not y_is_list:
        fields = ['y_transpose_x', 'beta', 'standard_error', 't_stat', 'p_value']
        linreg = mt_result[root]
        mt_result = mt_result.annotate_rows(
            **{root: linreg.annotate(**{f: linreg[f][0] for f in fields})})

    return mt_result


@typecheck(test=enumeration('wald', 'lrt', 'score', 'firth'),
           y=expr_float64,
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           root=str)
def logistic_regression(test, y, x, covariates=(), root='logreg') -> MatrixTable:
    r"""For each row, test an input variable for association with a
    binary response variable using logistic regression.

    Examples
    --------
    Run the logistic regression Wald test per variant using a Boolean
    phenotype and two covariates stored in column-indexed fields:

    >>> ds_result = hl.logistic_regression(
    ...     test='wald',
    ...     y=dataset.pheno.is_case,
    ...     x=dataset.GT.n_alt_alleles(),
    ...     covariates=[dataset.pheno.age, dataset.pheno.is_female])

    Notes
    -----
    This method performs, for each row, a significance test of the input
    variable in predicting a binary (case-control) response variable based on
    the logistic regression model. The response variable type must either be
    numeric (with all present values 0 or 1) or Boolean, in which case true and
    false are coded as 1 and 0, respectively.

    Hail supports the Wald test ('wald'), likelihood ratio test ('lrt'), Rao
    score test ('score'), and Firth test ('firth'). Hail only includes columns
    for which the response variable and all covariates are defined. For each
    row, Hail imputes missing input values as the mean of the non-missing
    values.

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

    The resulting variant annotations depend on the test statistic as shown
    in the tables below.

    ========== ======================= ======= ============================================
    Test       Field                   Type    Value
    ========== ======================= ======= ============================================
    Wald       `logreg.beta`           float64 fit effect coefficient,
                                               :math:`\hat\beta_1`
    Wald       `logreg.standard_error` float64 estimated standard error,
                                               :math:`\widehat{\mathrm{se}}`
    Wald       `logreg.z_stat`         float64 Wald :math:`z`-statistic, equal to
                                               :math:`\hat\beta_1 / \widehat{\mathrm{se}}`
    Wald       `logreg.p_value`        float64 Wald p-value testing :math:`\beta_1 = 0`
    LRT, Firth `logreg.beta`           float64 fit effect coefficient,
                                               :math:`\hat\beta_1`
    LRT, Firth `logreg.chi_sq_stat`    float64 deviance statistic
    LRT, Firth `logreg.p_value`        float64 LRT / Firth p-value testing
                                               :math:`\beta_1 = 0`
    Score      `logreg.chi_sq_stat`    float64 score statistic
    Score      `logreg.p_value`        float64 score p-value testing :math:`\beta_1 = 0`
    ========== ======================= ======= ============================================

    For the Wald and likelihood ratio tests, Hail fits the logistic model for
    each row using Newton iteration and only emits the above annotations
    when the maximum likelihood estimate of the coefficients converges. The
    Firth test uses a modified form of Newton iteration. To help diagnose
    convergence issues, Hail also emits three variant annotations which
    summarize the iterative fitting process:

    ================ ========================= ======= ===============================
    Test             Field                     Type    Value
    ================ ========================= ======= ===============================
    Wald, LRT, Firth `logreg.fit.n_iterations` int32   number of iterations until
                                                       convergence, explosion, or
                                                       reaching the max (25 for
                                                       Wald, LRT; 100 for Firth)
    Wald, LRT, Firth `logreg.fit.converged`    bool    ``True`` if iteration converged
    Wald, LRT, Firth `logreg.fit.exploded`     bool    ``True`` if iteration exploded
    ================ ========================= ======= ===============================

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
    logistic regression is then undefined but convergence may still occur after
    a large number of iterations due to a very flat likelihood surface. In
    testing, we find that such variants produce a secondary bump from 10 to 15
    iterations in the histogram of number of iterations per variant. We also
    find that this faux convergence produces large standard errors and large
    (insignificant) p-values. To not miss such variants, consider using Firth
    logistic regression, linear regression, or group-based tests.

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
    separation by penalizing maximum likelihood estimation by the
    `Jeffrey's invariant prior <https://en.wikipedia.org/wiki/Jeffreys_prior>`__.
    This test is slower, as both the null and full model must be fit per
    variant, and convergence of the modified Newton method is linear rather than
    quadratic. For Firth, 100 iterations are attempted for the null model and,
    if that is successful, for the full model as well. In testing we find 20
    iterations nearly always suffices. If the null model fails to converge, then
    the `logreg.fit` fields reflect the null model; otherwise, they reflect the
    full model.

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

    Those variants that don't vary across the included samples (e.g., all
    genotypes are HomRef) will have missing annotations.

    For Boolean covariate types, ``True`` is coded as 1 and ``False`` as 0. In
    particular, for the sample annotation `fam.is_case` added by importing a FAM
    file with case-control phenotype, case is 1 and control is 0.

    Hail's logistic regression tests correspond to the ``b.wald``, ``b.lrt``,
    and ``b.score`` tests in `EPACTS`_. For each variant, Hail imputes missing
    input values as the mean of non-missing input values, whereas EPACTS
    subsets to those samples with called genotypes. Hence, Hail and EPACTS
    results will currently only agree for variants with no missing genotypes.

    .. _EPACTS: http://genome.sph.umich.edu/wiki/EPACTS#Single_Variant_Tests

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
        List of column-indexed covariate expressions.
    root : :obj:`str`, optional
        Name of resulting row-indexed field.

    Returns
    -------
    :class:`.MatrixTable`
        Matrix table with regression results in a new row-indexed field.
    """
    mt = matrix_table_source('logistic_regression/x', x)
    check_entry_indexed('logistic_regression/x', x)

    analyze('logistic_regression/y', y, mt._col_indices)

    all_exprs = [y]
    for e in covariates:
        all_exprs.append(e)
        analyze('logistic_regression/covariates', e, mt._col_indices)

    # FIXME: remove this logic when annotation is better optimized
    if x in mt._fields_inverse:
        x_field_name = mt._fields_inverse[x]
        fields_to_drop = []
        entry_expr = {}
    else:
        x_field_name = Env.get_uid()
        fields_to_drop = [x_field_name]
        entry_expr = {x_field_name: x}

    y_field_name = '__y'
    cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))

    fields_to_drop.append(y_field_name)
    fields_to_drop.extend(cov_field_names)

    mt = mt._annotate_all(col_exprs=dict(**{y_field_name: y},
                                         **dict(zip(cov_field_names, covariates))),
                          entry_exprs=entry_expr)

    jmt = Env.hail().methods.LogisticRegression.apply(
        mt._jvds,
        test,
        y_field_name,
        x_field_name,
        jarray(Env.jvm().java.lang.String, cov_field_names),
        root)

    return MatrixTable(jmt).drop(*fields_to_drop)


@typecheck(kinship_matrix=KinshipMatrix,
           y=expr_float64,
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           global_root=str,
           row_root=str,
           run_assoc=bool,
           use_ml=bool,
           delta=nullable(numeric),
           sparsity_threshold=numeric,
           n_eigenvectors=nullable(int),
           dropped_variance_fraction=(nullable(float)))
def linear_mixed_regression(kinship_matrix, y, x, covariates=[], global_root="lmmreg_global",
                            row_root="lmmreg", run_assoc=True, use_ml=False, delta=None,
                            sparsity_threshold=1.0, n_eigenvectors=None, dropped_variance_fraction=None) -> MatrixTable:
    r"""Use a kinship-based linear mixed model to estimate the genetic component
    of phenotypic variance (narrow-sense heritability) and optionally test each
    variant for association.

    **We plan to change the interface to this method in Hail 0.2 while maintaining its functionality.**

    .. include:: ../_templates/req_tstring.rst

    Examples
    --------
    Compute a :class:`.KinshipMatrix`, and use it to test variants for
    association using a linear mixed model:

    >>> lmm_ds = hl.read_matrix_table("data/example_lmmreg.vds")
    >>> kinship_matrix = hl.realized_relationship_matrix(lmm_ds.filter_rows(lmm_ds.use_in_kinship)['GT'])
    >>> lmm_ds = hl.linear_mixed_regression(kinship_matrix,
    ...                                     y=lmm_ds.pheno,
    ...                                     x=lmm_ds.GT.n_alt_alleles(),
    ...                                     covariates=[lmm_ds.cov1, lmm_ds.cov2])

    Notes
    -----
    Suppose the variant dataset saved at :file:`data/example_lmmreg.vds` has a
    Boolean variant-indexed field `use_in_kinship` and numeric or Boolean
    sample-indexed fields `pheno`, `cov1`, and `cov2`. Then the
    :func:`.linear_mixed_regression` function in the above example will execute
    the following four steps in order:

    1) filter to samples in given kinship matrix to those for which
       `ds.pheno`, `ds.cov`, and `ds.cov2` are all defined
    2) compute the eigendecomposition :math:`K = USU^T` of the kinship matrix
    3) fit covariate coefficients and variance parameters in the
       sample-covariates-only (global) model using restricted maximum
       likelihood (`REML`_), storing results in a global field under
       `lmmreg_global`
    4) test each variant for association, storing results in a row-indexed
       field under `lmmreg`

    .. _REML: https://en.wikipedia.org/wiki/Restricted_maximum_likelihood

    This plan can be modified as follows:

    - Set `run_assoc` to :obj:`False` to not test any variants for association,
      i.e. skip Step 5.
    - Set `use_ml` to :obj:`True` to use maximum likelihood instead of REML in
      Steps 4 and 5.
    - Set the `delta` argument to manually set the value of :math:`\delta`
      rather that fitting :math:`\delta` in Step 4.
    - Set the `global_root` argument to change the global annotation root in
      Step 4.
    - Set the `row_root` argument to change the variant annotation root in
      Step 5.

    :func:`.linear_mixed_regression` adds 13 global annotations in Step 4.
    These global annotations are stored under the prefix `global_root`, which is
    by default ``lmmreg_global``. The prefix is not displayed in the table
    below.

    .. list-table::
       :header-rows: 1

       * - Field
         - Type
         - Value
       * - `use_ml`
         - bool
         - true if fit by ML, false if fit by REML
       * - `beta`
         - dict<str, float64>
         - map from *intercept* and the given `covariates` expressions to the
           corresponding fit :math:`\beta` coefficients
       * - `sigma_g_squared`
         - float64
         - fit coefficient of genetic variance, :math:`\hat{\sigma}_g^2`
       * - `sigma_e_squared`
         - float64
         - fit coefficient of environmental variance :math:`\hat{\sigma}_e^2`
       * - `delta`
         - float64
         - fit ratio of variance component coefficients, :math:`\hat{\delta}`
       * - `h_squared`
         - float64
         - fit narrow-sense heritability, :math:`\hat{h}^2`
       * - `n_eigenvectors`
         - int32
         - number of eigenvectors of kinship matrix used to fit model
       * - `dropped_variance_fraction`
         - float64
         - specified value of `dropped_variance_fraction`
       * - `eigenvalues`
         - array<float64>
         - all eigenvalues of the kinship matrix in descending order
       * - `fit.standard_error_h_squared`
         - float64
         - standard error of :math:`\hat{h}^2` under asymptotic normal
           approximation
       * - `fit.normalized_likelihood_h_squared`
         - array<float64>
         - likelihood function of :math:`h^2` normalized on the discrete grid
           ``0.01, 0.02, ..., 0.99``. Index ``i`` is the likelihood for
           percentage ``i``.
       * - `fit.max_log_likelihood`
         - float64
         - (restricted) maximum log likelihood corresponding to
           :math:`\hat{\delta}`
       * - `fit.log_delta_grid`
         - array<float64>
         - values of :math:`\mathrm{ln}(\delta)` used in the grid search
       * - `fit.log_likelihood_values`
         - array<float64>
         - (restricted) log likelihood of :math:`y` given :math:`X` and
           :math:`\mathrm{ln}(\delta)` at the (RE)ML fit of :math:`\beta` and
           :math:`\sigma_g^2`

    These global annotations are also added to ``hail.log``, with the ranked
    evals and :math:`\delta` grid with values in .tsv tabular form.  Use
    ``grep 'linear mixed regression' hail.log`` to find the lines just above
    each table.

    If Step 5 is performed, :func:`.linear_mixed_regression` also adds four
    linear regression row fields. These annotations are stored as `row_root`,
    which defaults to ``lmmreg``. Once again, the prefix is not displayed in the
    table.

    +-------------------+---------+------------------------------------------------+
    | Field             | Type    | Value                                          |
    +===================+=========+================================================+
    | `beta`            | float64 | fit genotype coefficient, :math:`\hat\beta_0`  |
    +-------------------+---------+------------------------------------------------+
    | `sigma_g_squared` | float64 | fit coefficient of genetic variance component, |
    |                   |         | :math:`\hat{\sigma}_g^2`                       |
    +-------------------+---------+------------------------------------------------+
    | `chi_sq_stat`     | float64 | :math:`\chi^2` statistic of the likelihood     |
    |                   |         | ratio test                                     |
    +-------------------+---------+------------------------------------------------+
    | `p_value`         | float64 | :math:`p`-value                                |
    +-------------------+---------+------------------------------------------------+

    Those variants that don't vary across the included samples (e.g., all
    genotypes are HomRef) will have missing annotations.

    **Performance**

    Hail's initial version of :func:`.linear_mixed_regression` scales beyond
    15k samples and to an essentially unbounded number of variants, making it
    particularly well-suited to modern sequencing studies and complementary to
    tools designed for SNP arrays. Analysts have used
    :func:`.linear_mixed_regression` in research to compute kinship from 100k
    common variants and test 32 million non-rare variants on 8k whole genomes in
    about 10 minutes on `Google cloud`_.

    .. _Google cloud:
        http://discuss.hail.is/t/using-hail-on-the-google-cloud-platform/80

    While :func:`.linear_mixed_regression` computes the kinship matrix
    :math:`K` using distributed matrix multiplication (Step 2), the full
    `eigendecomposition`_ (Step 3) is currently run on a single core of master
    using the `LAPACK routine DSYEVD`_, which we empirically find to be the most
    performant of the four available routines; laptop performance plots showing
    cubic complexity in :math:`n` are available `here
    <https://github.com/hail-is/hail/pull/906>`__. On Google cloud,
    eigendecomposition takes about 2 seconds for 2535 sampes and 1 minute for
    8185 samples. If you see worse performance, check that LAPACK natives are
    being properly loaded (see "BLAS and LAPACK" in Getting Started).

    .. _LAPACK routine DSYEVD:
        http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga694ddc6e5527b6223748e3462013d867.html

    .. _eigendecomposition:
        https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix

    Given the eigendecomposition, fitting the global model (Step 4) takes on
    the order of a few seconds on master. Association testing (Step 5) is fully
    distributed by variant with per-variant time complexity that is completely
    independent of the number of sample covariates and dominated by
    multiplication of the genotype vector :math:`v` by the matrix of
    eigenvectors :math:`U^T` as described below, which we accelerate with a
    sparse representation of :math:`v`.  The matrix :math:`U^T` has size about
    :math:`8n^2` bytes and is currently broadcast to each Spark executor. For
    example, with 15k samples, storing :math:`U^T` consumes about 3.6GB of
    memory on a 16-core worker node with two 8-core executors. So for large
    :math:`n`, we recommend using a high-memory configuration such as
    *highmem* workers.

    **Linear mixed model**

    :func:`.linear_mixed_regression` estimates the genetic proportion of
    residual phenotypic variance (narrow-sense heritability) under a
    kinship-based linear mixed model, and then optionally tests each variant for
    association using the likelihood ratio test. Inference is exact.

    We first describe the sample-covariates-only model used to estimate
    heritability, which we simply refer to as the *global model*. With
    :math:`n` samples and :math:`c` sample covariates, we define:

    - :math:`y = n \times 1` vector of phenotypes
    - :math:`X = n \times c` matrix of sample covariates and intercept column
      of ones
    - :math:`K = n \times n` kinship matrix
    - :math:`I = n \times n` identity matrix
    - :math:`\beta = c \times 1` vector of covariate coefficients
    - :math:`\sigma_g^2 =` coefficient of genetic variance component :math:`K`
    - :math:`\sigma_e^2 =` coefficient of environmental variance component
      :math:`I`
    - :math:`\delta = \frac{\sigma_e^2}{\sigma_g^2} =` ratio of environmental
      and genetic variance component coefficients
    - :math:`h^2 = \frac{\sigma_g^2}{\sigma_g^2 + \sigma_e^2} = \frac{1}{1 + \delta} =`
      genetic proportion of residual phenotypic variance

    Under a linear mixed model, :math:`y` is sampled from the
    :math:`n`-dimensional `multivariate normal distribution`_ with mean
    :math:`X \beta` and variance components that are scalar multiples of
    :math:`K` and :math:`I`:

    .. math::

      y \sim \mathrm{N}\left(X\beta, \sigma_g^2 K + \sigma_e^2 I\right)

    .. _multivariate normal distribution:
       https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    Thus the model posits that the residuals :math:`y_i - X_{i,:}\beta` and
    :math:`y_j - X_{j,:}\beta` have covariance :math:`\sigma_g^2 K_{ij}` and
    approximate correlation :math:`h^2 K_{ij}`. Informally: phenotype residuals
    are correlated as the product of overall heritability and pairwise kinship.
    By contrast, standard (unmixed) linear regression is equivalent to fixing
    :math:`\sigma_2` (equivalently, :math:`h^2`) at 0 above, so that all
    phenotype residuals are independent.

    **Caution:** while it is tempting to interpret :math:`h^2` as the
    `narrow-sense heritability`_ of the phenotype alone, note that its value
    depends not only the phenotype and genetic data, but also on the choice of
    sample covariates.

    .. _narrow-sense heritability: https://en.wikipedia.org/wiki/Heritability#Definition

    **Fitting the global model**

    The core algorithm is essentially a distributed implementation of the
    spectral approach taken in `FastLMM`_. Let :math:`K = USU^T` be the
    `eigendecomposition`_ of the real symmetric matrix :math:`K`. That is:

    - :math:`U = n \times n` orthonormal matrix whose columns are the
      eigenvectors of :math:`K`
    - :math:`S = n \times n` diagonal matrix of eigenvalues of :math:`K` in
      descending order. :math:`S_{ii}` is the eigenvalue of eigenvector
      :math:`U_{:,i}`
    - :math:`U^T = n \times n` orthonormal matrix, the transpose (and inverse)
      of :math:`U`

    .. _FastLMM: https://www.microsoft.com/en-us/research/project/fastlmm/

    A bit of matrix algebra on the multivariate normal density shows that the
    linear mixed model above is mathematically equivalent to the model

    .. math::

      U^Ty \sim \mathrm{N}\left(U^TX\beta, \sigma_g^2 (S + \delta I)\right)

    for which the covariance is diagonal (e.g., unmixed). That is, rotating the
    phenotype vector (:math:`y`) and covariate vectors (columns of :math:`X`)
    in :math:`\mathbb{R}^n` by :math:`U^T` transforms the model to one with
    independent residuals. For any particular value of :math:`\delta`, the
    restricted maximum likelihood (REML) solution for the latter model can be
    solved exactly in time complexity that is linear rather than cubic in
    :math:`n`.  In particular, having rotated, we can run a very efficient
    1-dimensional optimization procedure over :math:`\delta` to find the REML
    estimate :math:`(\hat{\delta}, \hat{\beta}, \hat{\sigma}_g^2)` of the
    triple :math:`(\delta, \beta, \sigma_g^2)`, which in turn determines
    :math:`\hat{\sigma}_e^2` and :math:`\hat{h}^2`.

    We first compute the maximum log likelihood on a :math:`\delta`-grid that
    is uniform on the log scale, with :math:`\mathrm{ln}(\delta)` running from
    -8 to 8 by 0.01, corresponding to :math:`h^2` decreasing from 0.9995 to
    0.0005. If :math:`h^2` is maximized at the lower boundary then standard
    linear regression would be more appropriate and Hail will exit; more
    generally, consider using standard linear regression when :math:`\hat{h}^2`
    is very small. A maximum at the upper boundary is highly suspicious and
    will also cause Hail to exit. In any case, the log file records the table
    of grid values for further inspection, beginning under the info line
    containing "linear mixed regression: table of delta".

    If the optimal grid point falls in the interior of the grid as expected,
    we then use `Brent's method`_ to find the precise location of the maximum
    over the same range, with initial guess given by the optimal grid point and
    a tolerance on :math:`\mathrm{ln}(\delta)` of 1e-6. If this location
    differs from the optimal grid point by more than 0.01, a warning will be
    displayed and logged, and one would be wise to investigate by plotting the
    values over the grid.

    .. _Brent's method: https://en.wikipedia.org/wiki/Brent%27s_method

    Note that :math:`h^2` is related to :math:`\mathrm{ln}(\delta)` through the
    `sigmoid function`_. More precisely,

    .. math::

      h^2 = 1 - \mathrm{sigmoid}(\mathrm{ln}(\delta))
          = \mathrm{sigmoid}(-\mathrm{ln}(\delta))

    .. _sigmoid function: https://en.wikipedia.org/wiki/Sigmoid_function

    Hence one can change variables to extract a high-resolution discretization
    of the likelihood function of :math:`h^2` over :math:`[0, 1]` at the
    corresponding REML estimators for :math:`\beta` and :math:`\sigma_g^2`, as
    well as integrate over the normalized likelihood function using
    `change of variables`_ and the `sigmoid differential equation`_.

    .. _change of variables: https://en.wikipedia.org/wiki/Integration_by_substitution
    .. _sigmoid differential equation: https://en.wikipedia.org/wiki/Sigmoid_function#Properties

    For convenience, `lmmreg.fit.normalized_likelihood_h_squared` records the
    the likelihood function of :math:`h^2` normalized over the discrete grid
    :math:`0.01, 0.02, \ldots, 0.98, 0.99`. The length of the array is 101 so
    that index ``i`` contains the likelihood at percentage ``i``. The values at
    indices 0 and 100 are left undefined.

    By the theory of maximum likelihood estimation, this normalized likelihood
    function is approximately normally distributed near the maximum likelihood
    estimate. So we estimate the standard error of the estimator of :math:`h^2`
    as follows. Let :math:`x_2` be the maximum likelihood estimate of
    :math:`h^2` and let :math:`x_ 1` and :math:`x_3` be just to the left and
    right of :math:`x_2`. Let :math:`y_1`, :math:`y_2`, and :math:`y_3` be the
    corresponding values of the (unnormalized) log likelihood function. Setting
    equal the leading coefficient of the unique parabola through these points
    (as given by Lagrange interpolation) and the leading coefficient of the log
    of the normal distribution, we have:

    .. math::

      \frac{x_3 (y_2 - y_1) + x_2 (y_1 - y_3) + x_1 (y_3 - y_2))}
           {(x_2 - x_1)(x_1 - x_3)(x_3 - x_2)} = -\frac{1}{2 \sigma^2}

    The standard error :math:`\hat{\sigma}` is then estimated by solving for
    :math:`\sigma`.

    Note that the mean and standard deviation of the (discretized or
    continuous) distribution held in
    `lmmreg.fit.normalized_likelihood_h_squared` will not coincide with
    :math:`\hat{h}^2` and :math:`\hat{\sigma}`, since this distribution only
    becomes normal in the infinite sample limit. One can visually assess
    normality by plotting this distribution against a normal distribution with
    the same mean and standard deviation, or use this distribution to
    approximate credible intervals under a flat prior on :math:`h^2`.

    **Testing each variant for association**

    Fixing a single variant, we define:

    - :math:`v = n \times 1` input vector, with missing values imputed as the
      mean of the non-missing values
    - :math:`X_v = \left[v | X \right] = n \times (1 + c)` matrix concatenating
      :math:`v` and :math:`X`
    - :math:`\beta_v = (\beta^0_v, \beta^1_v, \ldots, \beta^c_v) = (1 + c) \times 1`
      vector of covariate coefficients

    Fixing :math:`\delta` at the global REML estimate :math:`\hat{\delta}`, we
    find the REML estimate :math:`(\hat{\beta}_v, \hat{\sigma}_{g, v}^2)` via
    rotation of the model

    .. math::

      y \sim \mathrm{N}\left(X_v\beta_v, \sigma_{g,v}^2 (K + \hat{\delta} I)\right)

    Note that the only new rotation to compute here is :math:`U^T v`.

    To test the null hypothesis that the genotype coefficient :math:`\beta^0_v`
    is zero, we consider the restricted model with parameters
    :math:`((0, \beta^1_v, \ldots, \beta^c_v), \sigma_{g,v}^2)` within the full
    model with parameters
    :math:`(\beta^0_v, \beta^1_v, \ldots, \beta^c_v), \sigma_{g_v}^2)`, with
    :math:`\delta` fixed at :math:`\hat\delta` in both. The latter fit is
    simply that of the global model,
    :math:`((0, \hat{\beta}^1, \ldots, \hat{\beta}^c), \hat{\sigma}_g^2)`. The
    likelihood ratio test statistic is given by

    .. math::

      \chi^2 = n \, \mathrm{ln}\left(\frac{\hat{\sigma}^2_g}{\hat{\sigma}_{g,v}^2}\right)

    and follows a chi-squared distribution with one degree of freedom. Here the
    ratio :math:`\hat{\sigma}^2_g / \hat{\sigma}_{g,v}^2` captures the degree
    to which adding the variant :math:`v` to the global model reduces the
    residual phenotypic variance.

    **Kinship Matrix**

    FastLMM uses the Realized Relationship Matrix (RRM) for kinship. This can
    be computed with :func:`.rrm`. However, any instance of
    :class:`.KinshipMatrix` may be used, so long as
    :meth:`~.KinshipMatrix.sample_list` contains the complete samples of the
    caller variant dataset in the same order.

    **Low-rank approximation of kinship for improved performance**

    :func:`.linear_mixed_regression` can implicitly use a low-rank
    approximation of the kinship matrix to more rapidly fit delta and the
    statistics for each variant. The computational complexity per variant is
    proportional to the number of eigenvectors used. This number can be
    specified in two ways. Specify the parameter `n_eigenvectors` to use only the
    top `n_eigenvectors` eigenvectors. Alternatively, specify
    `dropped_variance_fraction` to use as many eigenvectors as necessary to
    capture all but at most this fraction of the sample variance (also known as
    the trace, or the sum of the eigenvalues). For example, setting
    `dropped_variance_fraction` to 0.01 will use the minimal number of
    eigenvectors to account for 99% of the sample variance. Specifying both
    parameters will apply the more stringent (fewest eigenvectors) of the two.

    **Further background**

    For the history and mathematics of linear mixed models in genetics,
    including `FastLMM`_, see `Christoph Lippert's PhD thesis
    <https://publikationen.uni-tuebingen.de/xmlui/bitstream/handle/10900/50003/pdf/thesis_komplett.pdf>`__.
    For an investigation of various approaches to defining kinship, see
    `Comparison of Methods to Account for Relatedness in Genome-Wide Association
    Studies with Family-Based Data
    <http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004445>`__.


    Parameters
    ----------
    kinship_matrix : :class:`.KinshipMatrix`
        Kinship matrix to be used.
    y : :class:`.Float64Expression`
        Column-indexed response expression.
    x : :class:`.Float64Expression`
        Entry-indexed expression for input variable.
    covariates : :obj:`list` of :class:`.Float64Expression`
        List of column-indexed covariate expressions.
    global_root : :obj:`str`
        Global field root.
    row_root : :obj:`str`
        Row-indexed field root.
    run_assoc : :obj:`bool`
        If true, run association testing in addition to fitting the global model.
    use_ml : :obj:`bool`
        Use ML instead of REML throughout.
    delta : :obj:`float` or :obj:`None`
        Fixed delta value to use in the global model, overrides fitting delta.
    sparsity_threshold : :obj:`float`
        Genotype vector sparsity at or below which to use sparse genotype
        vector in rotation (advanced).
    n_eigenvectors : :obj:`int`
        Number of eigenvectors of the kinship matrix used to fit the model.
    dropped_variance_fraction : :obj:`float`
        Upper bound on fraction of sample variance lost by dropping
        eigenvectors with small eigenvalues.

    Returns
    -------
    :class:`.MatrixTable`
        Matrix table with regression results in new global and (optionally) row-indexed fields.
    """

    mt = matrix_table_source('linear_mixed_regression/x', x)
    check_entry_indexed('linear_mixed_regression/x', x)

    analyze('linear_mixed_regression/y', y, mt._col_indices)

    all_exprs = [y]
    for e in covariates:
        all_exprs.append(e)
        analyze('linear_mixed_regression/covariates', e, mt._col_indices)

    # FIXME: remove this logic when annotation is better optimized
    if x in mt._fields_inverse:
        x_field_name = mt._fields_inverse[x]
        fields_to_drop = []
        entry_expr = {}
    else:
        x_field_name = Env.get_uid()
        fields_to_drop = [x_field_name]
        entry_expr = {x_field_name: x}

    y_field_name = '__y'
    cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))

    fields_to_drop.append(y_field_name)
    fields_to_drop.extend(cov_field_names)

    mt = mt._annotate_all(col_exprs=dict(**{y_field_name: y},
                                         **dict(zip(cov_field_names, covariates))),
                          entry_exprs=entry_expr)

    jmt = Env.hail().methods.LinearMixedRegression.apply(
        mt._jvds,
        kinship_matrix._jkm,
        y_field_name,
        x_field_name,
        jarray(Env.jvm().java.lang.String, cov_field_names),
        use_ml,
        global_root,
        row_root,
        run_assoc,
        joption(delta),
        sparsity_threshold,
        joption(n_eigenvectors),
        joption(dropped_variance_fraction))

    return MatrixTable(jmt).drop(*fields_to_drop)


@typecheck(key_expr=expr_any,
           weight_expr=expr_float64,
           y=expr_float64,
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           logistic=bool,
           max_size=int,
           accuracy=numeric,
           iterations=int)
def skat(key_expr, weight_expr, y, x, covariates=[], logistic=False,
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
    ...                      covariates=[burden_ds.burden.cov1, burden_ds.burden.cov2])

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

    Notes
    -----

    This method provides a scalable implementation of the score-based
    variance-component test originally described in
    `Rare-Variant Association Testing for Sequencing Data with the Sequence Kernel Association Test
    <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3135811/>`__.

    The test is run on columns with `y` and all `covariates` non-missing. For
    each row, missing input (`x`) values are imputed as the mean of all
    non-missing input values.

    Row weights must be non-negative. Rows with missing weights are ignored. In
    the R package ``skat``---which assumes rows are variants---default weights
    are given by evaluating the Beta(1, 25) density at the minor allele
    frequency. To replicate these weights in Hail using alternate allele
    frequencies stored in a row-indexed field `AF`, one can use the expression:

    >>> hl.dbeta(hl.min(ds2.AF), 1.0, 25.0) ** 2

    In the logistic case, the response `y` must either be numeric (with all
    present values 0 or 1) or Boolean, in which case true and false are coded
    as 1 and 0, respectively.

    The resulting :class:`.Table` provides the group's key, the size (number of
    rows) in the group, the variance component score `q_stat`, the SKAT
    p-value, and a fault flag. For the toy example above, the table has the
    form:

    +-------+------+--------+---------+-------+
    |  key  | size | q_stat | p_value | fault |
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
        mt._jvds,
        key_field_name,
        weight_field_name,
        y_field_name,
        x_field_name,
        jarray(Env.jvm().java.lang.String, cov_field_names),
        logistic,
        max_size,
        accuracy,
        iterations)

    return Table(jt)


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

    r = Env.hail().methods.PCA.apply(mt._jvds, field, k, compute_loadings)
    scores = Table(Env.hail().methods.PCA.scoresTable(mt._jvds, r._2()))
    loadings = from_option(r._3())
    if loadings:
        loadings = Table(loadings)
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
    """Compute relatedness estimates between individuals using a variant of the
    PC-Relate method.

    .. include:: ../_templates/experimental.rst
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

      \\widehat{\\phi_{ij}} :=
        \\frac{1}{|S_{ij}|}
        \\sum_{s \\in S_{ij}}
          \\frac{(g_{is} - 2 p_s) (g_{js} - 2 p_s)}
                {4 \\sum_{s \\in S_{ij} p_s (1 - p_s)}}

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

     - :math:`g_{is} \\in {0, 1, 2}` be the number of alternate alleles that
       individual :math:`i` has at genetic locus :math:`s`

     - :math:`\\widehat{\\mu_{is}} \\in [0, 1]` be the individual-specific allele
       frequency for individual :math:`i` at genetic locus :math:`s`

     - :math:`{\\widehat{\\sigma^2_{is}}} := \\widehat{\\mu_{is}} (1 - \\widehat{\\mu_{is}})`,
       the binomial variance of :math:`\\widehat{\\mu_{is}}`

     - :math:`\\widehat{\\sigma_{is}} := \\sqrt{\\widehat{\\sigma^2_{is}}}`,
       the binomial standard deviation of :math:`\\widehat{\\mu_{is}}`

     - :math:`\\text{IBS}^{(0)}_{ij} := \\sum_{s \\in S_{ij}} \\mathbb{1}_{||g_{is} - g_{js} = 2||}`,
       the number of genetic loci at which individuals :math:`i` and :math:`j`
       share no alleles

     - :math:`\\widehat{f_i} := 2 \\widehat{\\phi_{ii}} - 1`, the inbreeding
       coefficient for individual :math:`i`

     - :math:`g^D_{is}` be a dominance encoding of the genotype matrix, and
       :math:`X_{is}` be a normalized dominance-coded genotype matrix

    .. math::

        g^D_{is} :=
          \\begin{cases}
            \\widehat{\\mu_{is}}     & g_{is} = 0 \\\\
            0                        & g_{is} = 1 \\\\
            1 - \\widehat{\\mu_{is}} & g_{is} = 2
          \\end{cases}

        X_{is} := g^D_{is} - \\widehat{\\sigma^2_{is}} (1 - \\widehat{f_i})

    The estimator for kinship is given by:

    .. math::

      \\widehat{\phi_{ij}} :=
        \\frac{\sum_{s \\in S_{ij}}(g - 2 \\mu)_{is} (g - 2 \\mu)_{js}}
              {4 * \\sum_{s \\in S_{ij}}
                            \\widehat{\\sigma_{is}} \\widehat{\\sigma_{js}}}

    The estimator for identity-by-descent two is given by:

    .. math::

      \\widehat{k^{(2)}_{ij}} :=
        \\frac{\\sum_{s \\in S_{ij}}X_{is} X_{js}}{\sum_{s \\in S_{ij}}
          \\widehat{\\sigma^2_{is}} \\widehat{\\sigma^2_{js}}}

    The estimator for identity-by-descent zero is given by:

    .. math::

      \\widehat{k^{(0)}_{ij}} :=
        \\begin{cases}
          \\frac{\\text{IBS}^{(0)}_{ij}}
                {\\sum_{s \\in S_{ij}}
                       \\widehat{\\mu_{is}}^2(1 - \\widehat{\\mu_{js}})^2
                       + (1 - \\widehat{\\mu_{is}})^2\\widehat{\\mu_{js}}^2}
            & \\widehat{\\phi_{ij}} > 2^{-5/2} \\\\
          1 - 4 \\widehat{\\phi_{ij}} + k^{(2)}_{ij}
            & \\widehat{\\phi_{ij}} \\le 2^{-5/2}
        \\end{cases}

    The estimator for identity-by-descent one is given by:

    .. math::

      \\widehat{k^{(1)}_{ij}} :=
        1 - \\widehat{k^{(2)}_{ij}} - \\widehat{k^{(0)}_{ij}}

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

    Under the PC-Relate model, kinship, :math:`\\phi_{ij}`, ranges from 0 to
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
     - `kin` (:py:data:`.tfloat64`) -- Kinship estimate, :math:`\\widehat{\\phi_{ij}}`.
     - `ibd2` (:py:data:`.tfloat64`) -- IBD2 estimate, :math:`\\widehat{k^{(2)}_{ij}}`.
     - `ibd0` (:py:data:`.tfloat64`) -- IBD0 estimate, :math:`\\widehat{k^{(0)}_{ij}}`.
     - `ibd1` (:py:data:`.tfloat64`) -- IBD1 estimate, :math:`\\widehat{k^{(1)}_{ij}}`.

    Here ``col_key`` refers to the column key of the source matrix table,
    and ``col_key.dtype`` is a struct containing the column key fields.

    There is one row for each pair of distinct samples (columns), where `i`
    corresponds to the column of smaller column index. In particular, if the
    same column key value exists for :math:`n` columns, then the resulting
    table will have :math:`\\binom{n-1}{2}` rows with both key fields equal to
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

    ht = Table(scala_object(Env.hail().methods, 'PCRelate')
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

    col_keys = hl.literal(mt.col_key.collect(), dtype=tarray(mt.col_key.dtype))
    return ht.key_by(i=col_keys[ht.i], j=col_keys[ht.j])

class SplitMulti(object):
    """Split multiallelic variants.

    Example
    -------

    :func:`.split_multi_hts`, which splits
    multiallelic variants for the HTS genotype schema and updates
    the genotype annotations by downcoding the genotype, is
    implemented as:

    >>> sm = hl.SplitMulti(ds)
    >>> pl = hl.or_missing(
    ...      hl.is_defined(ds.PL),
    ...      (hl.range(0, 3).map(lambda i: hl.min(hl.range(0, hl.len(ds.PL))
    ...                     .filter(lambda j: hl.downcode(hl.unphased_diploid_gt_index_call(j), sm.a_index()) == hl.unphased_diploid_gt_index_call(i))
    ...                     .map(lambda j: ds.PL[j])))))
    >>> sm.update_rows(a_index=sm.a_index(), was_split=sm.was_split())
    >>> sm.update_entries(
    ...     GT=hl.downcode(ds.GT, sm.a_index()),
    ...     AD=hl.or_missing(hl.is_defined(ds.AD),
    ...                     [hl.sum(ds.AD) - ds.AD[sm.a_index()], ds.AD[sm.a_index()]]),
    ...     DP=ds.DP,
    ...     PL=pl,
    ...     GQ=hl.gq_from_pl(pl))
    >>> split_ds = sm.result()

    Warning
    -------
    Any entry and row fields that are not updated will be copied (unchanged)
    for each split variant.
    """

    @typecheck_method(ds=MatrixTable,
                      keep_star=bool,
                      left_aligned=bool)
    def __init__(self, ds, keep_star=False, left_aligned=False):
        """
        Parameters
        ----------
        ds : :class:`.MatrixTable`
            An unsplit dataset.
        keep_star : :obj:`bool`
            Do not filter out * alleles.
        left_aligned : :obj:`bool`
            If ``True``, variants are assumed to be left aligned and have unique
            loci. This avoids a shuffle. If the assumption is violated, an error
            is generated.

        Returns
        -------
        :class:`.SplitMulti`
        """
        self._ds = ds
        self._keep_star = keep_star
        self._left_aligned = left_aligned
        self._entry_fields = None
        self._row_fields = None

    def new_locus(self):
        """The new, split variant locus.

        Returns
        -------
        :class:`.LocusExpression`
        """
        return construct_reference(
            "newLocus", type=self._ds.locus.dtype, indices=self._ds._row_indices)

    def new_alleles(self):
        """The new, split variant alleles.

        Returns
        -------
        :class:`.ArrayStringExpression`
        """
        return construct_reference(
            "newAlleles", type=tarray(tstr), indices=self._ds._row_indices)

    def a_index(self):
        """The index of the input allele to the output variant.

        Returns
        -------
        :class:`.Expression` of type :py:data:`.tint32`
        """
        return construct_reference(
            "aIndex", type=tint32, indices=self._ds._row_indices)

    def was_split(self):
        """``True`` if the original variant was multiallelic.

        Returns
        -------
        :class:`.BooleanExpression`
        """
        return construct_reference(
            "wasSplit", type=tbool, indices=self._ds._row_indices)

    def update_rows(self, **kwargs):
        """Set the row field updates for this SplitMulti object.

        Note
        ----
        May only be called once.
        """
        if self._row_fields is None:
            self._row_fields = kwargs
        else:
            raise FatalError("You may only call update_rows once")

    def update_entries(self, **kwargs):
        """Set the entry field updates for this SplitMulti object.

        Note
        ----
        May only be called once.
        """
        if self._entry_fields is None:
            self._entry_fields = kwargs
        else:
            raise FatalError("You may only call update_entries once")

    def result(self):
        """Split the dataset.

        Returns
        -------
        :class:`.MatrixTable`
            A split dataset.
        """

        if not self._row_fields:
            self._row_fields = {}
        if not self._entry_fields:
            self._entry_fields = {}

        unmod_row_fields = set(self._ds.row) - set(self._row_fields) - {'locus', 'alleles', 'a_index', 'was_split'}
        unmod_entry_fields = set(self._ds.entry) - set(self._entry_fields)

        for name, fds in [('row', unmod_row_fields), ('entry', unmod_entry_fields)]:
            if fds:
                field = hl.utils.misc.plural('field', len(fds))
                word = hl.utils.misc.plural('was', len(fds), 'were')
                fds = ', '.join(["'" + f + "'" for f in fds])
                warn(f"SplitMulti: The following {name} {field} {word} not updated: {fds}. " \
                      "Data will be copied (unchanged) for each split variant.")

        base, _ = self._ds._process_joins(*itertools.chain(
            self._row_fields.values(), self._entry_fields.values()))

        annotate_rows = ','.join(['va.`{}` = {}'.format(k, v._ast.to_hql())
                                  for k, v in self._row_fields.items()])
        annotate_entries = ','.join(['g.`{}` = {}'.format(k, v._ast.to_hql())
                                     for k, v in self._entry_fields.items()])

        jvds = scala_object(Env.hail().methods, 'SplitMulti').apply(
            self._ds._jvds,
            annotate_rows,
            annotate_entries,
            self._keep_star,
            self._left_aligned)
        return MatrixTable(jvds)


@typecheck(ds=MatrixTable,
           keep_star=bool,
           left_aligned=bool)
def split_multi_hts(ds, keep_star=False, left_aligned=False) -> MatrixTable:
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
        PID: str,
      }

    For other entry fields, use :class:`.SplitMulti`.

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

    Each multiallelic `GT` field is downcoded once for each alternate allele. A
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

     - `was_split` (*Boolean*) -- ``True`` if this variant was originally
       multiallelic, otherwise ``False``.

     - `a_index` (*Int*) -- The original index of this alternate allele in the
       multiallelic representation (NB: 1 is the first alternate allele or the
       only alternate allele in a biallelic variant). For example, 1:100:A:T,C
       splits into two variants: 1:100:A:T with ``a_index = 1`` and 1:100:A:C
       with ``a_index = 2``.

    Parameters
    ----------
    keep_star : :obj:`bool`
        Do not filter out * alleles.
    left_aligned : :obj:`bool`
        If ``True``, variants are assumed to be left
        aligned and have unique loci. This avoids a shuffle. If the assumption
        is violated, an error is generated.

    Returns
    -------
    :class:`.MatrixTable`
        A biallelic variant dataset.

    """

    entry_fields = set(ds.entry)

    update_entries_expression = {}
    sm = SplitMulti(ds, keep_star=keep_star, left_aligned=left_aligned)

    if 'GT' in entry_fields:
        update_entries_expression['GT'] = hl.downcode(ds.GT, sm.a_index())
    if 'DP' in entry_fields:
        update_entries_expression['DP'] = ds.DP
    if 'AD' in entry_fields:
        update_entries_expression['AD'] = hl.or_missing(hl.is_defined(ds.AD),
                                                        [hl.sum(ds.AD) - ds.AD[sm.a_index()], ds.AD[sm.a_index()]])
    if 'PL' in entry_fields:
        pl = hl.or_missing(
            hl.is_defined(ds.PL),
            (hl.range(0, 3).map(lambda i:
                                hl.min((hl.range(0, hl.triangle(ds.alleles.length()))
                                        .filter(lambda j: hl.downcode(hl.unphased_diploid_gt_index_call(j),
                                                                      sm.a_index()) == hl.unphased_diploid_gt_index_call(i)
                                                ).map(lambda j: ds.PL[j]))))))
        update_entries_expression['PL'] = pl
        if 'GQ' in entry_fields:
            update_entries_expression['GQ'] = hl.gq_from_pl(pl)
    else:
        if 'GQ' in entry_fields:
            update_entries_expression['GQ'] = ds.GQ

    if 'PGT' in entry_fields:
        update_entries_expression['PGT'] = hl.downcode(ds.PGT, sm.a_index())
    if 'PID' in entry_fields:
        update_entries_expression['PID'] = ds.PID

    sm.update_rows(a_index=sm.a_index(), was_split=sm.was_split())
    sm.update_entries(**update_entries_expression)
    return sm.result()


@typecheck(call_expr=expr_call)
def genetic_relatedness_matrix(call_expr) -> KinshipMatrix:
    """Compute the genetic relatedness matrix (GRM).

    Examples
    --------
    
    >>> grm = hl.genetic_relatedness_matrix(dataset.GT)

    Notes
    -----
    The genetic relationship matrix (GRM) :math:`G` encodes genetic correlation
    between each pair of samples. It is defined by :math:`G = MM^T` where
    :math:`M` is a standardized version of the genotype matrix, computed as
    follows. Let :math:`C` be the :math:`n \\times m` matrix of raw genotypes
    in the variant dataset, with rows indexed by :math:`n` samples and columns
    indexed by :math:`m` bialellic autosomal variants; :math:`C_{ij}` is the
    number of alternate alleles of variant :math:`j` carried by sample
    :math:`i`, which can be 0, 1, 2, or missing. For each variant :math:`j`,
    the sample alternate allele frequency :math:`p_j` is computed as half the
    mean of the non-missing entries of column :math:`j`. Entries of :math:`M`
    are then mean-centered and variance-normalized as

    .. math::

        M_{ij} = \\frac{C_{ij}-2p_j}{\sqrt{2p_j(1-p_j)m}},

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

        G_{ik} = \\frac{1}{m} \\sum_{j=1}^m \\frac{(C_{ij}-2p_j)(C_{kj}-2p_j)}{2 p_j (1-p_j)}

    Note that variants for which the alternate allele frequency is zero or one are not
    normalizable, and therefore removed prior to calculating the GRM.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression.

    Returns
    -------
    :class:`.genetics.KinshipMatrix`
        Genetic relatedness matrix for all samples.
    """
    mt = matrix_table_source('genetic_relatedness_matrix/call_expr', call_expr)
    require_col_key_str(mt, 'genetic_relatedness_matrix')

    col_keys = mt.cols().select()

    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())
    mt = mt.annotate_rows(__AC=agg.sum(mt.__gt),
                          __n_called=agg.count_where(hl.is_defined(mt.__gt)))
    mt = mt.filter_rows((mt.__AC > 0) & (mt.__AC < 2 * mt.__n_called))
    mt = mt.persist()

    n_variants = mt.count_rows()
    if n_variants == 0:
        raise FatalError("Cannot run GRM: found 0 variants after filtering out monomorphic sites.")
    info("Computing GRM using {} variants.".format(n_variants))

    mt = mt.annotate_rows(__mean_gt=mt.__AC / mt.__n_called)
    mt = mt.annotate_rows(
        __hwe_scaled_std_dev=hl.sqrt(mt.__mean_gt * (2 - mt.__mean_gt) * n_variants / 2))

    normalized_gt = hl.or_else((mt.__gt - mt.__mean_gt) / mt.__hwe_scaled_std_dev, 0.0)

    bm = BlockMatrix.from_entry_expr(normalized_gt)
    mt.unpersist()
    grm = bm.T @ bm

    return KinshipMatrix._from_block_matrix(grm,
                                            col_keys,
                                            n_variants)


@typecheck(call_expr=expr_call)
def realized_relationship_matrix(call_expr) -> KinshipMatrix:
    """Computes the realized relationship matrix (RRM).

    Examples
    --------

    >>> rrm = hl.realized_relationship_matrix(dataset.GT)

    Notes
    -----
    The realized relationship matrix (RRM) is defined as follows. Consider the
    :math:`n \\times m` matrix :math:`C` of raw genotypes, with rows indexed by
    :math:`n` samples and columns indexed by the :math:`m` bialellic autosomal
    variants; :math:`C_{ij}` is the number of alternate alleles of variant
    :math:`j` carried by sample :math:`i`, which can be 0, 1, 2, or missing. For
    each variant :math:`j`, the sample alternate allele frequency :math:`p_j` is
    computed as half the mean of the non-missing entries of column :math:`j`.
    Entries of :math:`M` are then mean-centered and variance-normalized as

    .. math::

        M_{ij} =
          \\frac{C_{ij}-2p_j}
                {\sqrt{\\frac{m}{n} \\sum_{k=1}^n (C_{ij}-2p_j)^2}},

    with :math:`M_{ij} = 0` for :math:`C_{ij}` missing (i.e. mean genotype
    imputation). This scaling normalizes each variant column to have empirical
    variance :math:`1/m`, which gives each sample row approximately unit total
    variance (assuming linkage equilibrium) and yields the :math:`n \\times n`
    sample correlation or realized relationship matrix (RRM) :math:`K` as simply

    .. math::

        K = MM^T

    Note that the only difference between the realized relationship matrix and
    the genetic relatedness matrix (GRM) used in
    :func:`.realized_relationship_matrix` is the variant (column) normalization:
    where RRM uses empirical variance, GRM uses expected variance under
    Hardy-Weinberg Equilibrium.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression.

    Returns
    -------
    :class:`.genetics.KinshipMatrix`
        Realized relationship matrix for all samples.
    """
    mt = matrix_table_source('realized_relationship_matrix/call_expr', call_expr)
    require_col_key_str(mt, 'realized_relationship_matrix')

    col_keys = mt.cols().select()

    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())

    mt = mt.annotate_rows(__AC=agg.sum(mt.__gt),
                          __ACsq=agg.sum(mt.__gt * mt.__gt),
                          __n_called=agg.count_where(hl.is_defined(mt.__gt)))

    mt = mt.filter_rows((mt.__AC > 0) &
                        (mt.__AC < 2 * mt.__n_called) &
                        ((mt.__AC != mt.__n_called) |
                        (mt.__ACsq != mt.__n_called)))
    mt = mt.persist()

    n_variants, n_samples = mt.count()

    # once count_rows() adds partition_counts we can avoid annotating and filtering twice
    if n_variants == 0:
        raise FatalError("Cannot run RRM: found 0 variants after filtering out monomorphic sites.")
    info("Computing RRM using {} variants.".format(n_variants))

    mt = mt.annotate_rows(__mean_gt=mt.__AC / mt.__n_called)
    mt = mt.annotate_rows(__scaled_std_dev=hl.sqrt((mt.__ACsq + (n_samples - mt.__n_called) * mt.__mean_gt ** 2) /
                                              n_samples - mt.__mean_gt ** 2))

    normalized_gt = hl.or_else((mt.__gt - mt.__mean_gt) / mt.__scaled_std_dev, 0.0)

    bm = BlockMatrix.from_entry_expr(normalized_gt)
    mt.unpersist()

    rrm = (bm.T @ bm) / n_variants

    return KinshipMatrix._from_block_matrix(rrm,
                                            col_keys,
                                            n_variants)


@typecheck(n_populations=int,
           n_samples=int,
           n_variants=int,
           n_partitions=nullable(int),
           pop_dist=nullable(sequenceof(numeric)),
           fst=nullable(sequenceof(numeric)),
           af_dist=oneof(UniformDist, BetaDist, TruncatedBetaDist),
           seed=int,
           reference_genome=reference_genome_type,
           mixture=bool)
def balding_nichols_model(n_populations, n_samples, n_variants, n_partitions=None,
                          pop_dist=None, fst=None, af_dist=UniformDist(0.1, 0.9),
                          seed=0, reference_genome='default', mixture=False) -> MatrixTable:
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

    >>> from hail.stats import TruncatedBetaDist
    >>>
    >>> bn_ds = hl.balding_nichols_model(4, 40, 150, 3,
    ...          pop_dist=[0.1, 0.2, 0.3, 0.4],
    ...          fst=[.02, .06, .04, .12],
    ...          af_dist=TruncatedBetaDist(a=0.01, b=2.0, min=0.05, max=1.0),
    ...          seed=1)

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
      ``[0.1, 0.9]``. Other options are :class:`.UniformDist`,
      :class:`.BetaDist`, and :class:`.TruncatedBetaDist`.
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

    - `n_populations` (:py:data:`.tint32`) -- Number of populations.
    - `n_samples` (:py:data:`.tint32`) -- Number of samples.
    - `n_variants` (:py:data:`.tint32`) -- Number of variants.
    - `pop_dist` (:class:`.tarray` of :py:data:`.tfloat64`) -- Population distribution indexed by
      population.
    - `fst` (:class:`.tarray` of :py:data:`.tfloat64`) -- :math:`F_{ST}` values indexed by
      population.
    - `ancestral_af_dist` (:class:`.tstruct`) -- Description of the ancestral allele
      frequency distribution.
    - `seed` (:py:data:`.tint32`) -- Random seed.
    - `mixture` (:py:data:`.tbool`) -- Value of `mixture` parameter.

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
    af_dist : :class:`.UniformDist` or :class:`.BetaDist` or :class:`.TruncatedBetaDist`
        Ancestral allele frequency distribution.
        Default is ``UniformDist(0.1, 0.9)``.
    seed : :obj:`int`
        Random seed.
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
        jvm_pop_dist_opt = joption(pop_dist)
    else:
        jvm_pop_dist_opt = joption(jarray(Env.jvm().double, pop_dist))

    if fst is None:
        jvm_fst_opt = joption(fst)
    else:
        jvm_fst_opt = joption(jarray(Env.jvm().double, fst))

    jmt = Env.hc()._jhc.baldingNicholsModel(n_populations, n_samples, n_variants,
                                            joption(n_partitions),
                                            jvm_pop_dist_opt,
                                            jvm_fst_opt,
                                            af_dist._jrep(),
                                            seed,
                                            reference_genome._jrep,
                                            mixture)
    return MatrixTable(jmt)

@typecheck(mt=MatrixTable,f=anytype)
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
    new_locus, new_alleles = hl.min_rep(mt.locus, mt.new_to_old.map(lambda i: mt.alleles[i]))
    mt = mt.annotate_rows(__new_locus=new_locus, __new_alleles=new_alleles)
    mt = mt.filter_rows(hl.len(mt.__new_alleles) > 1)
    left = mt.filter_rows((mt.locus == mt.__new_locus) & (mt.alleles == mt.__new_alleles))

    right = mt.filter_rows((mt.locus != mt.__new_locus) | (mt.alleles != mt.__new_alleles))
    right = right.partition_rows_by('locus', locus=right.__new_locus, alleles=right.__new_alleles)
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

    sites_only_table = Table(Env.hail().methods.LocalLDPrune.apply(
        mt._jvds, call_field, float(r2), bp_window_size, max_queue_size))

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

    If you encounter a Hadoop write/replication error, consider:

    - increasing the size of persistent disk, e.g. by increasing the number of
      persistent workers or the disk size per persistent worker. The
      locally-pruned matrix table and block matrix are stored as temporary files
      on persistent disk.

    - limiting the Hadoop write buffer size, e.g. by setting the property on
      cluster startup: ``--properties 'core:fs.gs.io.buffersize.write=1048576``.
      This issue arises for very large sample size because, when writing the
      locally-pruned block matrix, the number of concurrently open files per
      task is ``n_samples / block_size``.

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
    require_partition_key_locus(mt, 'ld_prune')

    #  FIXME: remove once select_entries on a field is free
    if call_expr in mt._fields_inverse:
        field = mt._fields_inverse[call_expr]
    else:
        field = Env.get_uid()
        mt = mt.select_entries(**{field: call_expr})

    mt = mt.distinct_by_row()
    locally_pruned_table_path = new_temp_file()
    (_local_ld_prune(require_biallelic(mt, 'ld_prune'), field, r2, bp_window_size, memory_per_core)
        .add_index()
        .write(locally_pruned_table_path, overwrite=True))
    locally_pruned_table = hl.read_table(locally_pruned_table_path)

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

    std_gt_bm = BlockMatrix.from_entry_expr(standardized_mean_imputed_gt_expr, block_size)
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
        hl.is_defined(variants_to_remove[locally_pruned_table.idx]), keep=False).select()
