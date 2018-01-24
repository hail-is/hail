from hail.api2.matrixtable import MatrixTable, Table
from hail.expr.expression import *
from hail.genetics import KinshipMatrix, GenomeReference
from hail.genetics.ldMatrix import LDMatrix
from hail.linalg import BlockMatrix
from hail.typecheck import *
from hail.utils import wrap_to_list, new_temp_file, info
from hail.utils.java import handle_py4j, joption
from .misc import require_biallelic
from hail.expr import functions
import hail.expr.aggregators as agg
from math import sqrt
from hail.stats import UniformDist, BetaDist, TruncatedBetaDist

@handle_py4j
@typecheck(dataset=MatrixTable,
           maf=nullable(oneof(Float32Expression, Float64Expression)),
           bounded=bool,
           min=nullable(numeric),
           max=nullable(numeric))
def ibd(dataset, maf=None, bounded=True, min=None, max=None):
    """Compute matrix of identity-by-descent estimations.

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    To calculate a full IBD matrix, using minor allele frequencies computed
    from the dataset itself:

    >>> methods.ibd(dataset)

    To calculate an IBD matrix containing only pairs of samples with
    ``PI_HAT`` in :math:`[0.2, 0.9]`, using minor allele frequencies stored in
    the row field `panel_maf`:

    >>> methods.ibd(dataset, maf=dataset['panel_maf'], min=0.2, max=0.9)

    Notes
    -----

    The implementation is based on the IBD algorithm described in the `PLINK
    paper <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1950838>`__.

    :meth:`ibd` requires the dataset to be biallelic and does not perform LD
    pruning. Linkage disequilibrium may bias the result so consider filtering
    variants first.

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
        A variant-keyed :class:`.MatrixTable` containing genotype information.

    maf : :class:`.Float32Expression`, :class:`.Float64Expression` or :obj:`None`
        (optional) expression on `dataset` for the minor allele frequency.

    bounded : :obj:`bool`
        Forces the estimations for `Z0``, ``Z1``, ``Z2``, and ``PI_HAT`` to take
        on biologically meaningful values (in the range [0,1]).

    min : :obj:`float` or :obj:`None`
        Sample pairs with a ``PI_HAT`` below this value will
        not be included in the output. Must be in :math:`[0,1]`.

    max : :obj:`float` or :obj:`None`
        Sample pairs with a ``PI_HAT`` above this value will
        not be included in the output. Must be in :math:`[0,1]`.

    Return
    ------
    :class:`.Table`
        A table which maps pairs of samples to their IBD statistics
    """

    if maf is not None:
        analyze('ibd/maf', maf, dataset._row_indices)
        dataset, _ = dataset._process_joins(maf)
        maf = maf._ast.to_hql()

    return Table(Env.hail().methods.IBD.apply(require_biallelic(dataset, 'ibd')._jvds,
                                              joption(maf),
                                              bounded,
                                              joption(min),
                                              joption(max)))

@handle_py4j
@typecheck(dataset=MatrixTable,
           ys=oneof(Expression, listof(Expression)),
           x=NumericExpression,
           covariates=listof(Expression),
           root=strlike,
           block_size=integral)
def linreg(dataset, ys, x, covariates=[], root='linreg', block_size=16):
    """For each row, test a derived input variable for association with response variables using linear regression.

    Examples
    --------

    >>> dataset_result = methods.linreg(dataset, [dataset.pheno.height], dataset.GT.num_alt_alleles(),
    ...                                 covariates=[dataset.pheno.age, dataset.pheno.isFemale])

    Warning
    -------
    :meth:`.linreg` considers the same set of columns (i.e., samples, points) for every response variable and row,
    namely those columns for which **all** response variables and covariates are defined.
    For each row, missing values of ``x`` are mean-imputed over these columns.

    Notes
    -----

    With the default root, the following row-indexed fields are added.
    The indexing of the array fields corresponds to that of ``ys``.

    - **linreg.nCompleteSamples** (*Int32*) -- number of columns used
    - **linreg.AC** (*Float64*) -- sum of input values ``x``
    - **linreg.ytx** (*Array[Float64]*) -- array of dot products of each response vector ``y`` with the input vector ``x``
    - **linreg.beta** (*Array[Float64]*) -- array of fit effect coefficients of ``x``, :math:`\hat\\beta_1` below
    - **linreg.se** (*Array[Float64]*) -- array of estimated standard errors, :math:`\widehat{\mathrm{se}}_1`
    - **linreg.tstat** (*Array[Float64]*) -- array of :math:`t`-statistics, equal to :math:`\hat\\beta_1 / \widehat{\mathrm{se}}_1`
    - **linreg.pval** (*Array[Float64]*) -- array of :math:`p`-values

    In the statistical genetics example above, the input variable ``x`` encodes genotype
    as the number of alternate alleles (0, 1, or 2). For each variant (row), genotype is tested for association
    with height controlling for age and sex, by fitting the linear regression model:

    .. math::

        \mathrm{height} = \\beta_0 + \\beta_1 \, \mathrm{genotype} + \\beta_2 \, \mathrm{age} + \\beta_3 \, \mathrm{isFemale} + \\varepsilon, \quad \\varepsilon \sim \mathrm{N}(0, \sigma^2)

    Boolean covariates like :math:`\mathrm{isFemale}` are encoded as 1 for true and 0 for false.
    The null model sets :math:`\\beta_1 = 0`.

    The standard least-squares linear regression model is derived in Section
    3.2 of `The Elements of Statistical Learning, 2nd Edition
    <http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf>`__. See
    equation 3.12 for the t-statistic which follows the t-distribution with
    :math:`n - k - 2` degrees of freedom, under the null hypothesis of no
    effect, with :math:`n` samples and :math:`k` covariates in addition to
    ``x`` and the intercept.

    Parameters
    ----------
    ys : :obj:`list` of :class:`.NumericExpression`
        One or more response expressions.
    x : :class:`.NumericExpression`
        Input variable.
    covariates : :obj:`list` of :class:`.NumericExpression`
        Covariate expressions.
    root : :obj:`str`
        Name of resulting row-indexed field.
    block_size : :obj:`int`
        Number of row regressions to perform simultaneously per core. Larger blocks
        require more memory but may improve performance.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with regression results in a new row-indexed field.
    """

    all_exprs = [x]

    ys = wrap_to_list(ys)

    # x is entry-indexed
    analyze('linreg/x', x, dataset._entry_indices)

    # ys and covariates are col-indexed
    ys = wrap_to_list(ys)
    for e in ys:
        all_exprs.append(e)
        analyze('linreg/ys', e, dataset._col_indices)
    for e in covariates:
        all_exprs.append(e)
        analyze('linreg/covariates', e, dataset._col_indices)

    base, cleanup = dataset._process_joins(*all_exprs)

    jm = base._jvds.linreg(
        jarray(Env.jvm().java.lang.String, [y._ast.to_hql() for y in ys]),
        x._ast.to_hql(),
        jarray(Env.jvm().java.lang.String, [cov._ast.to_hql() for cov in covariates]),
        'va.`{}`'.format(root),
        block_size
    )

    return cleanup(MatrixTable(jm))

@handle_py4j
@typecheck(dataset=MatrixTable,
           test=strlike,
           y=Expression,
           x=Expression,
           covariates=listof(Expression),
           root=strlike)
def logreg(dataset, test, y, x, covariates=[], root='logreg'):
    r"""For each row, test a derived input variable for association with a
    Boolean response variable using logistic regression.

    Examples
    --------

    Run the logistic regression Wald test per variant using a Boolean
    phenotype and two covariates stored in column-indexed fields:

    >>> ds_result = methods.logreg(
    ...     dataset,
    ...     test='wald',
    ...     y=dataset.pheno.isCase,
    ...     x=dataset.GT.num_alt_alleles(),
    ...     covariates=[dataset.pheno.age, dataset.pheno.isFemale])

    Notes
    -----

    The :func:`.logreg` method performs, for each row, a significance test of
    the input variable in predicting a binary (case-control) response
    variable based on the logistic regression model. The response variable
    type must either be numeric (with all present values 0 or 1) or Boolean,
    in which case true and false are coded as 1 and 0, respectively.

    Hail supports the Wald test ('wald'), likelihood ratio test ('lrt'),
    Rao score test ('score'), and Firth test ('firth'). Hail only includes
    columns for which the response variable and all covariates are defined.
    For each row, Hail imputes missing input values as the mean of the
    non-missing values.

    The example above considers a model of the form

    .. math::

        \mathrm{Prob}(\mathrm{isCase}) =
            \mathrm{sigmoid}(\beta_0 + \beta_1 \, \mathrm{gt}
                            + \beta_2 \, \mathrm{age}
                            + \beta_3 \, \mathrm{isFemale} + \varepsilon),
        \quad
        \varepsilon \sim \mathrm{N}(0, \sigma^2)

    where :math:`\mathrm{sigmoid}` is the `sigmoid function`_, the genotype
    :math:`\mathrm{gt}` is coded as 0 for HomRef, 1 for Het, and 2 for
    HomVar, and the Boolean covariate :math:`\mathrm{isFemale}` is coded as
    for true (female) and 0 for false (male). The null model sets
    :math:`\beta_1 = 0`.

    .. _sigmoid function: https://en.wikipedia.org/wiki/Sigmoid_function

    The resulting variant annotations depend on the test statistic as shown
    in the tables below.

    ========== =============== ======  ============================================
    Test       Field           Type    Value
    ========== =============== ======  ============================================
    Wald       `logreg.beta`   Double  fit effect coefficient,
                                       :math:`\hat\beta_1`
    Wald       `logreg.se`     Double  estimated standard error,
                                       :math:`\widehat{\mathrm{se}}`
    Wald       `logreg.zstat`  Double  Wald :math:`z`-statistic, equal to
                                       :math:`\hat\beta_1 / \widehat{\mathrm{se}}`
    Wald       `logreg.pval`   Double  Wald p-value testing :math:`\beta_1 = 0`
    LRT, Firth `logreg.beta`   Double  fit effect coefficient,
                                       :math:`\hat\beta_1`
    LRT, Firth `logreg.chi2`   Double  deviance statistic
    LRT, Firth `logreg.pval`   Double  LRT / Firth p-value testing
                                       :math:`\beta_1 = 0`
    Score      `logreg.chi2`   Double  score statistic
    Score      `logreg.pval`   Double  score p-value testing :math:`\beta_1 = 0`
    ========== =============== ======  ============================================

    For the Wald and likelihood ratio tests, Hail fits the logistic model for
    each row using Newton iteration and only emits the above annotations
    when the maximum likelihood estimate of the coefficients converges. The
    Firth test uses a modified form of Newton iteration. To help diagnose
    convergence issues, Hail also emits three variant annotations which
    summarize the iterative fitting process:

    ================ ======================= ======= ===========================
    Test             Field                   Type    Value
    ================ ======================= ======= ===========================
    Wald, LRT, Firth `logreg.fit.nIter`      Int     number of iterations until
                                                     convergence, explosion, or
                                                     reaching the max (25 for
                                                     Wald, LRT; 100 for Firth)
    Wald, LRT, Firth `logreg.fit.converged`  Boolean true if iteration converged
    Wald, LRT, Firth `logreg.fit.exploded`   Boolean true if iteration exploded
    ================ ======================= ======= ===========================

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
    the `lmmreg.fit` fields reflect the null model; otherwise, they reflect the
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

    Phenotype and covariate sample annotations may also be specified using
    `programmatic expressions <exprlang.html>`__ without identifiers, such as:

    .. code-block:: text

        if (ds.isFemale) ds.cov.age else (2 * ds.cov.age + 10)

    For Boolean covariate types, true is coded as 1 and false as 0. In
    particular, for the sample annotation `fam.isCase` added by importing a FAM
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
    y : :class:`.AtomicExpression`
        Response expression. Must evaluate to Boolean or numeric with all values
        0 or 1.
    x : :class:`.NumericExpression`
        Expression for input variable.
    covariates : :obj:`list` of :class:`.NumericExpression`, optional
        Covariate expressions.
    root : :obj:`str`, optional
        Name of resulting row-indexed field.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with regression results in a new row-indexed field.
    """

    # x is entry-indexed
    analyze('logreg/x', x, dataset._entry_indices)

    # y and covariates are col-indexed
    analyze('logreg/y', y, dataset._col_indices)

    all_exprs = [x, y]
    for e in covariates:
        all_exprs.append(e)
        analyze('logreg/covariates', e, dataset._col_indices)

    base, cleanup = dataset._process_joins(*all_exprs)

    jds = base._jvds.logreg(
        test,
        y._ast.to_hql(),
        x._ast.to_hql(),
        jarray(Env.jvm().java.lang.String,
               [cov._ast.to_hql() for cov in covariates]),
        'va.`{}`'.format(root))

    return cleanup(MatrixTable(jds))

@handle_py4j
@typecheck(ds=MatrixTable,
           kinshipMatrix=KinshipMatrix,
           y=Expression,
           x=Expression,
           covariates=listof(Expression),
           global_root=strlike,
           va_root=strlike,
           run_assoc=bool,
           use_ml=bool,
           delta=nullable(numeric),
           sparsity_threshold=numeric,
           n_eigs=nullable(integral),
           dropped_variance_fraction=(nullable(float)))
def lmmreg(ds, kinshipMatrix, y, x, covariates=[], global_root="lmmreg_global", va_root="lmmreg",
           run_assoc=True, use_ml=False, delta=None, sparsity_threshold=1.0,
           n_eigs=None, dropped_variance_fraction=None):
    r"""Use a kinship-based linear mixed model to estimate the genetic component
    of phenotypic variance (narrow-sense heritability) and optionally test each
    variant for association.

    Examples
    --------
    Compute a :class:`.KinshipMatrix`, and use it to test variants for
    association using a linear mixed model:

    >>> ds = hc.read("data/example_lmmreg.vds")
    >>> kinship_matrix = methods.rrm(ds.filter_rows(ds.useInKinship)['GT'])
    >>> lmm_ds = methods.lmmreg(ds, kinship_matrix, ds.pheno, ds.GT.num_alt_alleles(), [ds.cov1, ds.cov2])

    Notes
    -----
    Suppose the variant dataset saved at :file:`data/example_lmmreg.vds` has a
    Boolean variant-indexed field `useInKinship` and numeric or Boolean
    sample-indexed fields `pheno`, `cov1`, and `cov2`. Then the
    :func:`.lmmreg` function in the above example will execute the following
    four steps in order:

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
    - Set the `va_root` argument to change the variant annotation root in
      Step 5.

    :func:`.lmmreg` adds 9 or 13 global annotations in Step 4, depending on
    whether :math:`\delta` is set or fit.

    .. list-table::
       :header-rows: 1

       * - Field
         - Type
         - Value
       * - `lmmreg_global.useML`
         - Boolean
         - true if fit by ML, false if fit by REML
       * - `lmmreg_global.beta`
         - Dict[String, Double]
         - map from *intercept* and the given ``covariates`` expressions to the
           corresponding fit :math:`\beta` coefficients
       * - `lmmreg_global.sigmaG2`
         - Double
         - fit coefficient of genetic variance, :math:`\hat{\sigma}_g^2`
       * - `lmmreg_global.sigmaE2`
         - Double
         - fit coefficient of environmental variance :math:`\hat{\sigma}_e^2`
       * - `lmmreg_global.delta`
         - Double
         - fit ratio of variance component coefficients, :math:`\hat{\delta}`
       * - `lmmreg_global.h2`
         - Double
         - fit narrow-sense heritability, :math:`\hat{h}^2`
       * - `lmmreg_global.nEigs`
         - Int
         - number of eigenvectors of kinship matrix used to fit model
       * - `lmmreg_global.dropped_variance_fraction`
         - Double
         - specified value of `dropped_variance_fraction`
       * - `lmmreg_global.evals`
         - Array[Double]
         - all eigenvalues of the kinship matrix in descending order
       * - `lmmreg_global.fit.seH2`
         - Double
         - standard error of :math:`\hat{h}^2` under asymptotic normal
           approximation
       * - `lmmreg_global.fit.normLkhdH2`
         - Array[Double]
         - likelihood function of :math:`h^2` normalized on the discrete grid
           ``0.01, 0.02, ..., 0.99``. Index ``i`` is the likelihood for
           percentage ``i``.
       * - `lmmreg_global.fit.maxLogLkhd`
         - Double
         - (restricted) maximum log likelihood corresponding to
           :math:`\hat{\delta}`
       * - `lmmreg_global.fit.logDeltaGrid`
         - Array[Double]
         - values of :math:`\mathrm{ln}(\delta)` used in the grid search
       * - `lmmreg_global.fit.logLkhdVals`
         - Array[Double]
         - (restricted) log likelihood of :math:`y` given :math:`X` and
           :math:`\mathrm{ln}(\delta)` at the (RE)ML fit of :math:`\beta` and
           :math:`\sigma_g^2`

    These global annotations are also added to ``hail.log``, with the ranked
    evals and :math:`\delta` grid with values in .tsv tabular form.  Use
    ``grep 'lmmreg:' hail.log`` to find the lines just above each table.

    If Step 5 is performed, :func:`.lmmreg` also adds four linear regression
    variant annotations.

    +------------------+--------+------------------------------------------------+
    | Field            | Type   | Value                                          |
    +==================+========+================================================+
    | `lmmreg.beta`    | Double | fit genotype coefficient, :math:`\hat\beta_0`  |
    +------------------+--------+------------------------------------------------+
    | `lmmreg.sigmaG2` | Double | fit coefficient of genetic variance component, |
    |                  |        | :math:`\hat{\sigma}_g^2`                       |
    +------------------+--------+------------------------------------------------+
    | `lmmreg.chi2`    | Double | :math:`\chi^2` statistic of the likelihood     |
    |                  |        | ratio test                                     |
    +------------------+--------+------------------------------------------------+
    | `lmmreg.pval`    | Double | :math:`p`-value                                |
    +------------------+--------+------------------------------------------------+

    Those variants that don't vary across the included samples (e.g., all
    genotypes are HomRef) will have missing annotations.

    **Performance**

    Hail's initial version of :func:`.lmmreg` scales beyond 15k samples and
    to an essentially unbounded number of variants, making it particularly
    well-suited to modern sequencing studies and complementary to tools
    designed for SNP arrays. Analysts have used :func:`.lmmreg` in research to
    compute kinship from 100k common variants and test 32 million non-rare
    variants on 8k whole genomes in about 10 minutes on `Google cloud`_.

    .. _Google cloud:
        http://discuss.hail.is/t/using-hail-on-the-google-cloud-platform/80

    While :func:`.lmmreg` computes the kinship matrix :math:`K` using
    distributed matrix multiplication (Step 2), the full `eigendecomposition`_
    (Step 3) is currently run on a single core of master using the
    `LAPACK routine DSYEVD`_, which we empirically find to be the most
    performant of the four available routines; laptop performance plots showing
    cubic complexity in :math:`n` are available
    `here <https://github.com/hail-is/hail/pull/906>`__. On Google cloud,
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

    :func:`.lmmreg` estimates the genetic proportion of residual phenotypic
    variance (narrow-sense heritability) under a kinship-based linear mixed
    model, and then optionally tests each variant for association using the
    likelihood ratio test. Inference is exact.

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
    containing "lmmreg: table of delta".

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

    For convenience, `lmmreg.fit.normLkhdH2` records the the likelihood
    function of :math:`h^2` normalized over the discrete grid
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

    Note that the mean and standard deviation of the (discretized or continuous)
    distribution held in `lmmreg.fit.normLkhdH2` will not coincide with
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

    :func:`.lmmreg` can implicitly use a low-rank approximation of the kinship
    matrix to more rapidly fit delta and the statistics for each variant. The
    computational complexity per variant is proportional to the number of
    eigenvectors used. This number can be specified in two ways. Specify the
    parameter `n_eigs` to use only the top `n_eigs` eigenvectors.
    Alternatively, specify `dropped_variance_fraction` to use as many
    eigenvectors as necessary to capture all but at most this fraction of the
    sample variance (also known as the trace, or the sum of the eigenvalues).
    For example, setting `dropped_variance_fraction` to 0.01 will use the
    minimal number of eigenvectors to account for 99% of the sample variance.
    Specifying both parameters will apply the more stringent (fewest
    eigenvectors) of the two.

    **Further background**

    For the history and mathematics of linear mixed models in genetics,
    including `FastLMM`_, see
    `Christoph Lippert's PhD thesis <https://publikationen.uni-tuebingen.de/xmlui/bitstream/handle/10900/50003/pdf/thesis_komplett.pdf>`__.
    For an investigation of various approaches to defining kinship, see
    `Comparison of Methods to Account for Relatedness in Genome-Wide Association Studies with Family-Based Data <http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004445>`__.

    Parameters
    ----------
    kinshipMatrix : :class:`.KinshipMatrix`
        Kinship matrix to be used.

    y : :class:`.NumericExpression`
        Response sample expression.

    x : :class:`.NumericExpression`
        Input variable.

    covariates : :obj:`list` of :class:`.NumericExpression`
        List of covariate sample expressions.

    global_root : :obj:`str`
        Global field root, a period-delimited path.

    va_root : :obj:`str`
        Variant-indexed field root, a period-delimited path.

    run_assoc : :obj:`bool`
        If true, run association testing in addition to fitting the global model.

    use_ml : :obj:`bool`
        Use ML instead of REML throughout.

    delta : :obj:`float` or :obj:`None`
        Fixed delta value to use in the global model, overrides fitting delta.

    sparsity_threshold : :obj:`float`
        Genotype vector sparsity at or below which to use sparse genotype
        vector in rotation (advanced).

    n_eigs : :obj:`int`
        Number of eigenvectors of the kinship matrix used to fit the model.

    dropped_variance_fraction : :obj:`float`
        Upper bound on fraction of sample variance lost by dropping
        eigenvectors with small eigenvalues.

    Returns
    -------
    :class:`.MatrixTable`
        Variant dataset with linear mixed regression annotations.
    """

    # x is entry-indexed
    analyze('lmmreg/x', x, ds._entry_indices)

    # y and covariates are col-indexed
    analyze('lmmreg/y', y, ds._col_indices)

    all_exprs = [x, y]
    for e in covariates:
        all_exprs.append(e)
        analyze('lmmreg/covariates', e, ds._col_indices)

    base, cleanup = ds._process_joins(*all_exprs)

    jds = base._jvds.lmmreg(kinshipMatrix._jkm,
                            y._ast.to_hql(),
                            x._ast.to_hql(),
                            jarray(Env.jvm().java.lang.String,
                                   [cov._ast.to_hql() for cov in covariates]),
                            use_ml,
                            'global.`{}`'.format(global_root),
                            'va.`{}`'.format(va_root),
                            run_assoc,
                            joption(delta),
                            sparsity_threshold,
                            joption(n_eigs),
                            joption(dropped_variance_fraction))

    return cleanup(MatrixTable(jds))

@handle_py4j
@typecheck(dataset=MatrixTable, force_local=bool)
def ld_matrix(dataset, force_local=False):
    """Computes the linkage disequilibrium (correlation) matrix for the variants in this VDS.

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    .. testsetup::

        dataset = vds.annotate_samples_expr('sa = drop(sa, qc)').to_hail2()
        from hail.methods import ld_matrix

    **Examples**

    >>> ld_matrix = ld_matrix(dataset)

    **Notes**

    Each entry (i, j) in the LD matrix gives the :math:`r` value between variants i and j, defined as
    `Pearson's correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__
    :math:`\\rho_{x_i,x_j}` between the two genotype vectors :math:`x_i` and :math:`x_j`.

    .. math::

        \\rho_{x_i,x_j} = \\frac{\\mathrm{Cov}(X_i,X_j)}{\\sigma_{X_i} \\sigma_{X_j}}

    Also note that variants with zero variance (:math:`\\sigma = 0`) will be dropped from the matrix.

    .. caution::

        The matrix returned by this function can easily be very large with most entries near zero
        (for example, entries between variants on different chromosomes in a homogenous population).
        Most likely you'll want to reduce the number of variants with methods like
        :meth:`.sample_variants`, :meth:`.filter_variants_expr`, or :meth:`.ld_prune` before
        calling this unless your dataset is very small.

    :param dataset: Variant-keyed dataset.
    :type dataset: :class:`.MatrixTable`

    :param bool force_local: If true, the LD matrix is computed using local matrix multiplication on the Spark driver.
        This may improve performance when the genotype matrix is small enough to easily fit in local memory.
        If false, the LD matrix is computed using distributed matrix multiplication if the number of entries
        exceeds :math:`5000^2` and locally otherwise.

    :return: Matrix of r values between pairs of variants.
    :rtype: :class:`.LDMatrix`
    """

    jldm = Env.hail().methods.LDMatrix.apply(require_biallelic(dataset, 'ld_matrix')._jvds, force_local)
    return LDMatrix(jldm)


@handle_py4j
@typecheck(dataset=MatrixTable,
           k=integral,
           compute_loadings=bool,
           as_array=bool)
def hwe_normalized_pca(dataset, k=10, compute_loadings=False, as_array=False):
    """Run principal component analysis (PCA) on the Hardy-Weinberg-normalized call matrix.

    Examples
    --------

    >>> eigenvalues, scores, loadings = methods.hwe_normalized_pca(dataset, k=5)

    Notes
    -----
    Variants that are all homozygous reference or all homozygous variant are removed before evaluation.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    k : :obj:`int`
        Number of principal components.
    compute_loadings : :obj:`bool`
        If ``True``, compute row loadings.
    as_array : :obj:`bool`
        If ``True``, return scores and loadings as an array field. If ``False``, return
        one field per element (`PC1`, `PC2`, ... `PCk`).

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """
    dataset = require_biallelic(dataset, 'hwe_normalized_pca')
    dataset = dataset.annotate_rows(AC=agg.sum(dataset.GT.num_alt_alleles()),
                                    n_called=agg.count_where(functions.is_defined(dataset.GT)))
    dataset = dataset.filter_rows((dataset.AC > 0) & (dataset.AC < 2 * dataset.n_called)).persist()

    n_variants = dataset.count_rows()
    if n_variants == 0:
        raise FatalError(
            "Cannot run PCA: found 0 variants after filtering out monomorphic sites.")
    info("Running PCA using {} variants.".format(n_variants))

    entry_expr = functions.bind(
        dataset.AC / dataset.n_called,
        lambda mean_gt: functions.cond(functions.is_defined(dataset.GT),
                                       (dataset.GT.num_alt_alleles() - mean_gt) /
                                       functions.sqrt(mean_gt * (2 - mean_gt) * n_variants / 2),
                                       0))
    result = pca(entry_expr,
                 k,
                 compute_loadings,
                 as_array)
    dataset.unpersist()
    return result


@handle_py4j
@typecheck(entry_expr=expr_numeric,
           k=integral,
           compute_loadings=bool,
           as_array=bool)
def pca(entry_expr, k=10, compute_loadings=False, as_array=False):
    """Run principal Component Analysis (PCA) on a matrix table, using `entry_expr` as the numerical entry.

    Examples
    --------

    Compute the top 2 principal component scores and eigenvalues of the call missingness matrix.

    >>> eigenvalues, scores, _ = methods.pca(functions.is_defined(dataset.GT).to_int32(),
    ...                                      k=2)

    Notes
    -----

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
    :math:`n \\times k`, :math:`k \\times k` and :math:`m \\times k`
    respectively.

    From the perspective of the samples or rows of :math:`M` as data,
    :math:`V_k` contains the variant loadings for the first :math:`k` PCs while
    :math:`MV_k = U_k S_k` contains the first :math:`k` PC scores of each
    sample. The loadings represent a new basis of features while the scores
    represent the projected data on those features. The eigenvalues of the GRM
    :math:`MM^T` are the squares of the singular values :math:`s_1^2, s_2^2,
    \ldots`, which represent the variances carried by the respective PCs. By
    default, Hail only computes the loadings if the ``loadings`` parameter is
    specified.

    Note
    ----
    In PLINK/GCTA the GRM is taken as the starting point and it is
    computed slightly differently with regard to missing data. Here the
    :math:`ij` entry of :math:`MM^T` is simply the dot product of rows :math:`i`
    and :math:`j` of :math:`M`; in terms of :math:`C` it is

    .. math::

      \\frac{1}{m}\sum_{l\in\mathcal{C}_i\cap\mathcal{C}_j}\\frac{(C_{il}-2p_l)(C_{jl} - 2p_l)}{2p_l(1-p_l)}

    where :math:`\mathcal{C}_i = \{l \mid C_{il} \\text{ is non-missing}\}`. In
    PLINK/GCTA the denominator :math:`m` is replaced with the number of terms in
    the sum :math:`\\lvert\mathcal{C}_i\cap\\mathcal{C}_j\\rvert`, i.e. the
    number of variants where both samples have non-missing genotypes. While this
    is arguably a better estimator of the true GRM (trading shrinkage for
    noise), it has the drawback that one loses the clean interpretation of the
    loadings and scores as features and projections.

    Separately, for the PCs PLINK/GCTA output the eigenvectors of the GRM; even
    ignoring the above discrepancy that means the left singular vectors
    :math:`U_k` instead of the component scores :math:`U_k S_k`. While this is
    just a matter of the scale on each PC, the scores have the advantage of
    representing true projections of the data onto features with the variance of
    a score reflecting the variance explained by the corresponding feature. (In
    PC bi-plots this amounts to a change in aspect ratio; for use of PCs as
    covariates in regression it is immaterial.)

    Scores are stored in a :class:`.Table` with the column keys of the matrix,
    and the following additional field:

     - **scores**: The principal component scores. This can have two different
       structures, depending on the value of the `as_array` flag.

    If `as_array` is ``False`` (default), then `scores` is a ``Struct`` with
    fields `PC1`, `PC2`, etc. If `as_array` is ``True``, then `scores` is a
    field of type ``Array[Float64]`` containing the principal component scores.

    Loadings are stored in a :class:`.Table` with a structure similar to the scores
    table:

     - **v**: Row key of the dataset.

     - **pcaLoadings**: Row loadings (same type as the scores)

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    entry_expr : :class:`.Expression`
        Numeric expression for matrix entries.
    k : :obj:`int`
        Number of principal components.
    compute_loadings : :obj:`bool`
        If ``True``, compute row loadings.
    as_array : :obj:`bool`
        If ``True``, return scores and loadings as an array field. If ``False``, return
        one field per element (`PC1`, `PC2`, ... `PCk`).

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """
    source = entry_expr._indices.source
    if not isinstance(source, MatrixTable):
        raise ValueError("Expect an expression of 'MatrixTable', found {}".format(
            "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'))
    dataset = source
    base, _ = dataset._process_joins(entry_expr)
    analyze('pca', entry_expr, dataset._entry_indices)

    r = Env.hail().methods.PCA.apply(dataset._jvds, to_expr(entry_expr)._ast.to_hql(), k, compute_loadings, as_array)
    scores = Table(Env.hail().methods.PCA.scoresTable(dataset._jvds, as_array, r._2()))
    loadings = from_option(r._3())
    if loadings:
        loadings = Table(loadings)
    return (jiterable_to_list(r._1()), scores, loadings)

@handle_py4j
@typecheck_method(k=integral,
                  maf=numeric,
                  block_size=integral,
                  min_kinship=numeric,
                  statistics=enumeration("phi", "phik2", "phik2k0", "all"))
def pc_relate(dataset, k, maf, block_size=512, min_kinship=-float("inf"), statistics="all"):
    """Compute relatedness estimates between individuals using a variant of the
    PC-Relate method.

    .. include:: ../_templates/experimental.rst

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    Estimate kinship, identity-by-descent two, identity-by-descent one, and
    identity-by-descent zero for every pair of samples, using 10 prinicpal
    components to correct for ancestral populations, and a minimum minor
    allele frequency filter of 0.01:

    >>> rel = vds.pc_relate(10, 0.01)

    Calculate values as above, but when performing distributed matrix
    multiplications use a matrix-block-size of 1024 by 1024.

    >>> rel = vds.pc_relate(10, 0.01, 1024)

    Calculate values as above, excluding sample-pairs with kinship less
    than 0.1. This is more efficient than producing the full table and
    filtering using :meth:`.Table.filter`.

    >>> rel = vds.pc_relate(5, 0.01, min_kinship=0.1)


    The traditional estimator for kinship between a pair of individuals
    :math:`i` and :math:`j`, sharing the set :math:`S_{ij}` of
    single-nucleotide variants, from a population with allele frequencies
    :math:`p_s`, is given by:

    .. math::

      \\widehat{\\phi_{ij}} :=
        \\frac{1}{|S_{ij}|}
        \\sum_{s \\in S_{ij}}
          \\frac{(g_{is} - 2 p_s) (g_{js} - 2 p_s)}
                {4 * \\sum_{s \\in S_{ij} p_s (1 - p_s)}}

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
    occurences of population allele frequency are replaced with an
    "individual-specific allele frequency". This modification allows the
    method to correctly weight an allele according to an individual's unique
    ancestry profile.

    The "individual-specific allele frequency" at a given genetic locus is
    modeled by PC-Relate as a linear function of their first ``k`` principal
    component coordinates. As such, the efficacy of this method rests on two
    assumptions:

     - an individual's first ``k`` principal component coordinates fully
       describe their allele-frequency-relevant ancestry, and

     - the relationship between ancestry (as described by principal
       component coordinates) and population allele frequency is linear

    The estimators for kinship, and identity-by-descent zero, one, and two
    follow. Let:

     - :math:`S_{ij}` be the set of genetic loci at which both individuals
       :math:`i` and :math:`j` have a defined genotype

     - :math:`g_{is} \\in {0, 1, 2}` be the number of alternate alleles that
       individual :math:`i` has at gentic locus :math:`s`

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

    Notes
    -----
    The PC-Relate method is described in "Model-free Estimation of Recent
    Genetic Relatedness". Conomos MP, Reiner AP, Weir BS, Thornton TA. in
    American Journal of Human Genetics. 2016 Jan 7. The reference
    implementation is available in the `GENESIS Bioconductor package
    <https://bioconductor.org/packages/release/bioc/html/GENESIS.html>`_ .

    :func:`methods.pc_relate` differs from the reference
    implementation in a couple key ways:

     - the principal components analysis does not use an unrelated set of
       individuals

     - the estimators do not perform small sample correction

     - the algorithm does not provide an option to use population-wide
       allele frequency estimates

     - the algorithm does not provide an option to not use "overall
       standardization" (see R ``pcrelate`` documentation)

    Note
    ----
    The ``block_size`` controls memory usage and parallelism. If it is large
    enough to hold an entire sample-by-sample matrix of 64-bit doubles in
    memory, then only one Spark worker node can be used to compute matrix
    operations. If it is too small, communication overhead will begin to
    dominate the computation's time. The author has found that on Google
    Dataproc (where each core has about 3.75GB of memory), setting
    ``block_size`` larger than 512 tends to cause memory exhaustion errors.

    Note
    ----
    The minimum allele frequency filter is applied per-pair: if either of
    the two individual's individual-specific minor allele frequency is below
    the threshold, then the variant's contribution to relatedness estimates
    is zero.

    Note
    ----
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



    Parameters
    ----------
    k : :obj:`int`
        The number of principal components to use to distinguish ancestries.
    maf : :obj:`float`
        The minimum individual-specific allele frequency for an allele used to
        measure relatedness.
    block_size : :obj:`int`
        the side length of the blocks of the block-distributed matrices; this
        should be set such that at least three of these matrices fit in memory
        (in addition to all other objects necessary for Spark and Hail).
    min_kinship : :obj:`float`
        Pairs of samples with kinship lower than ``min_kinship`` are excluded
        from the results.
    statistics : :obj:`str`
        the set of statistics to compute, `phi` will only compute the
        kinship statistic, `phik2` will compute the kinship and
        identity-by-descent two statistics, `phik2k0` will compute the
        kinship statistics and both identity-by-descent two and zero, `all`
        computes the kinship statistic and all three identity-by-descent
        statistics.

    Returns
    -------
    :class:`.Table`
        A :class:`.Table` mapping pairs of samples to estimations of their
        kinship and identity-by-descent zero, one, and two.

        The fields of the resulting :class:`.Table` entries are of types:
        `i`: `String`, `j`: `String`, `kin`: `Double`, `k2`: `Double`,
        `k1`: `Double`, `k0`: `Double`. The table is keyed by `i` and `j`.
    """
    dataset = require_biallelic(dataset, 'pc_relate')
    intstatistics = {"phi": 0, "phik2": 1, "phik2k0": 2, "all": 3}[statistics]
    _, scores, _ = hwe_normalized_pca(dataset, k, False, True)
    return Table(
        scala_object(Env.hail().methods, 'PCRelate')
            .apply(dataset._jvds,
                   k,
                   scores._jt,
                   maf,
                   block_size,
                   min_kinship,
                   intstatistics))

@handle_py4j
@typecheck(dataset=MatrixTable,
           fraction=numeric,
           seed=integral)
def sample_rows(dataset, fraction, seed=1):
    """Downsample rows to a given fraction of the dataset.

    Examples
    --------

    >>> small_dataset = methods.sample_rows(dataset, 0.01)

    Notes
    -----

    This method may not sample exactly ``(fraction * n_rows)`` rows from
    the dataset.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset to sample from.
    fraction : :obj:`float`
        (Expected) fraction of rows to keep.
    seed : :obj:`int`
        Random seed.

    Returns
    ------
    :class:`.MatrixTable`
        Downsampled matrix table.
    """

    return MatrixTable(dataset._jvds.sampleVariants(fraction, seed))

@handle_py4j
@typecheck(ds=MatrixTable,
           keep_star=bool,
           left_aligned=bool)
def split_multi_hts(ds, keep_star=False, left_aligned=False):
    """Split multiallelic variants for HTS :meth:`.MatrixTable.entry_schema`:

    .. code-block:: text

      Struct {
        GT: Call,
        AD: Array[!Int32],
        DP: Int32,
        GQ: Int32,
        PL: Array[!Int32].
      }

    For generic genotype schema, use :meth:`.split_multi`.

    Examples
    --------

    >>> methods.split_multi_hts(dataset).write('output/split.vds')

    Notes
    -----

    We will explain by example. Consider a hypothetical 3-allelic
    variant:

    .. code-block:: text

      A   C,T 0/2:7,2,6:15:45:99,50,99,0,45,99

    split_multi will create two biallelic variants (one for each
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

    `GQ` is recomputed from `PL`.

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
    field `aIndex` can be used to select the value corresponding to the split
    allele's position:

    >>> ds = methods.split_multi_hts(dataset)
    >>> ds = ds.filter_rows(ds.info.AC[ds.aIndex - 1] < 10, keep = False)

    VCFs split by Hail and exported to new VCFs may be
    incompatible with other tools, if action is not taken
    first. Since the "Number" of the arrays in split multiallelic
    sites no longer matches the structure on import ("A" for 1 per
    allele, for example), Hail will export these fields with
    number ".".

    If the desired output is one value per site, then it is
    possible to use annotate_variants_expr to remap these
    values. Here is an example:

    >>> ds = methods.split_multi_hts(dataset)
    >>> ds = ds.annotate_rows(info = Struct(AC=ds.info.AC[ds.aIndex - 1], **ds.info)) # doctest: +SKIP
    >>> methods.export_vcf(ds, 'output/export.vcf') # doctest: +SKIP

    The info field AC in *data/export.vcf* will have ``Number=1``.

    **New Fields**

    :meth:`.split_multi_hts` adds the following fields:

     - `wasSplit` (*Boolean*) -- ``True`` if this variant was originally
       multiallelic, otherwise ``False``.

     - `aIndex` (*Int*) -- The original index of this alternate allele in the
       multiallelic representation (NB: 1 is the first alternate allele or the
       only alternate allele in a biallelic variant). For example, 1:100:A:T,C
       splits into two variants: 1:100:A:T with ``aIndex = 1`` and 1:100:A:C
       with ``aIndex = 2``.

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

    hts_genotype_schema = TStruct._from_java(Env.hail().variant.Genotype.htsGenotypeType().deepOptional())
    if ds.entry_schema._deep_optional() != hts_genotype_schema:
        raise FatalError("""
split_multi_hts: entry schema must be the HTS genotype schema
  found: {}
  expected: {}

hint: Use `split_multi` to split entries with a non-HTS genotype schema.
""".format(ds.entry_schema, hts_genotype_schema))

    variant_expr = 'va.aIndex = aIndex, va.wasSplit = wasSplit'
    genotype_expr = '''
g = let
  newgt = downcode(g.GT, aIndex) and
  newad = if (isDefined(g.AD))
      let sum = g.AD.sum() and adi = g.AD[aIndex] in [sum - adi, adi]
    else
      NA: Array[Int] and
  newpl = if (isDefined(g.PL))
      range(3).map(i => range(g.PL.length).filter(j => downcode(Call(j), aIndex) == Call(i)).map(j => g.PL[j]).min())
    else
      NA: Array[Int] and
  newgq = gqFromPL(newpl)
in { GT: newgt, AD: newad, DP: g.DP, GQ: newgq, PL: newpl }
'''
    jds = scala_object(Env.hail().methods, 'SplitMulti').apply(
        ds._jvds, variant_expr, genotype_expr, keep_star, left_aligned)
    return MatrixTable(jds)

@typecheck(dataset=MatrixTable)
def grm(dataset):
    """Compute the Genetic Relatedness Matrix (GRM).

    .. include:: ../_templates/req_tvariant.rst
    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    >>> km = methods.grm(dataset)

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

    Warning
    -------
    Since Hardy-Weinberg normalization cannot be applied to variants that
    contain only reference alleles or only alternate alleles, all such variants
    are removed prior to calcularing the GRM.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset to sample from.

    Returns
    -------
    :class:`.genetics.KinshipMatrix`
        Genetic Relatedness Matrix for all samples.
    """

    dataset = require_biallelic(dataset, "grm")
    dataset = dataset.annotate_rows(AC=agg.sum(dataset.GT.num_alt_alleles()),
                                    n_called=agg.count_where(functions.is_defined(dataset.GT)))
    dataset = dataset.filter_rows((dataset.AC > 0) & (dataset.AC < 2 * dataset.n_called)).persist()

    n_variants = dataset.count_rows()
    if n_variants == 0:
        raise FatalError("Cannot run GRM: found 0 variants after filtering out monomorphic sites.")
    info("Computing GRM using {} variants.".format(n_variants))

    normalized_genotype_expr = functions.bind(
        dataset.AC / dataset.n_called,
        lambda mean_gt: functions.cond(functions.is_defined(dataset.GT),
                                       (dataset.GT.num_alt_alleles() - mean_gt) /
                                       functions.sqrt(mean_gt * (2 - mean_gt) * n_variants / 2),
                                       0))

    bm = BlockMatrix.from_matrix_table(normalized_genotype_expr)
    dataset.unpersist()
    grm = bm.T.dot(bm)

    return KinshipMatrix._from_block_matrix(TString(),
                                      grm,
                                      [row.s for row in dataset.cols_table().select('s').collect()],
                                      n_variants)

@handle_py4j
@typecheck(call_expr=CallExpression)
def rrm(call_expr):
    """Computes the Realized Relationship Matrix (RRM).

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    >>> kinship_matrix = methods.rrm(dataset['GT'])

    Notes
    -----

    The Realized Relationship Matrix is defined as follows. Consider the
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

    Note that the only difference between the Realized Relationship Matrix and
    the Genetic Relationship Matrix (GRM) used in :func:`.methods.grm` is the
    variant (column) normalization: where RRM uses empirical variance, GRM uses
    expected variance under Hardy-Weinberg Equilibrium.


    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Expression on a :class:`.MatrixTable` that gives the genotype call.

    Returns
        :return: Realized Relationship Matrix for all samples.
        :rtype: :class:`.KinshipMatrix`
    """
    source = call_expr._indices.source
    if not isinstance(source, MatrixTable):
        raise ValueError("Expect an expression of 'MatrixTable', found {}".format(
            "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'))
    dataset = source
    base, _ = dataset._process_joins(call_expr)
    analyze('rrm', call_expr, dataset._entry_indices)

    dataset = require_biallelic(dataset, 'rrm')

    call_expr._indices = dataset._entry_indices
    gt_expr = call_expr.num_alt_alleles()
    dataset = dataset.annotate_rows(AC=agg.sum(gt_expr),
                                    ACsq=agg.sum(gt_expr * gt_expr),
                                    n_called=agg.count_where(functions.is_defined(call_expr)))

    dataset = dataset.filter_rows((dataset.AC > 0) &
                                  (dataset.AC < 2 * dataset.n_called) &
                                  ((dataset.AC != dataset.n_called) |
                                   (dataset.ACsq != dataset.n_called)))

    n_samples = dataset.count_cols()
    n_variants = dataset.count_rows()
    if n_variants == 0:
        raise FatalError("Cannot run RRM: found 0 variants after filtering out monomorphic sites.")
    info("Computing RRM using {} variants.".format(n_variants))

    call_expr._indices = dataset._entry_indices
    gt_expr = call_expr.num_alt_alleles()
    normalized_genotype_expr = functions.bind(
        dataset.AC / dataset.n_called,
        lambda mean_gt: functions.bind(
            functions.sqrt((dataset.ACsq +
                            (n_samples - dataset.n_called) * mean_gt ** 2) /
                           n_samples - mean_gt ** 2),
            lambda stddev: functions.cond(functions.is_defined(call_expr),
                                          (gt_expr - mean_gt) / stddev, 0)))

    bm = BlockMatrix.from_matrix_table(normalized_genotype_expr)
    dataset.unpersist()
    rrm = bm.T.dot(bm) / n_variants

    return KinshipMatrix._from_block_matrix(TString(),
                                            rrm,
                                            [row.s for row in dataset.cols_table().select('s').collect()],
                                            n_variants)

@handle_py4j
@typecheck(populations=integral,
           samples=integral,
           variants=integral,
           num_partitions=nullable(integral),
           pop_dist=nullable(listof(numeric)),
           fst=nullable(listof(numeric)),
           af_dist=oneof(UniformDist, BetaDist, TruncatedBetaDist),
           seed=integral,
           reference_genome=nullable(GenomeReference))
def balding_nichols_model(populations, samples, variants, num_partitions=None,
                          pop_dist=None, fst=None, af_dist=UniformDist(0.1, 0.9),
                          seed=0, reference_genome=None):
    if pop_dist is None:
        jvm_pop_dist_opt = joption(pop_dist)
    else:
        jvm_pop_dist_opt = joption(jarray(Env.jvm().double, pop_dist))

    if fst is None:
        jvm_fst_opt = joption(fst)
    else:
        jvm_fst_opt = joption(jarray(Env.jvm().double, fst))

    rg = reference_genome if reference_genome else Env.hc().default_reference
    
    jmt = Env.hc()._jhc.baldingNicholsModel(populations, samples, variants,
                                            joption(num_partitions),
                                            jvm_pop_dist_opt,
                                            jvm_fst_opt,
                                            af_dist._jrep(),
                                            seed,
                                            rg._jrep)
    return MatrixTable(jmt)


class FilterAlleles(object):
    """Filter out a set of alternate alleles.  If all alternate alleles of
    a variant are filtered out, the variant itself is filtered out.
    `filter_expr` is an alternate allele indexed `Array[Boolean]`
    where the booleans, combined with `keep`, determine which
    alternate alleles to filter out.

    This object has bindings for values used in the filter alleles
    process: `old_to_new`, `new_to_old` and `new_v`.  Call
    :meth:`.FilterAlleles.annotate_rows` and/or
    :meth:`.FilterAlleles.annotate_entries` to update row-indexed and
    row- and column-indexed fields for the new, allele filtered
    variant.  Finally, call :meth:`.FilterAlleles.filter` to perform
    filter alleles.

    Examples
    --------
    
    Filter alleles with zero AC count on a dataset with the HTS entry
    schema and update the ``info.AC`` and entry fields.

    >>> fa = methods.FilterAlleles(dataset.info.AC.map(lambda AC: AC == 0), keep=False)
    ... fa.annotate_rows(
    ...     info = dataset.info.annotate(AC = fa.new_to_old[1:].map(lambda i: dataset.info.AC[i - 1])))
    ... newPL = functions.cond(
    ...     functions.is_defined(dataset.PL),
    ...     functions.range(0, fa.new_v.num_genotypes()).map(
    ...         lambda newi: functions.bind(
    ...             functions.call(newi),
    ...             lambda newc: dataset.PL[functions.gt_index(fa.new_to_old[newc.gtj()], fa.new_to_old[newc.gtk()])])),
    ...     functions.null(TArray(TInt32())))
    ... fa.annotate_entries(
    ...     GT = functions.call(functions.gt_from_pl(newPL)),
    ...     AD = functions.cond(
    ...         functions.is_defined(dataset.AD),
    ...         functions.range(0, fa.new_v.num_alleles()).map(
    ...             lambda newi: dataset.AD[fa.new_to_old[newi]]),
    ...         functions.null(TArray(TInt32()))),
    ...     GQ = functions.gq_from_pl(newPL),
    ...     PL = newPL)
    ... filtered_result = fa.filter()
    
    Parameters
    ----------
    filter_expr : :class:`.ArrayBooleanExpression`
        Boolean filter expression.
    keep : bool
        If ``True``, keep alternate alleles where the corresponding
        element of `filter_expr` is ``True``.  If False, remove the
        alternate alleles where the corresponding element is ``True``.
    left_aligned : bool
        If ``True``, variants are assumed to be left aligned and have
        unique loci.  This avoids a shuffle.  If the assumption is
        violated, an error is generated.
    keep_star : bool
        If ``True``, keep variants where the only unfiltered alternate
        alleles are ``*`` alleles.
    """
    
    @typecheck_method(filter_expr=ArrayBooleanExpression, keep=bool, left_aligned=bool, keep_star=bool)
    def __init__(self, filter_expr, keep=True, left_aligned=False, keep_star=False):
        source = filter_expr._indices.source
        if not isinstance(source, MatrixTable):
            raise ValueError("Expect an expression of 'MatrixTable', found {}".format(
                "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'))
        ds = source

        analyze('FilterAlleles', filter_expr, ds._row_indices)

        self._ds = ds
        self._filter_expr = filter_expr
        self._keep = keep
        self._left_aligned = left_aligned
        self._keep_star = keep_star
        self._row_exprs = None
        self._entry_exprs = None

        self._old_to_new = construct_reference('oldToNew', TArray(TInt32()), ds._row_indices)
        self._new_to_old = construct_reference('newToOld', TArray(TInt32()), ds._row_indices)
        self._new_v = construct_reference('newV', ds._rowkey_schema, ds._row_indices)

    @property
    def new_to_old(self):
        """The array of old allele indices, such that ``new_to_old[newIndex] =
        oldIndex`` and ``newToOld[0] = 0``.  A row-indexed expression.
        
        Returns
        -------
        :class:`.ArrayInt32Expression`
            The array of old indices.
        """
        return self._new_to_old

    @property
    def old_to_new(self):
        """The array of new allele indices.  All old filtered alleles have new
        index 0.  A row-indexed expression.
        
        Returns
        -------
        :class:`.ArrayInt32Expression`
            The array of new indices.
        """
        return self._old_to_new

    @property
    def new_v(self):
        """The new, allele-filtered variant.  A row-indexed expression.
        
        Returns
        -------
        :class:`.VariantExpression`
            The new, allele-filtered variant.
        """
        return self._new_v

    def annotate_rows(self, **named_exprs):
        """Create or update row-indexed fields for the new, allele filtered
        variant.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Field names and the row-indexed expressions to compute them.
        """
        if self._row_exprs:
            raise RuntimeError('annotate_rows already called')
        for k, v in named_exprs.items():
            analyze('FilterAlleles', v, base._row_indices)
        self._row_exprs = named_exprs

    def annotate_entries(self, **named_exprs):
        """Create or update row- and column-indexed fields (entry fields) for
        the new, allele filtered variant.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Field names and the row- and column-indexed expressions to
            compute them.
        """
        if self._entry_exprs:
            raise RuntimeError('annotate_entries already called')
        for k, v in named_exprs.items():
            analyze('FilterAlleles', v, base._entry_indices)
        self._entry_exprs = named_exprs

    def filter(self):
        """Perform the filter alleles, returning a new matrix table.

        Returns
        -------
        :class:`.MatrixTable`
            Returns a matrix table with alleles filtered.
        """
        if not self._row_exprs:
            self._row_exprs = {}
        if not self._entry_exprs:
            self._entry_exprs = {}
        
        base, cleanup = self._ds._process_joins(*(
            [self._filter_expr] + self._row_exprs.values() + self._entry_exprs.values()))
        
        filter_hql = self._filter_expr._ast.to_hql()
        
        row_hqls = []
        for k, v in self._row_exprs.items():
            row_hqls.append('va.`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))
            base._check_field_name(k, base._row_indices)
        row_hql = ',\n'.join(row_hqls)

        entry_hqls = []
        for k, v in self._entry_exprs.items():
            entry_hqls.append('g.`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))
            base._check_field_name(k, base._entry_indices)
        entry_hql = ',\n'.join(entry_hqls)

        m = MatrixTable(
            Env.hail().methods.FilterAlleles.apply(
                base._jvds, '({p})[aIndex - 1]'.format(p=filter_hql), row_hql, entry_hql, self._keep, self._left_aligned, self._keep_star))
        return cleanup(m)

@handle_py4j
@typecheck(ds=MatrixTable,
           num_cores=integral,
           r2=numeric,
           window=integral,
           memory_per_core=integral)
def ld_prune(ds, num_cores, r2=0.2, window=1000000, memory_per_core=256):
    jmt = Env.hail().methods.LDPrune.apply(ds._jvds, num_cores, r2, window, memory_per_core)
    return MatrixTable(jmt)

@handle_py4j
@typecheck(ds=MatrixTable,
           left_aligned=bool)
def min_rep(ds, left_aligned=False):
    """Gives minimal, left-aligned representation of alleles. 

    .. include:: ../_templates/req_tvariant.rst

    Notes
    -----
    Note that this can change the variant position.

    Examples
    --------

    Simple trimming of a multi-allelic site, no change in variant
    position `1:10000:TAA:TAA,AA` => `1:10000:TA:T,A`

    Trimming of a bi-allelic site leading to a change in position
    `1:10000:AATAA,AAGAA` => `1:10002:T:G`

    Parameters
    ----------
    left_aligned : bool
        If ``True``, variants are assumed to be left aligned and have
        unique loci.  This avoids a shuffle.  If the assumption is
        violated, an error is generated.

    Returns
    -------
    :class:`.MatrixTable`
    """
    return MatrixTable(ds._jvds.minRep(left_aligned))
