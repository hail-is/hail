import abc

import hail
import hail as hl
from hail.expr.expressions import *
from hail.table import Table
from hail.typecheck import *
from hail.utils import wrap_to_list
from hail.utils.java import *


def _warn_if_no_intercept(caller, covariates):
    if all([e._indices.axes for e in covariates]):
        warn(f'{caller}: model appears to have no intercept covariate.'
             '\n    To include an intercept, add 1.0 to the list of covariates.')
        return True
    return False


class RegressionModel(object):
    """Base class for regression models in Hail.

    Notes
    -----
    The standard regression models are:

     - :class:`.LinearRegressionModel`
     - :class:`.LogisticRegressionModel`
     - :class:`.LinearMixedModel`
    """
    @abc.abstractmethod
    def regress_rows(self, *args, **kwargs) -> hail.Table:
        ...


class LinearRegressionModel(RegressionModel):
    """Model object for linear regression."""
    def __init__(self):
        super(LinearRegressionModel, self).__init__()

    @typecheck_method(y=oneof(expr_float64, sequenceof(expr_float64)),
                      x=expr_float64,
                      covariates=sequenceof(expr_float64),
                      block_size=int)
    def regress_rows(self, y, x, covariates, block_size=16) -> hail.Table:
        """For each row, test an input variable for association with
        response variables using linear regression.

        Examples
        --------

        >>> model = hl.LinearRegressionModel()
        >>> result_ht = model.regress_rows(
        ...     y=dataset.pheno.height,
        ...     x=dataset.GT.n_alt_alleles(),
        ...     covariates=[1, dataset.pheno.age, dataset.pheno.is_female])

        Warning
        -------
        :meth:`.LinearRegressionModel.regress_rows` considers the same set of
        columns (i.e., samples, points) for every response variable and row,
        namely those columns for which **all** response variables and covariates
        are defined. For each row, missing values of `x` are mean-imputed over
        these columns. As in the example, the intercept covariate ``1`` must be
        included **explicitly** if desired.

        Notes
        -----
        With the default root and `y` a single expression, the following row-indexed
        fields are added.

        - **<row key fields>** (Any) -- Row key fields.
        - **n** (:py:data:`.tint32`) -- Number of columns used.
        - **sum_x** (:py:data:`.tfloat64`) -- Sum of input values `x`.
        - **y_transpose_x** (:py:data:`.tfloat64`) -- Dot product of response
          vector `y` with the input vector `x`.
        - **beta** (:py:data:`.tfloat64`) --
          Fit effect coefficient of `x`, :math:`\hat\\beta_1` below.
        - **standard_error** (:py:data:`.tfloat64`) --
          Estimated standard error, :math:`\widehat{\mathrm{se}}_1`.
        - **t_stat** (:py:data:`.tfloat64`) -- :math:`t`-statistic, equal to
          :math:`\hat\\beta_1 / \widehat{\mathrm{se}}_1`.
        - **p_value** (:py:data:`.tfloat64`) -- :math:`p`-value.

        If `y` is a list of expressions, then the last five fields instead have type
        :py:data:`.tarray` of :py:data:`.tfloat64`, with corresponding indexing of
        the list and each array.

        If `y` is a doubly-nested list of expressions, then the last five fields instead have type
        ``array<array<float64>>``, with corresponding indexing of
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
        :math:`n - k - 1` degrees of freedom, under the null hypothesis of no
        effect, with :math:`n` samples and :math:`k` covariates in addition to
        ``x``.

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

        Returns
        -------
        :class:`.Table`
        """
        mt = matrix_table_source('LinearRegressionModel.regress_rows/x', x)
        check_entry_indexed('LinearRegressionModel.regress_rows/x', x)

        y_is_list = isinstance(y, list)

        all_exprs = []
        y = wrap_to_list(y)
        for e in y:
            all_exprs.append(e)
            analyze('LinearRegressionModel.regress_rows/y', e, mt._col_indices)
        for e in covariates:
            all_exprs.append(e)
            analyze('LinearRegressionModel.regress_rows/covariates', e, mt._col_indices)

        _warn_if_no_intercept('LinearRegressionModel.regress_rows', covariates)

        x_field_name = Env.get_uid()
        y_field_names = list(f'__y{i}' for i in range(len(y)))
        cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))

        # FIXME: selecting an existing entry field should be emitted as a SelectFields
        mt = mt._select_all(col_exprs=dict(**dict(zip(y_field_names, y)),
                                           **dict(zip(cov_field_names, covariates))),
                            row_exprs=dict(mt.row_key),
                            col_key=[],
                            entry_exprs={x_field_name: x})

        jt = Env.hail().methods.LinearRegression.regress_rows(
            mt._jvds,
            jarray(Env.jvm().java.lang.String, y_field_names),
            x_field_name,
            jarray(Env.jvm().java.lang.String, cov_field_names),
            block_size)

        ht_result = Table(jt)

        if not y_is_list:
            fields = ['y_transpose_x', 'beta', 'standard_error', 't_stat', 'p_value']
            ht_result = ht_result.annotate(**{f: ht_result[f][0] for f in fields})

        return ht_result


class LogisticRegressionModel(RegressionModel):
    """Model object for logistic regression."""

    @typecheck_method(test=enumeration('wald', 'lrt', 'score', 'firth'))
    def __init__(self, test):
        self.test = test
        super(LogisticRegressionModel, self).__init__()

    @typecheck_method(y=expr_float64,
                      x=expr_float64,
                      covariates=sequenceof(expr_float64))
    def regress_rows(self, y, x, covariates) -> hail.Table:
        r"""For each row, test an input variable for association with a
        binary response variable using logistic regression.

        Examples
        --------
        Run the logistic regression Wald test per variant using a Boolean
        phenotype, intercept and two covariates stored in column-indexed
        fields:

        >>> model = hl.LogisticRegressionModel('wald')
        >>> result_ht = model.regress_rows(
        ...     y=dataset.pheno.is_case,
        ...     x=dataset.GT.n_alt_alleles(),
        ...     covariates=[1, dataset.pheno.age, dataset.pheno.is_female])

        Warning
        -------
        :meth:`.LogisticRegressionModel.regress_rows` considers the same set of
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

        Parameters
        ----------
        y : :class:`.Float64Expression`
            Column-indexed response expression.
            All non-missing values must evaluate to 0 or 1.
            Note that a :class:`.BooleanExpression` will be implicitly converted to
            a :class:`.Float64Expression` with this property.
        x : :class:`.Float64Expression`
            Entry-indexed expression for input variable.
        covariates : :obj:`list` of :class:`.Float64Expression`
            Non-empty list of column-indexed covariate expressions.

        Returns
        -------
        :class:`.Table`
        """
        if len(covariates) == 0:
            raise ValueError('logistic regression requires at least one covariate expression')

        mt = matrix_table_source('LogisticRegressionModel.regress_rows/x', x)
        check_entry_indexed('LogisticRegressionModel.regress_rows/x', x)

        analyze('LogisticRegressionModel.regress_rows/y', y, mt._col_indices)

        all_exprs = [y]
        for e in covariates:
            all_exprs.append(e)
            analyze('logistic_regression/covariates', e, mt._col_indices)

        _warn_if_no_intercept('LogisticRegressionModel.regress_rows', covariates)

        x_field_name = Env.get_uid()
        y_field_name = '__y'
        cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))

        # FIXME: selecting an existing entry field should be emitted as a SelectFields
        mt = mt._select_all(col_exprs=dict(**{y_field_name: y},
                                           **dict(zip(cov_field_names, covariates))),
                            row_exprs=dict(mt.row_key),
                            col_key=[],
                            entry_exprs={x_field_name: x})

        jt = Env.hail().methods.LogisticRegression.regress_rows(
            mt._jvds,
            self.test,
            y_field_name,
            x_field_name,
            jarray(Env.jvm().java.lang.String, cov_field_names))
        return Table(jt)

class PoissonRegressionModel(RegressionModel):
    """Model object for Poisson regression."""

    @typecheck_method(test=enumeration('wald', 'lrt', 'score'))
    def __init__(self, test):
        self.test = test
        super(PoissonRegressionModel, self).__init__()

    @typecheck_method(y=expr_float64,
               x=expr_float64,
               covariates=sequenceof(expr_float64))
    def regress_rows(self, test, y, x, covariates) -> Table:
        r"""For each row, test an input variable for association with a
        count response variable using `Poisson regression <https://en.wikipedia.org/wiki/Poisson_regression>`__.

        Notes
        -----
        See :meth:`.LogisticRegressionModel.regress_rows` for more info on statistical tests
        of general linear models.

        Parameters
        ----------
        y : :class:`.Float64Expression`
            Column-indexed response expression.
            All non-missing values must evaluate to a non-negative integer.
        x : :class:`.Float64Expression`
            Entry-indexed expression for input variable.
        covariates : :obj:`list` of :class:`.Float64Expression`
            Non-empty list of column-indexed covariate expressions.

        Returns
        -------
        :class:`.Table`
        """
        if len(covariates) == 0:
            raise ValueError('Poisson regression requires at least one covariate expression')

        mt = matrix_table_source('PoissonRegressionModel.regress_rows/x', x)
        check_entry_indexed('PoissonRegressionModel.regress_rows/x', x)

        analyze('PoissonRegressionModel.regress_rows/y', y, mt._col_indices)

        all_exprs = [y]
        for e in covariates:
            all_exprs.append(e)
            analyze('PoissonRegressionModel.regress_rows/covariates', e, mt._col_indices)

        _warn_if_no_intercept('PoissonRegressionModel.regress_rows', covariates)

        x_field_name = Env.get_uid()
        y_field_name = '__y'
        cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))

        # FIXME: selecting an existing entry field should be emitted as a SelectFields
        mt = mt._select_all(col_exprs=dict(**{y_field_name: y},
                                           **dict(zip(cov_field_names, covariates))),
                            row_exprs=dict(mt.row_key),
                            col_key=[],
                            entry_exprs={x_field_name: x})

        jt = Env.hail().methods.PoissonRegression.regress_rows(
            mt._jvds,
            self.test,
            y_field_name,
            x_field_name,
            jarray(Env.jvm().java.lang.String, cov_field_names))
        return Table(jt)
