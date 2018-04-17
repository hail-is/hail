import numpy as np
from hail.typecheck import *


class LinearMixedModel(object):
    r"""Class representing a linear mixed model.

    .. include:: ../_templates/experimental.rst

    This class represents a linear model with two variance components:

    .. math::

        y \sim \mathrm{N}\left(C \beta, \sigma^2 K + \tau^2 I\right)

    where :math:`\mathrm{N}` is an math:`n`-dimensional normal distribution and

    - :math:`y` is a known vector of :math:`n` observations.
    - :math:`C` is a known :math:`n \times k` design matrix for math:`k` fixed effects.
    - :math:`K` is a known :math:`n \times n` positive semi-definite kernel.
    - :math:`I` is the :math:`n \times n` identity matrix.
    - :math:`\beta` is a :math:`k`-parameter vector of fixed effects.
    - :math:`\sigma^2` is a the variance parameter on :math:`K`.
    - :math:`\tau^2` is a the variance parameter on :math:`I`.

    In other words, the residual covariance of observations :math:`y_i` and
    :math:`y_j` is :math:`\sigma^2 K_{ij} + \tau^2` if math:`i=j` and
    :math:`\sigma^2 K_{ij}` otherwise.

    This model is equivalent to a
    `mixed model <https://en.wikipedia.org/wiki/Mixed_model>`__
    of the form

    .. math::

        y = C \beta + Z \nu + \epsilon

    via :math:`K = ZZ^T`, where

    - :math:`Z` is a known :math:`n \times m` design matrix for :math:`m` random effects.
    - :math:`\nu` is a :math:`m`-vector of random effects drawn from :math:`\mathrm{N}\left(0, \sigma^2 I\right)`.
    - :math:`\epsilon` is a :math:`n`-vector of random errors drawn from :math:`\mathrm{N}\left(0, \tau^2 I\right)`.

    However, this class does not construct :math:`K` as the linear kernel
    of such a mixed model. Rather, it takes as input
    the images :math:`Py` and :math:`PC` of :math:`y` and :math:`C`, respectively,
    under the decorrelation transformation :math:`P` given by eigenvectors of
    :math:`K`, as well as the corresponding eigenvalues.

    More precisely, the model is equivalent to the
    linear model with diagonal covariance

    .. math::

        Py \sim \mathrm{N}\left((PC) \beta, \sigma_g^2 (\gamma S + I)\right)

    where

    - :math:`K` has eigendecomposition :math:`USU^T`.
    - :math:`P: \mathbb{R}^n \rightarrow \mathbb{R}^r` is the isometry :math:`U^T`.
    - :math:`\gamma` is the ratio of variance parameters :math:`\frac{\sigma^2}{\tau^2}`.

    Hence, the triple :math:`(Py, PC, S)` is sufficient to fit the original model,
    and we term this approach the `full-rank strategy`.

    The class also provides an efficient `low-rank strategy` to fit the original
    model with :math:`K` replaced by its rank-:math:`r` approximation. In this case,
    the quintuple :math:`(P_r y, P_r C, S_r, y, C)` is sufficient , where

    - :math:`P_r: \mathbb{R}^n \rightarrow \mathbb{R}^r` is the projection
      corresponding to the top :math:`r` eigenvectors.
    - :math:`S_r` is the corresponding vector of non-zero eigenvalues.

    Note that when :math:`K` actually has rank :math:`r`, this 
    low-rank strategy is not an approximation.
    This situation arises, for example, when :math:`K` is the linear
    kernel of a mixed model with fewer random effects than observations.

    The following attributes are set at initialization for the low-rank strategy.
    For the full-rank strategy, `y` and `c` are not set and :math:`r` must equal :math:`n`.

    .. list-table::
      :header-rows: 1

      * - Attribute
        - Type
        - Value
      * - `n`
        - int
        - Number of observations
      * - `k`
        - int
        - Number of fixed effects (covariates)
      * - `r`
        - int
        - Number of mixed effects
      * - `py`
        - numpy.ndarray
        - Projected vector :math:`Py` with shape :math:`(r)`
      * - `pc`
        - numpy.ndarray
        - Projected design matrix :math:`PC` with shape :math:`(r, k)`
      * - `s`
        - numpy.ndarray
        - Eigenvalues vector of projection with shape :math:`(r)`
      * - `y`
        - numpy.ndarray
        - Response vector with shape :math:`(n)`
      * - `c`
        - numpy.ndarray
        - Design matrix with shape :math:`(n, k)`

    :meth:`fit` uses `restricted maximum likelihood
    <https://en.wikipedia.org/wiki/Restricted_maximum_likelihood>`__ (REML)
    to estimate :math:`(\beta, \sigma^2, \tau^2)`,
    adding the following attributes at this estimate.

    .. list-table::
      :header-rows: 1

      * - Attribute
        - Type
        - Value
      * - `beta`
        - numpy.ndarray
        - :math:`\beta`
      * - `sigma_sq`
        - float
        - :math:`\sigma^2`
      * - `tau_sq`
        - float
        - :math:`\tau^2`
      * - `gamma`
        - float
        - :math:`\gamma = \frac{\sigma^2}{\tau^2}`
      * - `log_gamma`
        - float
        - :math:`\log{\gamma}`
      * - `h_sq`
        - float
        - :math:`\mathit{h}^2 = \frac{\sigma^2}{\sigma^2 + \tau^2}`
      * - `log_reml`
        - float
        - log of REML

    Estimation proceeds by minimizing the function :meth:`compute_neg_log_reml`
    with respect to the parameter :math:`\log{\gamma}` governing the (log)
    ratio of the variance parameters :math:`\sigma^2` and :math:`\tau^2`. For
    any fixed ratio, the REML estimate and value have closed form solutions.

    The above linear model is equivalent to its augmentation by an
    additional covariate :math:`x` under the null hypothesis that the corresponding
    fixed effect is zero. Once :meth:`fit` has been used to set :math:`\log{\gamma}` for the null model,
    the methods :meth:`fit_local_rows` and :meth:`fit_rows`
    may be used to test this null hypothesis for each of a collection of rows
    using the likelihood ratio test with both the null and alternative models
    constrained by :math:`\log{\gamma}`.
    The test statistic :math:`\chi^2` is given by :math:`n` times the log ratio of
    the squared residuals and follows a chi-squared distribution with one degree of freedom.

    For these methods, the vector :math:`Px` is sufficient for the full-rank strategy, whereas the low-rank strategy
    depends on both :math:`P_r x` and :math:`x`.

    Parameters
    ----------
    py: :class:`numpy.ndarray`
        Projected response vector.
    pc: :class:`numpy.ndarray`
        Projected design matrix.
    s: :class:`numpy.ndarray`
        Eigenvalues vector.
    y: :class:`numpy.ndarray`, optional.
        Response vector, included for low-rank strategy.
    c: :class:`numpy.ndarray`, optional.
        Design matrix, included for low-rank strategy.
    """
    @typecheck_method(y=np.ndarray, c=np.ndarray, py=np.ndarray, pc=np.ndarray, s=np.ndarray)
    def __init__(self, py, pc, s, y=None, c=None):
        assert y.ndim == 1 and c.ndim == 2 and py.ndim == 1 and pc.ndim == 2 and s.ndim == 1

        self.n, self.k = c.shape
        self.dof = self.n - self.k
        self.r = s.shape[0]

        assert self.dof > 0 and self.n > self.r > 0
        assert y.size == self.n
        assert py.size == self.r
        assert pc.shape == (self.r, self.k)

        self.y = y
        self.c = c
        self.py = py
        self.pc = pc
        self.s = s

        # constants for fit
        self._yty = y.T @ y
        self._cty = c.T @ y
        self._ctc = c.T @ c
        self._shift = self.dof * (1 + np.log(2 * np.pi)) - np.linalg.slogdet(self._ctc)[1]

        self._reset()

    def _reset(self):
        self._fitted = False

        # set by fit
        self.beta = None
        self.gamma = None
        self.h_sq = None
        self.log_gamma = None
        self.log_reml = None
        self.optimize_result = None
        self.sigma_sq = None
        self.tau_sq = None

        # constants used by fit_rows / fit_local_rows
        self._z = None
        self._ydy = None
        self._cdy = None
        self._cdc = None
        self._residual_sq = None

        # scala class used by fit_rows
        self._scala_model = None

    def compute_neg_log_reml(self, log_gamma, return_parameters=False):
        r"""Compute negative log REML constrained to a fixed value
        of :math:`\log{\gamma}`.

        Parameters
        ----------
        log_gamma: :obj:`float`
            Value of :math:`\log{\gamma}`.
        return_parameters:
            If ``True``, also return :math:`\beta`, :math:`\sigma_sq`,
            and :math:`\tau^2`.

        Returns
        -------
        :obj:`float` or (:obj:`float`, :class:`numpy.ndarray`, :obj:`float`, :obj:`float`)
            If ``return_parameters==False``, returns negative REML.
            Otherwise, returns negative REML, :math:`\beta`, :math:`\sigma_sq`,
            and :math:`\tau^2`.
        """
        from scipy.linalg import solve, LinAlgError

        gamma = np.exp(log_gamma)

        d = self.s + 1 / gamma
        z = 1 / d - gamma
        zpy = z * self.py

        ydy = self._yty * gamma + self.py @ zpy
        cdy = self._cty * gamma + self.pc.T @ zpy
        cdc = self._ctc * gamma + (self.pc.T * z) @ self.pc

        try:
            beta = solve(cdc, cdy, assume_a='pos', overwrite_a=True)
            residual_sq = ydy - cdy.T @ beta
            sigma_sq = residual_sq / self.dof
            tau_sq = sigma_sq / gamma

            logdet_d = np.sum(np.log(d)) + (self.r - self.n) * log_gamma
            neg_log_reml = (logdet_d + np.linalg.slogdet(cdc)[1] + self.dof * np.log(sigma_sq) + self._shift) / 2

            self._z, self._ydy, self._cdy, self._cdc = z, ydy, cdy, cdc

            if return_parameters:
                return neg_log_reml, beta, sigma_sq, tau_sq
            else:
                return neg_log_reml
        except LinAlgError as e:
            raise Exception(f'Linear algebra error while solving for REML estimate:\n  {e}')

    @typecheck_method(log_gamma=nullable(float), bounds=tupleof(numeric), tol=float, maxiter=int)
    def fit(self, log_gamma=None, bounds=(-8.0, 8.0), tol=1e-8, maxiter=500):
        r"""Find the triple :math:`(\beta, \sigma^2, \tau^2)`` maximizing REML.

        If `log_gamma` is provided, :meth:`fit` finds the REML solution
        with :math:`\log{\gamma}` constrained to this value.

        Otherwise, :meth:`fit` first uses the ``bounded`` method of
        `scipy.optimize.minimize_scalar <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html>`__
        to find the value of `:math:`\log{\gamma}` that minimizes
        :meth:`compute_neg_log_reml`. The attribute
        `optimize_result` is set to the returned value of type
        ``scipy.optimize.OptimizeResult``.

        Either way, :meth:`fit` computes and sets the attributes `beta`, `gamma`, `log_gamma`,
        `h_sq`, `log_reml`, `sigma_sq`, and `tau_sq` as described in the top-level
        class documentation.

        Parameters
        ----------
        log_gamma: :obj:`float`, optional.
            If provided, the solution is constrained to have this value of
            :math:`\log{\gamma}`.
        bounds: :obj:`float`, :obj:`float`
            Lower and upper bounds for :math:`\log{\gamma}`.
        tol: :obj:`float`
            Absolute tolerance for optimizing :math:`\log{\gamma}`.
        maxiter: :obj:`float`
            Maximum number of iterations for optimizing :math:`\log{\gamma}`.
        """
        if self._fitted:
            self._reset()

        if log_gamma:
            self.log_gamma = log_gamma
        else:
            from scipy.optimize import minimize_scalar

            self.optimize_result = minimize_scalar(
                self.compute_neg_log_reml,
                method='bounded',
                bounds=bounds,
                options={'xatol': tol, 'maxiter': maxiter})

            if self.optimize_result.success:
                if self.optimize_result.x - bounds[0] < tol:
                    raise Exception("Failed to fit log_gamma: optimum at lower bound.")
                elif bounds[1] - self.optimize_result.x < tol:
                    raise Exception("Failed to fit log_gamma: optimum at upper bound.")
                else:
                    self.log_gamma = self.optimize_result.x
            else:
                raise Exception(f'Failed to fit log_gamma:\n  {self.optimize_result}')

        neg_log_reml, self.beta, self.sigma_sq, self.tau_sq = (
            self.compute_neg_log_reml(self.log_gamma, return_parameters=True)
        )
        self.log_reml = -neg_log_reml
        self.gamma = np.exp(self.log_gamma)
        self.h_sq = self.sigma_sq / (self.sigma_sq + self.tau_sq)

        self._residual_sq = self.sigma_sq * self.dof

        cdy = np.zeros(self.k + 1)
        cdy[1:] = self._cdy
        self._cdy = cdy

        cdc = np.zeros((self.k + 1, self.k + 1))
        cdc[1:, 1:] = self._cdc
        self._cdc = cdc

        self._fitted = True

    @typecheck_method(x=np.ndarray, px=np.ndarray)
    def test_local_cols(self, x, px):
        """Test each column locally. # FIXME

        The resulting pandas DataFrame has the following fields.

        .. list-table::
          :header-rows: 1

          * - Field
            - Type
            - Value
          * - `beta`
            - float
            - :math:`\beta`
          * - `sigma_sq`
            - float
            - :math:`\sigma^2`
          * - `chi_sq`
            - float
            - :math:`\chi^2`
          * - `p_value`
            - float
            - p-value

        Parameters
        ----------
        x: :class:`numpy.ndarray`
            Matrix with shape :math:`(n,m)`.  // FIXME
        px: :class:`numpy.ndarray`
            Projected matrix with shape :math:`(r,m)`.
        Returns
        -------
        :class:`pandas.DataFrame`
            Data frame of results per column.
        """
        import pandas as pd

        if not self._fitted:
            raise Exception("Ratio of variance parameters is undefined. Run 'fit' first.")

        n_rows = x.shape[1]
        assert px.shape[1] == n_rows
        assert px.shape[0] == self.r

        data = [self._fit_local_row(x[:, i], px[:, i]) for i in range(n_rows)]  # could parallelize on master

        return pd.DataFrame.from_records(data, columns=['beta', 'sigma_sq', 'chi_sq', 'p_value'])

    @typecheck_method(path_xt=str, path_pxt=str, partition_size=int)
    def fit_rows(self, path_xt, path_pxt, partition_size):
        """Test each row. # FIXME

        The resulting Table has the following fields:

        .. list-table::
          :header-rows: 1

          * - Field
            - Type
            - Value
          * - `idx`
            - int64
            - :math:`\beta`
          * - `beta`
            - float64
            - :math:`\beta`
          * - `sigma_sq`
            - float64
            - :math:`\sigma^2`
          * - `chi_sq`
            - float64
            - :math:`\chi^2`
          * - `p_value`
            - float64
            - p-value

        Parameters
        ----------
        path_xt: :obj:`str`
            Path to block matrix with transpose of shape :math:`(n,m)`.  FIXME: written row-major, expose row matrix?
        path_pxt: :obj:`str`
            Path to projected block matrix with transpose of shape :math:`(r,m)`.

        Returns
        -------
        :class:`pandas.DataFrame`
            Data frame of results per row.
        """
        from hail.table import Table

        if partition_size <= 0:
            raise ValueError("partition_size must be positive")

        if not self._scala_model:
            self._set_scala_model()

        return Table(self._scala_model.fit(path_xt, path_pxt, partition_size))

    def _fit_local_row(self, x, px):
        from scipy.linalg import solve, LinAlgError
        from scipy.stats.distributions import chi2

        gamma = self.gamma
        zpx = self._z * px

        cdy = np.copy(self._cdy)
        cdy[0] = (self.y @ x) * gamma + self.py @ zpx

        cdc = np.copy(self._cdc)
        cdc[0, 0] = (x @ x) * gamma + px @ zpx
        cdc[0, 1:] = (self.c.T @ x) * gamma + self.pc.T @ zpx  # only using upper triangle

        try:
            beta = solve(cdc, cdy, assume_a='pos', overwrite_a=True)
            residual_sq = self._ydy - cdy.T @ beta
            sigma_sq = residual_sq / (self.dof - 1)
            chi_sq = self.n * np.log(self._residual_sq / residual_sq)  # division => precision
            p_value = chi2.sf(chi_sq, 1)

            return beta[0], sigma_sq, chi_sq, p_value
        except LinAlgError:
            return tuple(4 * [float('nan')])

    def _set_scala_model(self):
        from hail.utils.java import Env
        from hail.linalg import _jarray_from_ndarray, _breeze_from_ndarray

        if not self._fitted:
            raise Exception("Ratio of variance parameters is undefined. Run 'fit' first.")

        self._scala_model = Env.hail().stats.LinearMixedModel.apply(
            Env.hc()._jhc,
            self.gamma,
            self._residual_sq,
            _jarray_from_ndarray(self.y),
            _breeze_from_ndarray(self.c),
            _jarray_from_ndarray(self.py),
            _breeze_from_ndarray(self.pc),
            _jarray_from_ndarray(self._z),
            self._ydy,
            _jarray_from_ndarray(self._cdy),
            _breeze_from_ndarray(self._cdc)
        )
