import numpy as np
from hail.typecheck import *
from hail.utils.java import joption, jnone, jsome


class LinearMixedModel(object):
    r"""Class representing a linear mixed model.

    .. include:: ../_templates/experimental.rst

    :class:`LinearMixedModel` represents a linear model of the form

    .. math::

        y \sim \mathrm{N}(C \beta, \, \sigma^2 K + \tau^2 I)

    where

    - :math:`\mathrm{N}` is a :math:`n`-dimensional normal distribution.
    - :math:`y` is a known vector of :math:`n` observations.
    - :math:`C` is a known :math:`n \times k` design matrix for :math:`k` fixed effects.
    - :math:`K` is a known :math:`n \times n` positive semi-definite kernel.
    - :math:`I` is the :math:`n \times n` identity matrix.
    - :math:`\beta` is a :math:`k`-parameter vector of fixed effects.
    - :math:`\sigma^2` is the variance parameter on :math:`K`.
    - :math:`\tau^2` is the variance parameter on :math:`I`.

    In particular, the residuals for the :math:`i^\mathit{th}` and :math:`j^\mathit{th}`
    observations have covariance :math:`\sigma^2 K_{ij}` for :math:`i \neq j`.

    This model is equivalent to a
    `mixed model <https://en.wikipedia.org/wiki/Mixed_model>`__
    of the form

    .. math::

        y = C \beta + Z \nu + \epsilon

    by setting :math:`K = ZZ^T` where

    - :math:`Z` is a known :math:`n \times r` design matrix for :math:`r` random effects.
    - :math:`\nu` is a :math:`r`-vector of random effects drawn from :math:`\mathrm{N}(0, \sigma^2 I)`.
    - :math:`\epsilon` is a :math:`n`-vector of random errors drawn from :math:`\mathrm{N}(0, \tau^2 I)`.

    However, :class:`LinearMixedModel` does not itself realize :math:`K` as a linear kernel
    with respect to random effects, nor does it take :math:`K` explicitly as input. Rather,
    via the eigendecomposion :math:`K = P^T S P`, the the class leverages a third, decorrelated
    form of the model

    .. math::

        Py \sim \mathrm{N}(PC \beta, \, \sigma_g^2 (\gamma S + I))

    where

    - :math:`P: \mathbb{R}^n \rightarrow \mathbb{R}^n` is an orthonormal transformation
      that decorrelates the fixed effects. The rows of :math:`P` are an eigenbasis for :math:`K`.
    - :math:`S` is the :math:`n \times n` diagonal matrix of corresponding eigenvalues.
    - :math:`\gamma = \frac{\sigma^2}{\tau^2}` is the ratio of variance parameters.

    Hence, the triple :math:`(Py, PC, S)` determines the probability
    of the observations for any choice of model parameters, and is
    therefore sufficient for inference.
    This triple, with S encoded as a vector, is the default
    ("full-rank") initialization of the class.

    :class:`LinearMixedModel` also provides an efficient strategy to fit the
    model above with :math:`K` replaced by its rank-:math:`r` approximation
    :math:`K_r = P_r^T S_r P_r` where

    - :math:`P_r: \mathbb{R}^n \rightarrow \mathbb{R}^r` has orthonormal rows
      consisting of the top :math:`r`  eigenvectors of :math:`K`.
    - :math:`S_r` is the :math:`r \times r` diagonal matrix of corresponding
      non-zero eigenvalues.

    For this low-rank model, the quintuple :math:`(P_r y, P_r C, S_r, y, C)`
    is similarly sufficient for inference and corresponds to the "low-rank"
    initialization of the class. Morally, :math:`y` and :math:`C` are
    required for low-rank inference because the diagonal :math:`\gamma S + I`
    is always full-rank.

    If :math:`K` actually has rank :math:`r`, then :math:`K = K_r`
    and the low-rank and full-rank models are equivalent.
    Hence low-rank inference provides a more efficient, equally-exact
    algorithm for fitting the full-rank model.
    This situation arises, for example, when :math:`K` is the linear kernel
    of a mixed model with fewer random effects than observations.

    Even when :math:`K` has full rank, using a lower-rank approximation may
    be an effective from of regularization, in addition to boosting
    computational efficiency.

    For low-rank initialization, all of the following class attributes are 
    set. For full-rank initialization, :math:`n` must equal :math:`r`
    and `y` and `c` are not set.

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
        - Number of random effects
      * - `py`
        - numpy.ndarray
        - Projected response vector :math:`Py` with shape :math:`(r)`
      * - `pc`
        - numpy.ndarray
        - Projected (decorrelated) design matrix :math:`PC` with shape :math:`(r, k)`
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

    Estimation proceeds by minimizing the function :meth:`compute_neg_log_reml`
    with respect to the parameter :math:`\log{\gamma}` governing the (log)
    ratio of the variance parameters :math:`\sigma^2` and :math:`\tau^2`. For
    any fixed ratio, the REML estimate and log likelihood have closed form solutions.

    The model is also equivalent to its augmentation

    .. math::

        y \sim \mathrm{N}\left(x_\star\beta_\star + C \beta, \, \sigma^2 K + \tau^2 I\right)

    by an additional covariate of interest :math:`x_\star` under the
    null hypothesis that the corresponding fixed effect :math:`\beta_\star` is zero.

    After running :meth:`fit` to fit the null model,
    the methods :meth:`fit_alternatives` and :meth:`fit_alternatives_numpy`
    may be used to test the null hypothesis :math:`\beta_\star = 0` versus
    the alternative hypothesis :math:`\beta_\star \neq 0` for
    each :math:`n`-vector :math:`x_\star` in a collection of augmentations.
    Testing uses the likelihood ratio test with both the null and alternative models
    constrained by the REML estimate of :math:`\log{\gamma}` under the null hypothesis.
    The test statistic :math:`\chi^2` equals :math:`n` times the log ratio of
    the squared residuals and follows a chi-squared distribution with one degree of freedom.
    For these methods, the vector :math:`Px` is sufficient for full-rank inference,
    whereas the low-rank inference requires both :math:`P_r x` and :math:`x`.

    Parameters
    ----------
    py: :class:`numpy.ndarray`
        Projected response vector with shape :math:`(r)`.
    pc: :class:`numpy.ndarray`
        Decorrelated design matrix with shape :math:`(r, k)`.
    s: :class:`numpy.ndarray`
        Eigenvalues vector with shape :math:`(r)`.
    y: :class:`numpy.ndarray`, optional
        Response vector with shape :math:`(n)`.
        Required for low-rank inference.
    c: :class:`numpy.ndarray`, optional
        Design matrix with shape :math:`(n, k)`.
        Required for low-rank inference.
    """
    @typecheck_method(y=np.ndarray, c=np.ndarray, py=np.ndarray, pc=np.ndarray, s=np.ndarray)
    def __init__(self, py, pc, s, y=None, c=None):
        if y is None and c is None:
            low_rank = False
        elif y is not None and c is not None:
            low_rank = True
        else:
            raise ValueError('Set y and c for low-rank')  # FIXME

        assert py.ndim == 1 and pc.ndim == 2 and s.ndim == 1
        r, k = pc.shape
        assert r > k >= 0
        assert py.size == r
        assert s.size == r

        if low_rank:
            assert y.ndim == 1 and c.ndim == 2
            assert c.shape == (y.size, k)
            assert y.size > r

        self.r = r
        self.k = k
        self.n = y.size if low_rank else r
        self.dof = self.n - k
        self.py = py
        self.pc = pc
        self.s = s
        self.y = y
        self.c = c

        self.beta = None
        self.sigma_sq = None
        self.tau_sq = None
        self.gamma = None
        self.log_gamma = None
        self.h_sq = None
        self.optimize_result = None

        self._fitted = False
        self._low_rank = low_rank 
        if low_rank:
            self._yty = y @ y
            self._cty = c.T @ y
            self._ctc = c.T @ c
        self._d = None
        self._ydy = None
        self._cdy = None
        self._cdc = None
        self._d_alt = None
        self._ydy_alt = None
        self._cdy_alt = np.zeros(self.k + 1)
        self._cdc_alt = np.zeros((self.k + 1, self.k + 1))
        self._residual_sq = None
        self._scala_model = None

    def _reset(self):
        self._fitted = False

        self.beta = None
        self.sigma_sq = None
        self.tau_sq = None
        self.gamma = None
        self.log_gamma = None
        self.h_sq = None
        self.optimize_result = None

    def compute_neg_log_reml(self, log_gamma, return_parameters=False):
        r"""Compute negative log REML constrained to a fixed value
        of :math:`\log{\gamma}`.

        This function computes the triple :math:`(\beta, \sigma^2, \tau^2)` with
        :math:`\gamma = \frac{\sigma^2}{\tau^2}` at which the restricted likelihood
        is maximized and returns the negative of the log of the
        restricted likelihood at these parameters, shifted by a constant whose value is
        independent of the input.

        To compute the actual negative log REML, add

        .. math::

            \frac{1}{2}\left((n - k)(1 + \log(2\pi)) - \log(\det(C^T C)\right)

        to the returned value.

        Parameters
        ----------
        log_gamma: :obj:`float`
            Value of :math:`\log{\gamma}`.
        return_parameters:
            If ``True``, also return :math:`\beta`, :math:`\sigma^2`,
            and :math:`\tau^2`.

        Returns
        -------
        :obj:`float` or (:obj:`float`, :class:`numpy.ndarray`, :obj:`float`, :obj:`float`)
            If `return_parameters` is ``False``, returns (shifted) negative log REML.
            Otherwise, returns (shifted) negative log REML, :math:`\beta`, :math:`\sigma^2`,
            and :math:`\tau^2`.
        """
        from scipy.linalg import solve, LinAlgError

        gamma = np.exp(log_gamma)
        d = 1 / (self.s + 1 / gamma)
        logdet_d = np.sum(np.log(d)) + (self.n - self.r) * log_gamma

        if self._low_rank:
            d -= gamma
            dpy = d * self.py
            ydy = self.py @ dpy + gamma * self._yty
            cdy = self.pc.T @ dpy + gamma * self._cty
            cdc = (self.pc.T * d) @ self.pc + gamma * self._ctc
        else:
            dpy = d * self.py
            ydy = self.py @ dpy
            cdy = self.pc.T @ dpy
            cdc = (self.pc.T * d) @ self.pc

        try:
            beta = solve(cdc, cdy, assume_a='pos', overwrite_a=True)
            residual_sq = ydy - cdy.T @ beta
            sigma_sq = residual_sq / self.dof
            tau_sq = sigma_sq / gamma
            neg_log_reml = (np.linalg.slogdet(cdc)[1] - logdet_d + self.dof * np.log(sigma_sq)) / 2

            if return_parameters:
                return neg_log_reml, beta, sigma_sq, tau_sq
            else:
                self._d, self._ydy, self._cdy, self._cdc = d, ydy, cdy, cdc
                return neg_log_reml
        except LinAlgError as e:
            raise Exception(f'Linear algebra error while solving for REML estimate:\n  {e}')

    @typecheck_method(log_gamma=nullable(float), bounds=tupleof(numeric), tol=float, maxiter=int)
    def fit(self, log_gamma=None, bounds=(-8.0, 8.0), tol=1e-8, maxiter=500):
        r"""Find the triple :math:`(\beta, \sigma^2, \tau^2)` maximizing REML.

        This method sets the attributes `beta`, `sigma_sq`, `tau_sq`, `gamma`,
        `log_gamma`, and `h_sq` as described in the top-level class documentation.

        If `log_gamma` is provided, :meth:`fit` finds the REML solution
        with :math:`\log{\gamma}` constrained to this value.
        Otherwise, :meth:`fit` searches for the value of :math:`\log{\gamma}`
        that minimizes :meth:`compute_neg_log_reml`, and also sets the attribute
        `optimize_result` of type `scipy.optimize.OptimizeResult
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`__.

        Parameters
        ----------
        log_gamma: :obj:`float`, optional
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
                if self.optimize_result.x - bounds[0] < 0.01:
                    raise Exception("Failed to fit log_gamma: optimum within 0.01 of lower bound.")
                elif bounds[1] - self.optimize_result.x < tol:
                    raise Exception("Failed to fit log_gamma: optimum within 0.01 of upper bound.")
                else:
                    self.log_gamma = self.optimize_result.x
            else:
                raise Exception(f'Failed to fit log_gamma:\n  {self.optimize_result}')

        _, self.beta, self.sigma_sq, self.tau_sq = self.compute_neg_log_reml(self.log_gamma, return_parameters=True)

        self.gamma = np.exp(self.log_gamma)
        self.h_sq = self.sigma_sq / (self.sigma_sq + self.tau_sq)
        self._residual_sq = self.sigma_sq * self.dof
        self._d_alt = self._d
        self._ydy_alt = self._ydy
        self._cdy_alt[1:] = self._cdy
        self._cdc_alt[1:, 1:] = self._cdc

        self._fitted = True

    @typecheck_method(path_pxt=str, path_xt=nullable(str), partition_size=int)
    def fit_alternatives(self, path_pxt, path_xt=None, partition_size=1024):
        r"""Fit and test alternative model for each augmented design matrix in parallel.

        The resulting Table has the following fields:

        .. list-table::
          :header-rows: 1

          * - Field
            - Type
            - Value
          * - `idx`
            - int64
            - Index of augmented design matrix.
          * - `beta`
            - float64
            - :math:`\beta_\star`
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
        path_pxt: :obj:`str`
            Path to the transpose of the projected block matrix :math:`PX`.
            The transposed matrix has shape :math:`(m, r)`.
            Each row is a projected augmentation :math:`P x_\star`.
        path_xt: :obj:`str`, optional
            Path to the transpose of block matrix :math:`X`.
            The transposed matrix has shape :math:`(m , n)`.
            Each row is an augmentation :math:`x_\star`.
            Required for low-rank inference.
        partition_size: :obj:`int`
            Number of rows per partition.

        Returns
        -------
        :class:`.Table`
            Table of results for each augmented design matrix.
        """
        from hail.table import Table

        if partition_size <= 0:
            raise ValueError("partition_size must be positive")

        if not self._scala_model:
            self._set_scala_model()

        return Table(self._scala_model.fit(path_pxt, joption(path_xt), partition_size))

    @typecheck_method(px=np.ndarray, x=nullable(np.ndarray))
    def fit_alternatives_numpy(self, px, x=None):
        r"""Fit and test alternative model for each augmented design matrix.

        The resulting pandas DataFrame has the following fields.

        .. list-table::
          :header-rows: 1

          * - Field
            - Type
            - Value
          * - `beta`
            - float
            - :math:`\beta_\star`
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
        px: :class:`numpy.ndarray`
            Matrix :math:`PX` with shape :math:`(r, m)`.
            Each column is a projected augmentation :math:`P x_\star`.
        x: :class:`numpy.ndarray`, optional
            Matrix :math:`X` with shape :math:`(n, m)`.
            Each column is an augmentation :math:`x_\star`.
            Required for low-rank inference.

        Returns
        -------
        :class:`pandas.DataFrame`
            Data frame of results for each augmented design matrix.
        """
        import pandas as pd

        if not self._fitted:
            raise Exception("Null model is not fit. Run 'fit' first.")

        n_cols = px.shape[1]
        assert px.shape[0] == self.r

        if self._low_rank:
            assert x.shape[0] == self.n and x.shape[1] == n_cols
            data = [self._fit_alternative_numpy(px[:, i], x[:, i]) for i in range(n_cols)]
        else:
            data = [self._fit_alternative_numpy(px[:, i], None) for i in range(n_cols)]

        return pd.DataFrame.from_records(data, columns=['beta', 'sigma_sq', 'chi_sq', 'p_value'])

    def _fit_alternative_numpy(self, px, x):
        from scipy.linalg import solve, LinAlgError
        from scipy.stats.distributions import chi2

        gamma = self.gamma
        dpx = self._d_alt * px

        # single thread => no need to copy
        ydy = self._ydy_alt
        cdy = self._cdy_alt
        cdc = self._cdc_alt

        if self._low_rank:
            cdy[0] = self.py @ dpx + gamma * (self.y @ x)
            cdc[0, 0] = px @ dpx + gamma * (x @ x)
            cdc[0, 1:] = self.pc.T @ dpx + gamma * (self.c.T @ x)
        else:
            cdy[0] = self.py @ dpx
            cdc[0, 0] = px @ dpx
            cdc[0, 1:] = self.pc.T @ dpx

        try:
            beta = solve(cdc, cdy, assume_a='pos', overwrite_a=True)  # only uses upper triangle
            residual_sq = ydy - cdy.T @ beta
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
            raise Exception("Null model is not fit. Run 'fit' first.")

        self._scala_model = Env.hail().stats.LinearMixedModel.apply(
            Env.hc()._jhc,
            self.gamma,
            self._residual_sq,
            _jarray_from_ndarray(self.py),
            _breeze_from_ndarray(self.pc),
            _jarray_from_ndarray(self._d_alt),
            self._ydy_alt,
            _jarray_from_ndarray(self._cdy_alt),
            _breeze_from_ndarray(self._cdc_alt),
            jsome(_jarray_from_ndarray(self.y)) if self._low_rank else jnone,
            jsome(_breeze_from_ndarray(self.c)) if self._low_rank else jnone
        )
