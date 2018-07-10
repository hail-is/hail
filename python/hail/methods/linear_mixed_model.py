import numpy as np
import pandas as pd
from hail.utils.misc import plural
from hail.typecheck import *
from hail.utils.java import Env, jnone, jsome
from hail.table import Table


class LinearMixedModel(object):
    r"""Class representing a linear mixed model.

    .. include:: ../_templates/experimental.rst

    :class:`LinearMixedModel` represents a linear model of the form

    .. math::

        y \sim \mathrm{N}(X \beta, \, \sigma^2 K + \tau^2 I)

    where

    - :math:`\mathrm{N}` is a :math:`n`-dimensional normal distribution.
    - :math:`y` is a known vector of :math:`n` observations.
    - :math:`X` is a known :math:`n \times p` design matrix for :math:`p` fixed effects.
    - :math:`K` is a known :math:`n \times n` positive semi-definite kernel.
    - :math:`I` is the :math:`n \times n` identity matrix.
    - :math:`\beta` is a :math:`p`-parameter vector of fixed effects.
    - :math:`\sigma^2` is the variance parameter on :math:`K`.
    - :math:`\tau^2` is the variance parameter on :math:`I`.

    In particular, the residuals for the :math:`i^\mathit{th}` and :math:`j^\mathit{th}`
    observations have covariance :math:`\sigma^2 K_{ij}` for :math:`i \neq j`.

    This model is equivalent to a
    `mixed model <https://en.wikipedia.org/wiki/Mixed_model>`__
    of the form

    .. math::

        y = X \beta + Z u + \epsilon

    by setting :math:`K = ZZ^T` where

    - :math:`Z` is a known :math:`n \times r` design matrix for :math:`r` random effects.
    - :math:`u` is a :math:`r`-vector of random effects drawn from :math:`\mathrm{N}(0, \sigma^2 I)`.
    - :math:`\epsilon` is a :math:`n`-vector of random errors drawn from :math:`\mathrm{N}(0, \tau^2 I)`.

    However, :class:`LinearMixedModel` does not itself realize :math:`K` as a linear kernel
    with respect to random effects, nor does it take :math:`K` explicitly as input. Rather,
    via the eigendecomposion :math:`K = U S U^T`, the the class leverages a third, decorrelated
    form of the model

    .. math::

        Py \sim \mathrm{N}(PX \beta, \, \sigma^2 (\gamma S + I))

    where

    - :math:`P = U^T: \mathbb{R}^n \rightarrow \mathbb{R}^n` is an orthonormal transformation
      that decorrelates the observations. The rows of :math:`P` are an eigenbasis for :math:`K`.
    - :math:`S` is the :math:`n \times n` diagonal matrix of corresponding eigenvalues.
    - :math:`\gamma = \frac{\sigma^2}{\tau^2}` is the ratio of variance parameters.

    Hence, the triple :math:`(Py, PX, S)` determines the probability
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

    For this low-rank model, the quintuple :math:`(P_r y, P_r X, S_r, y, X)`
    is similarly sufficient for inference and corresponds to the "low-rank"
    initialization of the class. Morally, :math:`y` and :math:`X` are
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

    **Initialization**

    With full-rank initialization by :math:`(Py, PX, S)`, the following class attributes are set:

    .. list-table::
      :header-rows: 1

      * - Attribute
        - Type
        - Value
      * - `low_rank`
        - bool
        - ``False``
      * - `n`
        - int
        - Number of observations :math:`n`
      * - `f`
        - int
        - Number of fixed effects :math:`p`
      * - `r`
        - int
        - Effective number of random effects, must equal :math:`n`
      * - `py`
        - numpy.ndarray
        - Rotated response vector :math:`P y` with shape :math:`(n)`
      * - `px`
        - numpy.ndarray
        - Rotated design matrix :math:`P X` with shape :math:`(n, p)`.
      * - `s`
        - numpy.ndarray
        - Eigenvalues vector :math:`S` of :math:`K` with shape :math:`(n)`

    With low-rank initialization by :math:`(P_r y, P_r X, S_r, y, X)`, the following class attributes are set:

    .. list-table::
      :header-rows: 1

      * - Attribute
        - Type
        - Value
      * - `low_rank`
        - bool
        - ``True``
      * - `n`
        - int
        - Number of observations :math:`n`
      * - `f`
        - int
        - Number of fixed effects :math:`p`
      * - `r`
        - int
        - Effective number of random effects, must be less than :math:`n`
      * - `py`
        - numpy.ndarray
        - Projected response vector :math:`P_r y` with shape :math:`(r)`
      * - `px`
        - numpy.ndarray
        - Projected design matrix :math:`P_r X` with shape :math:`(r, p)`
      * - `s`
        - numpy.ndarray
        - Eigenvalues vector :math:`S_r` of :math:`K_r` with shape :math:`(r)`
      * - `y`
        - numpy.ndarray
        - Response vector with shape :math:`(n)`
      * - `x`
        - numpy.ndarray
        - Design matrix with shape :math:`(n, p)`

    **Fitting the model**

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
    any fixed ratio, the REML estimate and log likelihood have closed-form solutions.

    **Testing alternative models**

    The model is also equivalent to its augmentation

    .. math::

        y \sim \mathrm{N}\left(x_\star\beta_\star + X \beta, \, \sigma^2 K + \tau^2 I\right)

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

    When testing alternatives, full-rank inference only requires the vector :math:`P x_\star`,
    whereas low-rank inference requires both :math:`P_r x_\star` and :math:`x_\star`.

    Parameters
    ----------
    py: :class:`numpy.ndarray`
        Projected response vector :math:`P_r y` with shape :math:`(r)`.
    px: :class:`numpy.ndarray`
        Projected design matrix :math:`P_r X` with shape :math:`(r, p)`.
    s: :class:`numpy.ndarray`
        Eigenvalues vector :math:`S` with shape :math:`(r)`.
    y: :class:`numpy.ndarray`, optional
        Response vector with shape :math:`(n)`.
        Include for low-rank inference.
    x: :class:`numpy.ndarray`, optional
        Design matrix with shape :math:`(n, p)`.
        Include for low-rank inference.
    """
    @typecheck_method(py=np.ndarray,
                      px=np.ndarray,
                      s=np.ndarray,
                      y=nullable(np.ndarray),
                      x=nullable(np.ndarray))
    def __init__(self, py, px, s, y=None, x=None):
        if y is None and x is None:
            low_rank = False
        elif y is not None and x is not None:
            low_rank = True
        else:
            raise ValueError('for low-rank, set both y and x; for full-rank, do not set y or x.')

        assert py.ndim == 1
        assert px.ndim == 2
        assert s.ndim == 1

        r, f = px.shape
        if f == 0:
            raise ValueError('LinearMixedModel must have at least one fixed effect.')  # could relax

        assert py.size == r
        assert s.size == r

        if low_rank:
            assert y.ndim == 1
            assert x.ndim == 2
            assert x.shape == (y.size, f)
            assert y.size > r
            n = y.size
        else:
            n = r

        self.low_rank = low_rank
        self.n = n
        self.f = f
        self.r = r
        self.py = py
        self.px = px
        self.s = s
        self.y = y
        self.x = x

        self._check_dof()

        self.beta = None
        self.sigma_sq = None
        self.tau_sq = None
        self.gamma = None
        self.log_gamma = None
        self.h_sq = None
        self.optimize_result = None

        self._fitted = False

        if low_rank:
            self._yty = y @ y
            self._xty = x.T @ y
            self._xtx = x.T @ x

        self._dof = n - f
        self._d = None
        self._ydy = None
        self._xdy = None
        self._xdx = None

        self._dof_alt = n - (f + 1)
        self._d_alt = None
        self._ydy_alt = None
        self._xdy_alt = np.zeros(f + 1)
        self._xdx_alt = np.zeros((f + 1, f + 1))

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

            \frac{1}{2}\left((n - p)(1 + \log(2\pi)) - \log(\det(X^T X)\right)

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

        if self.low_rank:
            d -= gamma
            dpy = d * self.py
            ydy = self.py @ dpy + gamma * self._yty
            xdy = self.px.T @ dpy + gamma * self._xty
            xdx = (self.px.T * d) @ self.px + gamma * self._xtx
        else:
            dpy = d * self.py
            ydy = self.py @ dpy
            xdy = self.px.T @ dpy
            xdx = (self.px.T * d) @ self.px

        try:
            beta = solve(xdx, xdy, assume_a='pos')
            residual_sq = ydy - xdy.T @ beta
            sigma_sq = residual_sq / self._dof
            tau_sq = sigma_sq / gamma
            neg_log_reml = (np.linalg.slogdet(xdx)[1] - logdet_d + self._dof * np.log(sigma_sq)) / 2

            if return_parameters:
                return neg_log_reml, beta, sigma_sq, tau_sq
            else:
                self._d, self._ydy, self._xdy, self._xdx = d, ydy, xdy, xdx
                return neg_log_reml
        except LinAlgError as e:
            raise Exception(f'linear algebra error while solving for REML estimate') from e

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
                    raise Exception("failed to fit log_gamma: optimum within 0.01 of lower bound.")
                elif bounds[1] - self.optimize_result.x < tol:
                    raise Exception("failed to fit log_gamma: optimum within 0.01 of upper bound.")
                else:
                    self.log_gamma = self.optimize_result.x
            else:
                raise Exception(f'failed to fit log_gamma:\n  {self.optimize_result}')

        _, self.beta, self.sigma_sq, self.tau_sq = self.compute_neg_log_reml(self.log_gamma, return_parameters=True)

        self.gamma = np.exp(self.log_gamma)
        self.h_sq = self.sigma_sq / (self.sigma_sq + self.tau_sq)

        self._residual_sq = self.sigma_sq * self._dof
        self._d_alt = self._d
        self._ydy_alt = self._ydy
        self._xdy_alt[1:] = self._xdy
        self._xdx_alt[1:, 1:] = self._xdx

        self._fitted = True

    @typecheck_method(pa_t_path=str,
                      a_t_path=nullable(str),
                      partition_size=nullable(int))
    def fit_alternatives(self, pa_t_path, a_t_path=None, partition_size=None):
        r"""Fit and test alternative model for each augmented design matrix in parallel.

        Notes
        -----
        The resulting table has the following fields:

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

        :math:`(P_r A)^T` and :math:`A^T` (if given) must have the same number
        of rows (augmentations). These rows are grouped into partitions for
        parallel processing. The number of partitions equals the ceiling of
        ``n_rows / partition_size``, and should be at least the number or cores
        to make use of all cores. By default, there is one partition per row of
        blocks in :math:`(P_r A)^T`. Setting the partition size to an exact
        (rather than approximate) divisor or multiple of the block size reduces
        superfluous shuffling of data.

        The number of columns in each block matrix must be less than :math:`2^{31}`.

        Warning
        -------
        The block matrices must be stored in row-major format, as results
        from :meth:`.BlockMatrix.write` with ``force_row_major=True`` and from
        :meth:`.BlockMatrix.write_from_entry_expr`. Otherwise, this method
        will produce an error message.

        Parameters
        ----------
        pa_t_path: :obj:`str`
            Path to block matrix :math:`(P_r A)^T` with shape :math:`(m, r)`.
            Each row is a projected augmentation :math:`P_r x_\star` of :math:`P_r X`.
        a_t_path: :obj:`str`, optional
            Path to block matrix :math:`A^T` with shape :math:`(m, n)`.
            Each row is an augmentation :math:`x_\star` of :math:`X`.
            Include for low-rank inference.
        partition_size: :obj:`int`, optional
            Number of rows to process per partition.
            Default given by block size of :math:`(P_r A)^T`.

        Returns
        -------
        :class:`.Table`
            Table of results for each augmented design matrix.
        """
        from hail.table import Table

        self._check_dof(self.f + 1)

        if self.low_rank and a_t_path is None:
            raise ValueError('model is low-rank so a_t is required.')
        elif not (self.low_rank or a_t_path is None):
            raise ValueError('model is full-rank so a_t must not be set.')

        if self._scala_model is None:
            self._set_scala_model()

        if partition_size is None:
            block_size = Env.hail().linalg.BlockMatrix.readMetadata(Env.hc()._jhc, pa_t_path).blockSize()
            partition_size = block_size
        elif partition_size <= 0:
            raise ValueError(f'partition_size must be positive, found {partition_size}')

        jpa_t = Env.hail().linalg.RowMatrix.readBlockMatrix(Env.hc()._jhc, pa_t_path, jsome(partition_size))

        if a_t_path is None:
            maybe_ja_t = jnone()
        else:
            maybe_ja_t = jsome(
                Env.hail().linalg.RowMatrix.readBlockMatrix(Env.hc()._jhc, a_t_path, jsome(partition_size)))

        return Table(self._scala_model.fit(jpa_t, maybe_ja_t))

    @typecheck_method(pa=np.ndarray, a=nullable(np.ndarray))
    def fit_alternatives_numpy(self, pa, a=None):
        r"""Fit and test alternative model for each augmented design matrix.

        Notes
        -----
        The resulting table has the following fields:

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
        pa: :class:`numpy.ndarray`
            Projected matrix :math:`P_r A` of alternatives with shape :math:`(r, m)`.
            Each column is a projected augmentation :math:`P_r x_\star` of :math:`P_r X`.
        a: :class:`numpy.ndarray`, optional
            Matrix :math:`A` of alternatives with shape :math:`(n, m)`.
            Each column is an augmentation :math:`x_\star` of :math:`X`.
            Required for low-rank inference.

        Returns
        -------
        :class:`.Table`
            Table of results for each augmented design matrix.
        """
        self._check_dof(self.f + 1)

        if not self._fitted:
            raise Exception("null model is not fit. Run 'fit' first.")

        n_cols = pa.shape[1]
        assert pa.shape[0] == self.r

        if self.low_rank:
            assert a.shape[0] == self.n and a.shape[1] == n_cols
            data = [(i,) + self._fit_alternative_numpy(pa[:, i], a[:, i]) for i in range(n_cols)]
        else:
            data = [(i,) + self._fit_alternative_numpy(pa[:, i], None) for i in range(n_cols)]

        df = pd.DataFrame.from_records(data, columns=['idx', 'beta', 'sigma_sq', 'chi_sq', 'p_value'])

        return Table.from_pandas(df, key='idx')

    def _fit_alternative_numpy(self, pa, a):
        from scipy.linalg import solve, LinAlgError
        from scipy.stats.distributions import chi2

        gamma = self.gamma
        dpa = self._d_alt * pa

        # single thread => no need to copy
        ydy = self._ydy_alt
        xdy = self._xdy_alt
        xdx = self._xdx_alt

        if self.low_rank:
            xdy[0] = self.py @ dpa + gamma * (self.y @ a)
            xdx[0, 0] = pa @ dpa + gamma * (a @ a)
            xdx[0, 1:] = self.px.T @ dpa + gamma * (self.x.T @ a)
        else:
            xdy[0] = self.py @ dpa
            xdx[0, 0] = pa @ dpa
            xdx[0, 1:] = self.px.T @ dpa

        try:
            beta = solve(xdx, xdy, assume_a='pos')  # only uses upper triangle
            residual_sq = ydy - xdy.T @ beta
            sigma_sq = residual_sq / self._dof_alt
            chi_sq = self.n * np.log(self._residual_sq / residual_sq)  # division => precision
            p_value = chi2.sf(chi_sq, 1)

            return beta[0], sigma_sq, chi_sq, p_value
        except LinAlgError:
            return tuple(4 * [float('nan')])

    def _set_scala_model(self):
        from hail.utils.java import Env
        from hail.linalg import _jarray_from_ndarray, _breeze_from_ndarray

        if not self._fitted:
            raise Exception("null model is not fit. Run 'fit' first.")

        self._scala_model = Env.hail().stats.LinearMixedModel.apply(
            Env.hc()._jhc,
            self.gamma,
            self._residual_sq,
            _jarray_from_ndarray(self.py),
            _breeze_from_ndarray(self.px),
            _jarray_from_ndarray(self._d_alt),
            self._ydy_alt,
            _jarray_from_ndarray(self._xdy_alt),
            _breeze_from_ndarray(self._xdx_alt),
            jsome(_jarray_from_ndarray(self.y)) if self.low_rank else jnone(),
            jsome(_breeze_from_ndarray(self.x)) if self.low_rank else jnone()
        )

    def _check_dof(self, f=None):
        if f is None:
            f = self.f
        dof = self.n - f
        if dof <= 0:
            raise ValueError(f"{self.n} {plural('observation', self.n)} with {f} fixed {plural('effect', f)}"
                             f"implies {dof} {plural('degree', dof)} of freedom. Must be positive.")
