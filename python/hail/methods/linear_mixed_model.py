import numpy as np
from hail.typecheck import *


class LinearMixedModel(object):
    """
    .. include:: _templates/experimental.rst

    .. math::

      y \sim \mathrm{N}\left(X\beta, \sigma^2 K + \tau^2 I\right)

    The following attributes are set at initialization.
    - `n` (:obj:`int`) -- Number of observations
    - `k` (:obj:`int`) -- Number of fixed effects
    - `r` (:obj:`int`) -- Number of random effects
    - `y` (:class:`numpy.ndarray`) -- Response vector with shape (math:`n`,)
    - `c` (:class:`numpy.ndarray`) -- Covariate matrix of fixed effects with shape (math:`n`, math:`k`)
    - `py` (:class:`numpy.ndarray`) -- Projected response vector math:`Py` with shape (math:`r`,)
    - `pc` (:class:`numpy.ndarray`) -- Projected covariate matrix math:`PC` with shape (math:`r`, math:`k`)
    - `s` (:class: numpy.ndarray`) -- Eigenvalues vector of projection with shape (:math:`r`,)

    The following attributes are set by :meth:`fit` to values which jointly maximize the
    [restricted maximum likelihood](https://en.wikipedia.org/wiki/Restricted_maximum_likelihood) (REML).
    - `beta` (:class:`numpy.ndarray`) -- math:`\beta`
    - `sigma_sq` (:obj:`float`) -- math:`\sigma^2`
    - `tau_sq` (:obj:`float`) -- math:`\tau^2`

    :meth:`fit` also computes the following values at this estimate:
    - `gamma` (:obj:`float`) -- math:`\gamma = \frac{\sigma^2}{\tau^2}`
    - `log_gamma` (:obj:`float`) -- math:`\log(\gamma)`
    - `h_sq` (:obj:`float`) -- math:`h^2 = \frac{\sigma^2}{\sigma^2 + \tau^2}`
    - `log_reml` (:obj:`float`) -- log of the restricted maximum likelihood
    - `optimize_result` (:class:`scipy.optimize.OptimizeResult`) -- class returned by `scipy.optimize.minimize_scalar < https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html>`__

    Parameters
    ----------
    y: :class:`numpy.ndarray`
        Response vector.
    c: :class:`numpy.ndarray`
        Covariate matrix
    py: :class:`numpy.ndarray`
        Projected response vector.
    pc: :class:`numpy.ndarray`
        Projected covariate matrix.
    s: :class:`numpy.ndarray`
        Eigenvalues vector.
    """

    @typecheck_method(y=np.ndarray, c=np.ndarray, py=np.ndarray, pc=np.ndarray, s=np.ndarray)
    def __init__(self, y, c, py, pc, s):
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

        # set by fit
        self._fitted = False
        self.beta = None
        self.gamma = None
        self.h_sq = None
        self.log_gamma = None
        self.log_reml = None
        self.optimize_result = None
        self.sigma_sq = None
        self.tau_sq = None

        # constants for fit_rows / fit_local_rows
        self._expanded = False
        self._z = None
        self._ydy = None
        self._cdy = None
        self._cdc = None
        self._residual_sq = None

        # scala class used by fit_rows
        self._scala_model = None

    @typecheck_method(log_gamma=nullable(float), lower=float, upper=float, tol=float, maxiter=int)
    def fit(self, log_gamma=None, lower=-8.0, upper=8.0, tol=1e-8, maxiter=500):
        r"""Find the triple :math:`(\beta, \sigma^2, \tau^2)`` maximizing REML.

        This method sets the value of the class attributes `beta`, `gamma`,
        `log_gamma`, `h_sq`, `log_reml`, `sigma_sq`, and `tau_sq` described
        above. If `log_gamma` is not set, the attribute `optimize_result` is set
        as well.

        Parameters
        ----------
        log_gamma: :obj:`float`, optional.
            If provided, :math:`\log(\frac{\sigma^2}{\tau^2}` is constrained to
            this value rather than fit.
        lower: :obj:`float`
            Lower bound for :math:`\log(\gamma)`.
        upper: :obj:`float`
            Upper bound for :math:`\log(\gamma)`.
        tol: :obj:`float`
            Absolute tolerance for optimizing :math:`\log(\gamma)`.
        maxiter: :obj:`float`
            Maximum number of iterations for optimizing :math:`\log(\gamma)`.
        """
        if self._fitted:
            self._reset()

        if log_gamma:
            self._neg_log_reml(log_gamma)
            self.log_gamma = log_gamma
        else:
            from scipy.optimize import minimize_scalar
            self.optimize_result = minimize_scalar(self._neg_log_reml, method='bounded', bounds=(lower, upper),
                                         options={'xatol': tol, 'maxiter': maxiter})
            if not self.optimize_result.success:
                raise Exception('Failed to fit log_gamma:' + self.optimize_result.message)
            self.log_gamma = self.optimize_result.x

        self.gamma = np.exp(self.log_gamma)
        self.tau_sq = self.sigma_sq / self.gamma
        self.h_sq = self.sigma_sq / (self.sigma_sq + self.tau_sq)
        self._fitted = True

    @typecheck_method(path_xt=str, path_pxt=str, partition_size=int)
    def fit_rows(self, path_xt, path_pxt, partition_size):
        from hail.table import Table

        assert partition_size > 0

        if not self._scala_model:
            self._set_scala_model()

        return Table(self._scala_model.fit(path_xt, path_pxt, partition_size))

    @typecheck_method(x=np.ndarray, px=np.ndarray)
    def fit_local_rows(self, x, px):
        import pandas as pd

        n_rows = x.shape[1]
        assert px.shape[1] == n_rows
        assert px.shape[0] == self.r

        if not self._expanded:
            self._expand()

        data = [self._fit_local_row(x[:, i], px[:, i]) for i in range(n_rows)]  # could parallelize on master

        return pd.DataFrame.from_records(data, columns=['beta', 'sigma_sq', 'chi_sq', 'p_value'])

    def compute_log_reml(self, log_gamma):
        from scipy.linalg import solve

        gamma = np.exp(log_gamma)
        d = self.s + 1 / gamma
        z = 1 / d - gamma
        zpy = z * self.py

        ydy = self._yty * gamma + self.py @ zpy
        cdy = self._cty * gamma + self.pc.T @ zpy
        cdc = self._ctc * gamma + (self.pc.T * z) @ self.pc

        beta = solve(cdc, cdy, assume_a='pos', overwrite_a=True)
        residual_sq = ydy - cdy.T @ beta
        sigma_sq = residual_sq / self.dof
        tau_sq = sigma_sq / gamma

        logdet_d = np.sum(np.log(d)) + (self.r - self.n) * log_gamma
        log_reml = -0.5 * (logdet_d + np.linalg.slogdet(cdc)[1] + self.dof * np.log(sigma_sq) + self._shift)

        return log_reml, beta, sigma_sq, tau_sq

    def _neg_log_reml(self, log_gamma):
        self.log_reml, self.beta, self.sigma_sq, _ = self.compute_log_reml(log_gamma)
        return -self.log_reml

    def _expand(self):
        assert self.dof > 1

        if not self._fitted:
            raise Exception("First fit with 'fit'.")

        gamma = self.gamma
        new_k = self.k + 1

        z = 1 / (self.s + 1 / self.gamma) - gamma
        zpy = z * self.py
        ydy = self._yty * gamma + self.py @ zpy

        cdy = np.zeros(new_k)
        cdy[1:] = self._cty * gamma + self.pc.T @ zpy

        cdc = np.zeros((new_k, new_k))
        cdc[1:, 1:] = self._ctc * gamma + (self.pc.T * z) @ self.pc

        self._z = z
        self._ydy = ydy
        self._cdy = cdy
        self._cdc = cdc
        self._residual_sq = self.sigma_sq * self.dof
        self._expanded = True

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

        if not self._expanded:
            self._expand()

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

    def _reset(self):
        self._fitted = False
        self.beta = None
        self.gamma = None
        self.h_sq = None
        self.log_gamma = None
        self.log_reml = None
        self.optimize_result = None
        self.sigma_sq = None
        self.tau_sq = None

        self._expanded = False
        self._z = None
        self._ydy = None
        self._cdy = None
        self._cdc = None
        self._residual_sq = None

        self._scala_model = None
