import numpy as np
from hail.typecheck import *


class LinearMixedModel(object):
    """
    Experimental linear mixed model class.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        Response vector with shape (n)
    c : :class:`numpy.ndarray`
        Covariate matrix with shape (n, k), including intercept
    py : :class:`numpy.ndarray`
        Projected response vector P @ y with shape (r)
    pc : :class:`numpy.ndarray`
        Projected covariate matrix P @ c with shape (r, k)
    s : :class:`numpy.ndarray`
        Eigenvalues vector with shape (r)
    """

    @typecheck_method(y=np.ndarray, c=np.ndarray, py=np.ndarray, pc=np.ndarray, s=np.ndarray)
    def __init__(self, y, c, py, pc, s):
        assert y.ndim == 1 and c.ndim == 2 and py.ndim == 1 and pc.ndim == 2 and s.ndim == 1

        self.n, self.k = c.shape
        self.dof = self.n - self.k
        self.rank = s.shape[0]
        self.corank = self.n - self.rank

        assert self.dof > 0 and self.n > self.rank > 0
        assert y.size == self.n
        assert py.size == self.rank
        assert pc.shape == (self.rank, self.k)

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
        self.fitted = False
        self.delta = None
        self.beta = None
        self.sigma_sq = None
        self.reml = None
        self.optim = None

        # constants for fit_rows / fit_local_rows
        self._expanded = False
        self._z = None
        self._ydy = None
        self._cdy = None
        self._cdc = None
        self._residual_sq = None

        # scala class used by fit_rows
        self._scala_model = None

    @typecheck_method(delta=nullable(float), lower=float, upper=float, tol=float, maxiter=int)
    def fit(self, delta=None, lower=-8.0, upper=8.0, tol=1e-8, maxiter=500):
        if self.fitted:
            self._reset()

        if delta:
            self._neg_reml(np.log(delta))
            self.delta = delta
        else:
            from scipy.optimize import minimize_scalar
            self.optim = minimize_scalar(self._neg_reml, method='bounded', bounds=(lower, upper),
                                         options={'xatol': tol, 'maxiter': maxiter})
            if not self.optim.success:
                raise Exception('Failed to fit delta:' + self.optim.message)
            self.delta = np.exp(self.optim.x)

        self.fitted = True
        return self

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
        assert px.shape[0] == self.rank

        if not self._expanded:
            self._expand()

        data = [self._fit_local_row(x[:, i], px[:, i]) for i in range(n_rows)]  # could parallelize on master

        return pd.DataFrame.from_records(data, columns=['beta', 'sigma_sq', 'chi_sq', 'p_value'])

    @typecheck_method(log_delta=float)
    def _neg_reml(self, log_delta):
        from scipy.linalg import solve

        delta = np.exp(log_delta)
        d = self.s + delta

        z = 1 / d - 1 / delta
        zpy = z * self.py

        ydy = self._yty / delta + self.py @ zpy
        cdy = self._cty / delta + self.pc.T @ zpy
        cdc = self._ctc / delta + (self.pc.T * z) @ self.pc

        beta = solve(cdc, cdy, assume_a='pos', overwrite_a=True)
        residual_sq = ydy - cdy.T @ beta

        logdet_d = np.sum(np.log(d)) + self.corank * log_delta
        logdet_cdc = np.linalg.slogdet(cdc)[1]

        self.beta = np.squeeze(beta)
        self.sigma_sq = residual_sq / self.dof
        self.reml = -0.5 * (logdet_d + logdet_cdc + self.dof * np.log(self.sigma_sq) + self._shift)

        return -self.reml

    def _expand(self):
        assert self.dof > 1

        if not self.fitted:
            raise Exception("First fit with 'fit'.")

        delta = self.delta
        new_k = self.k + 1

        z = 1 / (self.s + self.delta) - 1 / delta
        zpy = z * self.py
        ydy = self._yty / delta + self.py @ zpy

        cdy = np.zeros(new_k)
        cdy[1:] = self._cty / delta + self.pc.T @ zpy

        cdc = np.zeros((new_k, new_k))
        cdc[1:, 1:] = self._ctc / delta + (self.pc.T * z) @ self.pc

        self._z = z
        self._ydy = ydy
        self._cdy = cdy
        self._cdc = cdc
        self._residual_sq = self.sigma_sq * self.dof
        self._expanded = True

    def _fit_local_row(self, x, px):
        from scipy.linalg import solve, LinAlgError
        from scipy.stats.distributions import chi2

        delta = self.delta
        zpx = self._z * px

        cdy = np.copy(self._cdy)
        cdy[0] = (self.y @ x) / delta + self.py @ zpx

        cdc = np.copy(self._cdc)
        cdc[0, 0] = (x @ x) / delta + px @ zpx
        cdc[0, 1:] = (self.c.T @ x) / delta + self.pc.T @ zpx  # only using upper triangle

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
            self.delta,
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
        self.delta = None
        self.beta = None
        self.sigma_sq = None
        self.reml = None
        self.optim = None

        self._z = None
        self._ydy = None
        self._cdy = None
        self._cdc = None
        self._residual_sq = None
        self._expanded = False

        self._scala_model = None
