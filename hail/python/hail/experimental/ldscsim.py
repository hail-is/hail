#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation framework for testing LDSC

Models for SNP effects:
    - Infinitesimal (can simulate n correlated traits)
    - Spike & slab (can simulate up to 2 correlated traits)
    - Annotation-informed

Features:
   - Field aggregation tools for annotation-informed model and
     population stratification with many covariates.
   - Automatic adjustment of genetic correlation parameters
     to allow for the joint simulation of up to 100 randomly
     correlated phenotypes.
   - Methods for binarizing phenotypes to have a certain prevalence
     and for adding ascertainment bias to binarized phenotypes.

@author: nbaya
"""

import numpy as np
import pandas as pd
from scipy import stats

import hail as hl
from hail.expr.expressions import expr_array, expr_call, expr_float64, expr_int32
from hail.matrixtable import MatrixTable
from hail.table import Table
from hail.typecheck import nullable, oneof, typecheck
from hail.utils.java import Env


@typecheck(
    mt=MatrixTable,
    genotype=oneof(expr_int32, expr_float64, expr_call),
    h2=(oneof(float, int, list, np.ndarray)),
    pi=nullable(oneof(float, int, list, np.ndarray)),
    rg=nullable(oneof(float, int, list, np.ndarray)),
    annot=nullable(oneof(expr_float64, expr_int32)),
    popstrat=nullable(oneof(expr_int32, expr_float64)),
    popstrat_var=nullable(oneof(float, int)),
    exact_h2=bool,
)
def simulate_phenotypes(
    mt, genotype, h2, pi=None, rg=None, annot=None, popstrat=None, popstrat_var=None, exact_h2=False
):
    r"""Simulate phenotypes for testing LD score regression.

    Simulates betas (SNP effects) under the infinitesimal, spike & slab, or
    annotation-informed models, depending on parameters passed. Optionally adds
    population stratification.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        :class:`.MatrixTable` containing genotypes to be used. Also should contain
        variant annotations as row fields if running the annotation-informed
        model or covariates as column fields if adding population stratification.
    genotype : :class:`.Expression` or :class:`.CallExpression`
        Entry field containing genotypes of individuals to be used for the
        simulation.
    h2 : :obj:`float` or :obj:`int` or :obj:`list` or :class:`numpy.ndarray`
        SNP-based heritability of simulated trait.
    pi : :obj:`float` or :obj:`int` or :obj:`list` or :class:`numpy.ndarray`, optional
        Probability of SNP being causal when simulating under the spike & slab
        model.
    rg : :obj:`float` or :obj:`int` or :obj:`list` or :class:`numpy.ndarray`, optional
        Genetic correlation between traits.
    annot : :class:`.Expression`, optional
        Row field to use as our aggregated annotations.
    popstrat: :class:`.Expression`, optional
        Column field to use as our aggregated covariates for adding population
        stratification.
    exact_h2: :obj:`bool`, optional
        Whether to exactly simulate ratio of variance of genetic component of
        phenotype to variance of phenotype to be h2. If `False`, ratio will be
        h2 in expectation. Observed h2 in the simulation will be close to
        expected h2 for large-scale simulations.

    Returns
    -------
    :class:`.MatrixTable`
        :class:`.MatrixTable` with simulated betas and phenotypes, simulated according
        to specified model.
    """
    h2 = h2.tolist() if isinstance(h2, np.ndarray) else ([h2] if not isinstance(h2, list) else h2)
    pi = pi.tolist() if isinstance(pi, np.ndarray) else pi
    uid = Env.get_uid(base=100)
    mt = annotate_all(
        mt=mt,
        row_exprs={} if annot is None else {'annot_' + uid: annot},
        col_exprs={} if popstrat is None else {'popstrat_' + uid: popstrat},
        entry_exprs={'gt_' + uid: genotype.n_alt_alleles() if genotype.dtype is hl.dtype('call') else genotype},
    )
    mt, pi, rg = make_betas(mt=mt, h2=h2, pi=pi, annot=None if annot is None else mt['annot_' + uid], rg=rg)
    mt = calculate_phenotypes(
        mt=mt,
        genotype=mt['gt_' + uid],
        beta=mt['beta'],
        h2=h2,
        popstrat=None if popstrat is None else mt['popstrat_' + uid],
        popstrat_var=popstrat_var,
        exact_h2=exact_h2,
    )
    mt = annotate_all(
        mt=mt,
        global_exprs={
            'ldscsim': hl.struct(**{
                'h2': h2[0] if len(h2) == 1 else h2,
                **({} if pi == [None] else {'pi': pi}),
                **({} if rg == [None] else {'rg': rg[0] if len(rg) == 1 else rg}),
                **({} if annot is None else {'is_annot_inf': True}),
                **({} if popstrat is None else {'is_popstrat_inf': True}),
                **({} if popstrat_var is None else {'popstrat_var': popstrat_var}),
                'exact_h2': exact_h2,
            })
        },
    )
    mt = _clean_fields(mt, uid)
    return mt


@typecheck(
    mt=MatrixTable,
    h2=(oneof(float, int, list, np.ndarray)),
    pi=nullable(oneof(float, int, list, np.ndarray)),
    annot=nullable(oneof(expr_float64, expr_int32)),
    rg=nullable(oneof(float, int, list, np.ndarray)),
)
def make_betas(mt, h2, pi=None, annot=None, rg=None):
    r"""Generates betas under different models.

    Simulates betas (SNP effects) under the infinitesimal, spike & slab, or
    annotation-informed models, depending on parameters passed.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        MatrixTable containing genotypes to be used. Also should contain
        variant annotations as row fields if running the annotation-informed
        model or covariates as column fields if adding population stratification.
    h2 : :obj:`float` or :obj:`int` or :obj:`list` or :class:`numpy.ndarray`
        SNP-based heritability of simulated trait(s).
    pi : :obj:`float` or :obj:`int` or :obj:`list` or :class:`numpy.ndarray`, optional
        Probability of SNP being causal when simulating under the spike & slab
        model. If doing two-trait spike & slab `pi` is a list of probabilities for
        overlapping causal SNPs (see docstring of :func:`.multitrait_ss`)
    annot : :class:`.Expression`, optional
        Row field of aggregated annotations for annotation-informed model.
    rg : :obj:`float` or :obj:`int` or :obj:`list` or :class:`numpy.ndarray`, optional
        Genetic correlation between traits.

    Returns
    -------
    mt : :class:`.MatrixTable`
        :class:`.MatrixTable` with betas as a row field, simulated according to specified model.
    pi : :obj:`list`
        Probability of a SNP being causal for different traits, possibly altered
        from input `pi` if covariance matrix for multitrait simulation was not
        positive semi-definite.
    rg : :obj:`list`
        Genetic correlation between traits, possibly altered from input `rg` if
        covariance matrix for multitrait simulation was not positive semi-definite.

    """
    h2 = h2.tolist() if isinstance(h2, np.ndarray) else ([h2] if not isinstance(h2, list) else h2)
    pi = pi.tolist() if isinstance(pi, np.ndarray) else ([pi] if not isinstance(pi, list) else pi)
    rg = rg.tolist() if isinstance(rg, np.ndarray) else ([rg] if not isinstance(rg, list) else rg)
    assert all(x >= 0 and x <= 1 for x in h2), 'h2 values must be between 0 and 1'
    assert (pi is not [None]) or all(
        x >= 0 and x <= 1 for x in pi
    ), 'pi values for spike & slab must be between 0 and 1'
    assert rg == [None] or all(x >= -1 and x <= 1 for x in rg), 'rg values must be between -1 and 1 or None'
    if annot is not None:  # multi-trait annotation-informed
        assert rg == [None], 'Correlated traits not supported for annotation-informed model'
        h2 = h2 if isinstance(h2, list) else [h2]
        annot_sum = mt.aggregate_rows(hl.agg.sum(annot))
        mt = mt.annotate_rows(beta=hl.literal(h2).map(lambda x: hl.rand_norm(0, hl.sqrt(annot * x / (annot_sum * M)))))
    elif len(h2) > 1 and (pi == [None] or pi == [1]):  # multi-trait correlated infinitesimal
        mt, rg = multitrait_inf(mt=mt, h2=h2, rg=rg)
    elif len(h2) == 2 and len(pi) > 1 and len(rg) == 1:  # two trait correlated spike & slab
        print('multitrait ss')
        mt, pi, rg = multitrait_ss(mt=mt, h2=h2, rg=0 if rg is [None] else rg[0], pi=pi)
    elif len(h2) == 1 and len(pi) == 1:  # single trait infinitesimal/spike & slab
        M = mt.count_rows()
        pi_temp = 1 if pi == [None] else pi[0]
        mt = mt.annotate_rows(beta=hl.rand_bool(pi_temp) * hl.rand_norm(0, hl.sqrt(h2[0] / (M * pi_temp))))
    else:
        raise ValueError('Parameters passed do not match any models.')
    return mt, pi, rg


@typecheck(
    mt=MatrixTable,
    h2=nullable(oneof(float, int, list, np.ndarray)),
    rg=nullable(oneof(float, int, list)),
    cov_matrix=nullable(np.ndarray),
    seed=nullable(int),
)
def multitrait_inf(mt, h2=None, rg=None, cov_matrix=None, seed=None):
    r"""Generates correlated betas for multi-trait infinitesimal simulations for
    any number of phenotypes.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        MatrixTable for simulated phenotype.
    h2 : :obj:`float` or :obj:`int` or :obj:`list` or :class:`numpy.ndarray`, optional
        Desired SNP-based heritability (:math:`h^2`) of simulated traits.
        If `h2` is ``None``, :math:`h^2` is based on diagonal of `cov_matrix`.
    rg : :obj:`float` or :obj:`int` or :obj:`list` or :class:`numpy.ndarray`, optional
        Desired genetic correlation (:math:`r_g`) between simulated traits.
        If simulating more than two correlated traits, `rg` should be a list
        of :math:`rg` values corresponding to the upper right triangle of the
        covariance matrix. If `rg` is ``None`` and `cov_matrix` is ``None``, :math:`r_g`
        is assumed to be 0 between traits. If `rg` and `cov_matrix` are both
        not None, :math:`r_g` values from `cov_matrix` take precedence.
    cov_matrix : :class:`numpy.ndarray`, optional
        Covariance matrix for traits, **unscaled by :math:`M`**, the number of SNPs.
        Overrides `h2` and `rg` even when `h2` or `rg` are not ``None``.
    seed : :obj:`int`, optional
        Seed for random number generator. If `seed` is ``None``, `seed` is set randomly.

    Returns
    -------
    mt : :class:`.MatrixTable`
        :class:`.MatrixTable` with simulated SNP effects as a row field of arrays.
    rg : :obj:`list`
        Genetic correlation between traits, possibly altered from input `rg` if
        covariance matrix was not positive semi-definite.
    """
    uid = Env.get_uid(base=100)
    h2 = h2.tolist() if isinstance(h2, np.ndarray) else ([h2] if not isinstance(h2, list) else h2)
    rg = rg.tolist() if isinstance(rg, np.ndarray) else ([rg] if not isinstance(rg, list) else rg)
    assert all(x >= 0 and x <= 1 for x in h2), 'h2 values must be between 0 and 1'
    assert h2 is not [None] or cov_matrix is not None, 'h2 and cov_matrix cannot both be None'
    M = mt.count_rows()
    if cov_matrix is not None:
        n_phens = cov_matrix.shape[0]
    else:
        n_phens = len(h2)
        if rg == [None]:
            print(f'Assuming rg=0 for all {n_phens} traits')
            rg = [0] * int((n_phens**2 - n_phens) / 2)
        assert all(x >= -1 and x <= 1 for x in rg), 'rg values must be between 0 and 1'
        cov, rg = get_cov_matrix(h2, rg)
    cov = (1 / M) * cov
    # seed random state for replicability
    randstate = np.random.RandomState(int(seed))
    betas = randstate.multivariate_normal(
        mean=np.zeros(n_phens),
        cov=cov,
        size=[
            M,
        ],
    )
    df = pd.DataFrame([0] * M, columns=['beta'])
    tb = hl.Table.from_pandas(df)
    tb = tb.add_index().key_by('idx')
    tb = tb.annotate(beta=hl.literal(betas.tolist())[hl.int32(tb.idx)])
    mt = mt.add_row_index(name='row_idx' + uid)
    mt = mt.annotate_rows(beta=tb[mt['row_idx' + uid]]['beta'])
    mt = _clean_fields(mt, uid)
    return mt, rg


@typecheck(
    mt=MatrixTable, h2=oneof(list, np.ndarray), pi=oneof(list, np.ndarray), rg=oneof(float, int), seed=nullable(int)
)
def multitrait_ss(mt, h2, pi, rg=0, seed=None):
    r"""Generates spike & slab betas for simulation of two correlated phenotypes.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        :class:`.MatrixTable` for simulated phenotype.
    h2 : :obj:`list` or :class:`numpy.ndarray`
        Desired SNP-based heritability of simulated traits.
    pi : :obj:`list` or :class:`numpy.ndarray`
        List of proportion of SNPs: :math:`p_{TT}`, :math:`p_{TF}`, :math:`p_{FT}`
        :math:`p_{TT}` is the proportion of SNPs that are causal for both traits,
        :math:`p_{TF}` is the proportion of SNPs that are causal for trait 1 but not trait 2,
        :math:`p_{FT}` is the proportion of SNPs that are causal for trait 2 but not trait 1.
    rg : :obj:`float` or :obj:`int`
        Genetic correlation between traits.
    seed : :obj:`int`, optional
        Seed for random number generator. If `seed` is ``None``, `seed` is set randomly.

    Warning
    -------
    May give inaccurate results if chosen parameters make the covariance matrix
    not positive semi-definite. Covariance matrix is likely to not be positive
    semi-definite when :math:`p_{TT}` is small and rg is large.

    Returns
    -------
    mt : :class:`.MatrixTable`
        :class:`.MatrixTable` with simulated SNP effects as a row field of arrays.
    pi : :obj:`list` or :class:`numpy.ndarray`
        List of proportion of SNPs: :math:`p_{TT}`, :math:`p_{TF}`, :math:`p_{FT}`.
        Possibly altered if covariance matrix of traits was not positive semi-definite.
    rg : :obj:`list`
        Genetic correlation between traits, possibly altered from input `rg` if
        covariance matrix was not positive semi-definite.
    """
    assert sum(pi) <= 1, "probabilities of being causal must sum to be less than 1"
    ptt, ptf, pft, pff = pi[0], pi[1], pi[2], 1 - sum(pi)
    cov_matrix = np.asarray([[1 / (ptt + ptf), rg / ptt], [rg / ptt, 1 / (ptt + pft)]])
    M = mt.count_rows()
    # seed random state for replicability
    randstate = np.random.RandomState(int(seed))
    if np.any(np.linalg.eigvals(cov_matrix) < 0):
        print('adjusting parameters to make covariance matrix positive semidefinite')
        rg0, ptt0 = rg, ptt
        while np.any(np.linalg.eigvals(cov_matrix) < 0):  # check positive semidefinite
            rg = round(0.99 * rg, 6)
            ptt = round(ptt + (pff) * 0.001, 6)
            cov_matrix = np.asarray([[1 / (ptt + ptf), rg / ptt], [rg / ptt, 1 / (ptt + pft)]])
        pff0, pff = pff, 1 - sum([ptt, ptf, pft])
        print(f'rg: {rg0} -> {rg}\nptt: {ptt0} -> {ptt}\npff: {pff0} -> {pff}')
        pi = [ptt, ptf, pft, pff]
    beta = randstate.multivariate_normal(
        mean=np.zeros(2),
        cov=cov_matrix,
        size=[
            int(M),
        ],
    )
    zeros = np.zeros(shape=int(M)).T
    beta_matrix = np.stack(
        (beta, np.asarray([beta[:, 0], zeros]).T, np.asarray([zeros, zeros]).T, np.asarray([zeros, beta[:, 1]]).T),
        axis=1,
    )
    idx = np.random.choice(a=[0, 1, 2, 3], size=int(M), p=[ptt, ptf, pft, pff])
    betas = beta_matrix[range(int(M)), idx, :]
    betas[:, 0] *= (h2[0] / M) ** (1 / 2)
    betas[:, 1] *= (h2[1] / M) ** (1 / 2)
    df = pd.DataFrame([0] * M, columns=['beta'])
    tb = hl.Table.from_pandas(df)
    tb = tb.add_index().key_by('idx')
    tb = tb.annotate(beta=hl.literal(betas.tolist())[hl.int32(tb.idx)])
    mt = mt.add_row_index()
    mt = mt.annotate_rows(beta=tb[mt.row_idx]['beta'])
    return mt, pi, [rg]


@typecheck(h2=oneof(list, np.ndarray), rg=oneof(list, np.ndarray), psd_rg=bool)
def get_cov_matrix(h2, rg, psd_rg=False):
    r"""Creates covariance matrix for simulating correlated SNP effects.

    Given a list of heritabilities and a list of genetic correlations, :func:`.get_cov_matrix`
    constructs the covariance matrix necessary to draw from a multivariate normal
    distribution to generate correlated SNP effects.

    Examples
    --------
    Suppose we have three traits enumerated as trait 1, trait 2, and trait 3.
    Each trait has a heritability: :math:`h^2_1`,:math:`h^2_2`,:math:`h^2_3`
    Traits have the following genetic correlations: :math:`r_{g, 12}`,:math:`r_{g, 13}`, :math:`r_{g, 23}`
    The ordering of indices in the subscript is arbitrary (e.g. :math:`r_{g, 12}` = :math:`r_{g, 21}`)
    as both values are the genetic correlation between trait 1 and trait 2.
    We can calculate :math:`\rho_{g,ab}`, the genetic covariance between two traits :math:`a` and :math:`b`,
    as :math:`\rho_{g,ab}=r_{g,ab}\sqrt{h^2_a\cdot h^2_b}`. The covariance matrix is thus:

    .. math::

        \begin{pmatrix}
        h^2_1                            & r_{g, 12}\sqrt{h^2_1\cdot h^2_2}  & r_{g, 13}\sqrt{h^2_1\cdot h^2_3} \\
        r_{g, 12}\sqrt{h^2_1\cdot h^2_2} & h^2_2                             & r_{g, 23}\sqrt{h^2_2\cdot h^2_3} \\
        r_{g, 13}\sqrt{h^2_1\cdot h^2_3} & r_{g, 23}*\sqrt{h^2_2\cdot h^2_3} & h^2_3
        \end{pmatrix}

    Now suppose we have four traits with the following heritabilities (:math:`h^2`): 0.1, 0.3, 0.2, 0.6.
    That is, trait 1 has an :math:`h^2` of 0.1, trait 2 has an :math:`h^2` of 0.3 and so on.
    Suppose the genetic correlations (:math:`r_g`) between traits are the following:
    trait 1 & trait 2 :math:`r_g` = 0.4
    trait 1 & trait 3 :math:`r_g` = 0.3
    trait 1 & trait 4 :math:`r_g` = 0.1
    trait 2 & trait 3 :math:`r_g` = 0.2
    trait 2 & trait 4 :math:`r_g` = 0.15
    trait 3 & trait 4 :math:`r_g` = 0.6
    To obtain the covariance matrix corresponding to this scenario :math:`h^2` values are
    ordered according to user specification and :math:`r_g` values are ordered by the
    order in which the corresponding genetic covariance terms will appear in the
    covariance matrix, reading lines in the upper triangular matrix from left to
    right, top to bottom (read first row left to right, read second row left to
    right, etc.), exluding the diagonal.

    >>> cov_matrix, rg = get_cov_matrix(h2=[0.1, 0.3, 0.2, 0.6], rg=[0.4, 0.3, 0.1, 0.2, 0.15, 0.6])
    >>> cov_matrix
    array([[0.1       , 0.06928203, 0.04242641, 0.0244949 ],
           [0.06928203, 0.3       , 0.04898979, 0.06363961],
           [0.04242641, 0.04898979, 0.2       , 0.2078461 ],
           [0.0244949 , 0.06363961, 0.2078461 , 0.6       ]])

    The diagonal corresponds directly to `h2`, the list of h2 values for all traits.
    In the upper triangular matrix, excluding the diagonal, the entry :math:`(a, b)`,
    where :math:`a` and :math:`b` are in :math:`{1,2,3,4}`, is the genetic covariance
    (:math:`\rho_g`) between traits :math:`a` and :math:`b`.
    Genetic covariance is calculated as :math:`\rho_g= r_g*\sqrt{h^2_a*h^2_b}`
    where :math:`r_g` is the genetic correlation between traits :math:`a` and
    :math:`b` and :math:`h^2_a` and :math:`h^2_b` are heritabilities corresponding
    to traits :math:`a` and :math:`b`.

    Notes
    -----
    Covariance matrix is not scaled by number of SNPs.

    If the h2 and rg parameters passed cause the resulting covariance matrix to
    not be positive semidefinite, this may cause the distribution of SNP effects
    generated by this covariance matrix to not have the properties specified by
    the h2 and rg parameters. To automatically adjust rg values so that the
    covariance matrix is positive semidefinite, set `psd_rg` = True.

    Parameters
    ----------
    h2 : :obj:`list` or :class:`numpy.ndarray`
        :math:`h^2` values for traits. :math:`h^2` values in list should be
        ordered by their order in the diagonal of the covariance array, reading
        from top left to bottom right.
    rg : :obj:`list` or :class:`numpy.ndarray`
        :math:`r_g` values for traits. :math:`r_g` values should be ordered in
        the order they appear in the upper triangle of the covariance matrix,
        from left to right, top to bottom.
    psd_rg :  :obj:`bool`
        Whether to automatically adjust rg values to get a positive semi-definite
        covariance matrix, which ensures that SNP effects simulated with that
        covariance matrix have the desired variance and correlation properties
        specified by the h2 and rg parameters.

    Returns
    -------
    cov_matrix : :class:`numpy.ndarray`
        Covariance matrix calculated using `h2` and (possibly altered) `rg` values.
    rg : :obj:`list`
        Genetic correlation between traits, possibly altered from input `rg` if
        covariance matrix was not positive semi-definite.
    """
    assert all(x >= 0 and x <= 1 for x in h2), 'h2 values must be between 0 and 1'
    assert all(x >= -1 and x <= 1 for x in rg), 'rg values must be between -1 and 1'
    rg = np.asarray(rg) if isinstance(rg, list) else rg
    n_rg = len(rg)
    n_h2 = len(h2)
    # expected number of rg values, given number of traits
    exp_n_rg = int((n_h2**2 - n_h2) / 2)
    assert n_rg == exp_n_rg, f'The number of rg values given is {n_rg}, expected {exp_n_rg}'
    cor = np.zeros(shape=(n_h2, n_h2))
    # set upper triangle of correlation matrix to be rg
    cor[np.triu_indices(n=n_h2, k=1)] = rg
    cor += cor.T
    cor[np.diag_indices(n=n_h2)] = 1
    if psd_rg:
        cor0 = cor
        cor = _nearpsd(cor)
        idx = np.triu_indices(n=n_h2, k=1)
        maxlines = 50
        msg = ['Adjusting rg values to make covariance matrix positive semidefinite']
        msg += (
            [(f'{cor0[idx[0][i],idx[1][i]]} -> {cor[idx[0][i],idx[1][i]]}') for i in range(n_rg)]
            if n_rg <= maxlines
            else [(f'{cor0[idx[0][i],idx[1][i]]} -> {cor[idx[0][i],idx[1][i]]}') for i in range(maxlines)]
            + [f'[ printed first {maxlines} rg changes -- omitted {n_rg - maxlines} ]']
        )
        print('\n'.join(msg))
        rg = np.ravel(cor[idx])
    S = np.diag(h2) ** (1 / 2)
    cov_matrix = S @ cor @ S  # covariance matrix decomposition

    # check positive semidefinite
    if not np.all(np.linalg.eigvals(cov_matrix) >= 0) and not psd_rg:
        msg = 'WARNING: Covariance matrix is not positive semidefinite.\n'
        msg += 'Multivariate Gaussian distributions generated with this \n'
        msg += 'covariance matrix may not have the desired h2 and rg.\n'
        msg += 'To make the covariance matrix positive semidefinite,\n'
        msg += 'adjust h2 and rg values or set psd_rg=True.'
        print(msg)
    rg = rg.tolist()
    return cov_matrix, rg


@typecheck(A=np.ndarray)
def _nearpsd(A):
    r"""Obtain the "closest" positive semidefinite matrix to A."""
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval, 0))
    vec = np.matrix(eigvec)
    T = 1 / (np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = np.real(B * B.T)
    return out


@typecheck(
    mt=MatrixTable,
    genotype=oneof(expr_int32, expr_float64, expr_call),
    beta=oneof(expr_float64, expr_array(expr_float64)),
    h2=oneof(float, int, list, np.ndarray),
    popstrat=nullable(oneof(expr_int32, expr_float64)),
    popstrat_var=nullable(oneof(float, int)),
    exact_h2=bool,
)
def calculate_phenotypes(mt, genotype, beta, h2, popstrat=None, popstrat_var=None, exact_h2=False):
    r"""Calculates phenotypes by multiplying genotypes and betas.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        :class:`.MatrixTable` with all relevant fields passed as parameters.
    genotype : :class:`.Expression` or :class:`.CallExpression`
        Entry field of genotypes.
    beta : :class:`.Expression`
        Row field of SNP effects.
    h2 : :obj:`float` or :obj:`int` or :obj:`list` or :class:`numpy.ndarray`
        SNP-based heritability (:math:`h^2`) of simulated trait. Can only be
        ``None`` if running annotation-informed model.
    popstrat : :class:`.Expression`, optional
        Column field containing population stratification term.
    popstrat_var : :obj:`float` or :obj:`int`
        Variance of population stratification term.
    exact_h2: :obj:`bool`
        Whether to exactly simulate ratio of variance of genetic component of
        phenotype to variance of phenotype to be h2. If `False`, ratio will be
        h2 in expectation. Observed h2 in the simulation will be close to
        expected h2 for large-scale simulations.

    Returns
    -------
    :class:`.MatrixTable`
        :class:`.MatrixTable` with simulated phenotype as column field.
    """
    print('calculating phenotype')
    h2 = h2.tolist() if isinstance(h2, np.ndarray) else ([h2] if not isinstance(h2, list) else h2)
    assert popstrat_var is None or (popstrat_var >= 0), 'popstrat_var must be non-negative'
    uid = Env.get_uid(base=100)
    mt = annotate_all(
        mt=mt,
        row_exprs={'beta_' + uid: beta},
        col_exprs={} if popstrat is None else {'popstrat_' + uid: popstrat},
        entry_exprs={'gt_' + uid: genotype.n_alt_alleles() if genotype.dtype is hl.dtype('call') else genotype},
    )
    mt = mt.filter_rows(hl.agg.stats(mt['gt_' + uid]).stdev > 0)
    mt = normalize_genotypes(mt['gt_' + uid])
    if mt['beta_' + uid].dtype == hl.dtype('array<float64>'):  # if >1 traits
        if exact_h2:
            raise ValueError('exact_h2=True not supported for multitrait simulations')
        else:
            mt = mt.annotate_cols(
                y_no_noise=hl.agg.array_agg(lambda beta: hl.agg.sum(beta * mt['norm_gt']), mt['beta_' + uid])
            )
            mt = mt.annotate_cols(y=mt.y_no_noise + hl.literal(h2).map(lambda x: hl.rand_norm(0, hl.sqrt(1 - x))))
    else:
        if exact_h2 and min([h2[0], 1 - h2[0]]) != 0:
            print('exact h2')
            mt = mt.annotate_cols(**{'y_no_noise_' + uid: hl.agg.sum(mt['beta_' + uid] * mt['norm_gt'])})
            y_no_noise_stdev = mt.aggregate_cols(hl.agg.stats(mt['y_no_noise_' + uid]).stdev)
            mt = mt.annotate_cols(
                y_no_noise=hl.sqrt(h2[0]) * mt['y_no_noise_' + uid] / y_no_noise_stdev
            )  # normalize genetic component of phenotype to have variance of exactly h2
            mt = mt.annotate_cols(**{'noise_' + uid: hl.rand_norm(0, hl.sqrt(1 - h2[0]))})
            noise_stdev = mt.aggregate_cols(hl.agg.stats(mt['noise_' + uid]).stdev)
            mt = mt.annotate_cols(noise=hl.sqrt(1 - h2[0]) * mt['noise_' + uid] / noise_stdev)
            mt = mt.annotate_cols(y=mt.y_no_noise + hl.sqrt(1 - h2[0]) * mt['noise_' + uid] / noise_stdev)
        else:
            mt = mt.annotate_cols(y_no_noise=hl.agg.sum(mt['beta_' + uid] * mt['norm_gt']))
            mt = mt.annotate_cols(y=mt.y_no_noise + hl.rand_norm(0, hl.sqrt(1 - h2[0])))
    if popstrat is not None:
        var_factor = (
            1
            if popstrat_var is None
            else (popstrat_var ** (1 / 2)) / mt.aggregate_cols(hl.agg.stats(mt['popstrat_' + uid])).stdev
        )
        mt = mt.rename({'y': 'y_no_popstrat'})
        mt = mt.annotate_cols(y=mt.y_no_popstrat + mt['popstrat_' + uid] * var_factor)
    mt = _clean_fields(mt, uid)
    return mt


@typecheck(genotype=oneof(expr_int32, expr_float64, expr_call))
def normalize_genotypes(genotype):
    r"""Normalizes genotypes to have mean 0 and variance 1 at each SNP

    Parameters
    ----------
    genotype : :class:`.Expression` or :class:`.CallExpression`
        Entry field of genotypes.

    Returns
    -------
    :class:`.MatrixTable`
        :class:`.MatrixTable` with normalized genotypes.
    """
    uid = Env.get_uid(base=100)
    mt = genotype._indices.source
    mt = mt.annotate_entries(**{
        'gt_' + uid: genotype.n_alt_alleles() if genotype.dtype is hl.dtype('call') else genotype
    })
    mt = mt.annotate_rows(**{'gt_stats_' + uid: hl.agg.stats(mt['gt_' + uid])})
    # TODO: Add MAF filter to remove invariant SNPs?
    mt = mt.annotate_entries(norm_gt=(mt['gt_' + uid] - mt['gt_stats_' + uid].mean) / mt['gt_stats_' + uid].stdev)
    mt = _clean_fields(mt, uid)
    return mt


@typecheck(mt=MatrixTable, str_expr=str)
def _clean_fields(mt, str_expr):
    r"""Removes fields with names that have `str_expr` in them.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        :class:`.MatrixTable` with fields to be removed.
    str_expr : :class:`str`
        string to filter names of fields to remove.

    Returns
    -------
    :class:`.MatrixTable`
        :class:`.MatrixTable` with specified fields removed.
    """
    all_fields = list(mt.col) + list(mt.row) + list(mt.entry) + list(mt.globals)
    return mt.drop(*(x for x in all_fields if str_expr in x))


@typecheck(mt=MatrixTable, row_exprs=dict, col_exprs=dict, entry_exprs=dict, global_exprs=dict)
def annotate_all(mt, row_exprs={}, col_exprs={}, entry_exprs={}, global_exprs={}):
    r"""Equivalent of _annotate_all, but checks source MatrixTable of exprs"""
    exprs = {**row_exprs, **col_exprs, **entry_exprs, **global_exprs}
    for key, value in exprs.items():
        if value.dtype in (hl.tfloat64, hl.tint32):
            assert value._indices.source == mt, 'Cannot combine expressions from different source objects.'
    return mt._annotate_all(row_exprs, col_exprs, entry_exprs, global_exprs)


@typecheck(tb=oneof(MatrixTable, Table), coef_dict=nullable(dict), str_expr=nullable(str), axis=str)
def agg_fields(tb, coef_dict=None, str_expr=None, axis='rows'):
    r"""Aggregates by linear combination fields matching either keys in `coef_dict`
    or `str_expr`. Outputs the aggregation in a :class:`.MatrixTable` or :class:`.Table`
    as a new row field "agg_annot" or a new column field "agg_cov".

    Parameters
    ----------
    tb : :class:`.MatrixTable` or :class:`.Table`
        :class:`.MatrixTable` or :class:`.Table` containing fields to be aggregated.
    coef_dict : :obj:`dict`, optional
        Coefficients to multiply each field. The coefficients are specified by
        `coef_dict` value, the row (or col) field name is specified by `coef_dict` key.
        If not included, coefficients are assumed to be 1.
    str_expr : :class:`str`, optional
        String expression to match against row (or col) field names.
    axis : :class:`str`
        Either 'rows' or 'cols'. If 'rows', this aggregates across row fields.
        If 'cols', this aggregates across col fields. If tb is a Table, axis = 'rows'.

    Returns
    -------
    :class:`.MatrixTable` or :class:`.Table`
        :class:`.MatrixTable` or :class:`.Table` containing aggregation field.
    """
    assert str_expr is not None or coef_dict is not None, "str_expr and coef_dict cannot both be None"
    assert axis == 'rows' or axis == 'cols', "axis must be 'rows' or 'cols'"
    coef_dict = get_coef_dict(tb=tb, str_expr=str_expr, ref_coef_dict=coef_dict, axis=axis)
    axis_field = 'annot' if axis == 'rows' else 'cov'
    annotate_fn = (
        (MatrixTable.annotate_rows if axis == 'rows' else MatrixTable.annotate_cols)
        if isinstance(tb, MatrixTable)
        else Table.annotate
    )
    tb = annotate_fn(self=tb, **{'agg_' + axis_field: 0})
    print(f'Fields and associated coefficients used in {axis_field} aggregation: {coef_dict}')
    for field, coef in coef_dict.items():
        tb = annotate_fn(self=tb, **{'agg_' + axis_field: tb['agg_' + axis_field] + coef * tb[field]})
    return tb


@typecheck(tb=oneof(MatrixTable, Table), str_expr=nullable(str), ref_coef_dict=nullable(dict), axis=str)
def get_coef_dict(tb, str_expr=None, ref_coef_dict=None, axis='rows'):
    r"""Gets either col or row fields matching `str_expr` and take intersection
    with keys in coefficient reference dict.

    Parameters
    ----------
    tb : :class:`.MatrixTable` or :class:`.Table`
        :class:`.MatrixTable` or :class:`.Table` containing row (or col) for `coef_dict`.
    str_expr : :class:`str`, optional
        String expression pattern to match against row (or col) fields. If left
        unspecified, the intersection of field names is only between existing
        row (or col) fields in `mt` and keys of `ref_coef_dict`.
    ref_coef_dict : :obj:`dict`, optional
        Reference coefficient dictionary with keys that are row (or col) field
        names from which to subset. If not included, coefficients are assumed to be 1.
    axis : :class:`str`
        Field type in which to search for field names. Options: 'rows', 'cols'

    Returns
    -------
    coef_dict : :obj:`dict`
        Coefficients to multiply each field. The coefficients are specified by
        `coef_dict` value, the row (or col) field name is specified by `coef_dict` key.
    """
    assert str_expr is not None or ref_coef_dict is not None, "str_expr and ref_coef_dict cannot both be None"
    assert axis == 'rows' or axis == 'cols', "axis must be 'rows' or 'cols'"
    fields_to_search = tb.row if axis == 'rows' or isinstance(tb, Table) else tb.col
    # when axis='rows' we're searching for annotations, axis='cols' searching for covariates
    axis_field = 'annotation' if axis == 'rows' else 'covariate'
    if str_expr is None:
        # take all row (or col) fields in mt matching keys in coef_dict
        coef_dict = {k: ref_coef_dict[k] for k in ref_coef_dict.keys() if k in fields_to_search}
        # if intersect is empty: return error
        assert len(coef_dict) > 0, f'None of the keys in ref_coef_dict match any {axis[:-1]} fields'
        return coef_dict  # return subset of ref_coef_dict
    else:
        # str_expr search in list of row (or col) fields
        fields = [rf for rf in list(fields_to_search) if str_expr in rf]
        assert len(fields) > 0, f'No {axis[:-1]} fields matched str_expr search: {str_expr}'
        if ref_coef_dict is None:
            print(f'Assuming coef = 1 for all {axis_field}s')
            return {k: 1 for k in fields}
        in_ref_coef_dict = set(fields).intersection(set(ref_coef_dict.keys()))  # fields in ref_coef_dict
        # if >0 fields returned by search are not in ref_coef_dict
        if in_ref_coef_dict != set(fields):
            # if none of the fields returned by search are in ref_coef_dict
            assert len(in_ref_coef_dict) > 0, f'None of the {axis_field} fields in ref_coef_dict match search results'
            fields_to_ignore = set(fields).difference(in_ref_coef_dict)
            print(f'Ignored fields from {axis_field} search: {fields_to_ignore}')
            print('To include ignored fields, change str_expr to match desired fields')
            fields = list(in_ref_coef_dict)
        return {k: ref_coef_dict[k] for k in fields}


@typecheck(mt=MatrixTable, y=expr_int32, P=oneof(int, float))
def ascertainment_bias(mt, y, P):
    r"""Adds ascertainment bias to a binary phenotype to give it a sample
    prevalence of `P` = cases/(cases+controls).

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        :class:`.MatrixTable` containing binary phenotype to be used.
    y : :class:`.Expression`
        Column field of binary phenotype.
    P : :obj:`int` or :obj:`float`
        Desired "sample prevalence" of phenotype.

    Returns
    -------
    :class:`.MatrixTable`
        :class:`.MatrixTable` containing binary phenotype with prevalence of approx. P
    """
    assert P >= 0 and P <= 1, 'P must be in [0,1]'
    uid = Env.get_uid(base=100)
    mt = mt.annotate_cols(y_w_asc_bias=y)
    y_stats = mt.aggregate_cols(hl.agg.stats(mt.y_w_asc_bias))
    K = y_stats.mean
    n = y_stats.n
    assert abs(P - K) < 1, 'Specified sample prevalence is incompatible with population prevalence.'
    if P < K:
        p = (1 - K) * P / (K * (1 - P))
        con = mt.filter_cols(mt.y_w_asc_bias == 0)
        cas = mt.filter_cols(mt.y_w_asc_bias == 1).add_col_index(name='col_idx_' + uid)
        keep = round(p * n * K) * [1] + round((1 - p) * n * K) * [0]
        cas = cas.annotate_cols(**{'keep_' + uid: hl.literal(keep)[hl.int32(cas['col_idx_' + uid])]})
        cas = cas.filter_cols(cas['keep_' + uid] == 1)
        cas = _clean_fields(cas, uid)
        mt = cas.union_cols(con)
    elif P > K:
        p = K * (1 - P) / ((1 - K) * P)
        cas = mt.filter_cols(mt.y_w_asc_bias == 1)
        con = mt.filter_cols(mt.y_w_asc_bias == 0).add_col_index(name='col_idx_' + uid)
        keep = round(p * n * (1 - K)) * [1] + round((1 - p) * n * (1 - K)) * [0]
        con = con.annotate_cols(**{'keep_' + uid: hl.literal(keep)[hl.int32(con['col_idx_' + uid])]})
        con = con.filter_cols(con['keep_' + uid] == 1)
        con = _clean_fields(con, uid)
        mt = con.union_cols(cas)
    return mt


@typecheck(mt=MatrixTable, y=oneof(expr_int32, expr_float64), K=oneof(int, float), exact=bool)
def binarize(mt, y, K, exact=False):
    r"""Binarize phenotype `y` such that it has prevalence `K` = cases/(cases+controls)
    Uses inverse CDF of Gaussian to set binarization threshold when `exact` = False,
    otherwise uses ranking to determine threshold.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        :class:`.MatrixTable` containing phenotype to be binarized.
    y : :class:`.Expression`
        Column field of phenotype.
    K : :obj:`int` or :obj:`float`
        Desired "population prevalence" of phenotype.
    exact : :obj:`bool`
        Whether to get prevalence as close as possible to `K` (does not use inverse CDF)

    Returns
    -------
    :class:`.MatrixTable`
        :class:`.MatrixTable` containing binary phenotype with prevalence of approx. `K`
    """
    if exact:
        key = list(mt.col_key)
        uid = Env.get_uid(base=100)
        mt = mt.annotate_cols(**{'y_' + uid: y})
        tb = mt.cols().order_by('y_' + uid)
        tb = tb.add_index('idx_' + uid)
        n = tb.count()
        # "+ 1" because of zero indexing
        tb = tb.annotate(y_binarized=tb['idx_' + uid] + 1 <= round(n * K))
        tb, mt = tb.key_by('y_' + uid), mt.key_cols_by('y_' + uid)
        mt = mt.annotate_cols(y_binarized=tb[mt['y_' + uid]].y_binarized)
        mt = mt.key_cols_by(*map(lambda x: mt[x], key))
    else:  # use inverse CDF
        y_stats = mt.aggregate_cols(hl.agg.stats(y))
        threshold = stats.norm.ppf(1 - K, loc=y_stats.mean, scale=y_stats.stdev)
        mt = mt.annotate_cols(y_binarized=y > threshold)
    return mt
