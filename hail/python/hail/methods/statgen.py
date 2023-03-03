import builtins
import itertools
import math
from typing import Dict, Callable, Optional, Union, Tuple, List

import hail
import hail as hl
import hail.expr.aggregators as agg
from hail import ir
from hail.expr import (Expression, ExpressionException, expr_float64, expr_call,
                       expr_any, expr_numeric, expr_locus, analyze, check_entry_indexed,
                       check_row_indexed, matrix_table_source, table_source)
from hail.expr.types import tbool, tarray, tfloat64, tint32
from hail.genetics.reference_genome import reference_genome_type
from hail.linalg import BlockMatrix
from hail.matrixtable import MatrixTable
from hail.methods.misc import require_biallelic, require_row_key_variant
from hail.stats import LinearMixedModel
from hail.table import Table
from hail.typecheck import (typecheck, nullable, numeric, oneof, sized_tupleof,
                            sequenceof, enumeration, anytype)
from hail.utils import wrap_to_list, new_temp_file, FatalError
from hail.utils.java import Env, info, warning
from . import pca
from . import relatedness
from ..backend.spark_backend import SparkBackend

pc_relate = relatedness.pc_relate
identity_by_descent = relatedness.identity_by_descent
_blanczos_pca = pca._blanczos_pca
_hwe_normalized_blanczos = pca._hwe_normalized_blanczos
_spectral_moments = pca._spectral_moments
_pca_and_moments = pca._pca_and_moments
hwe_normalized_pca = pca.hwe_normalized_pca
pca = pca.pca


tvector64 = hl.tndarray(hl.tfloat64, 1)
tmatrix64 = hl.tndarray(hl.tfloat64, 2)
numerical_regression_fit_dtype = hl.tstruct(
    b=tvector64,
    score=tvector64,
    fisher=tmatrix64,
    mu=tvector64,
    n_iterations=hl.tint32,
    log_lkhd=hl.tfloat64,
    converged=hl.tbool,
    exploded=hl.tbool)


@typecheck(call=expr_call,
           aaf_threshold=numeric,
           include_par=bool,
           female_threshold=numeric,
           male_threshold=numeric,
           aaf=nullable(str))
def impute_sex(call, aaf_threshold=0.0, include_par=False, female_threshold=0.2, male_threshold=0.8, aaf=None) -> Table:
    r"""Impute sex of samples by calculating inbreeding coefficient on the
    X chromosome.

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    Remove samples where imputed sex does not equal reported sex:

    >>> imputed_sex = hl.impute_sex(dataset.GT)
    >>> dataset_result = dataset.filter_cols(imputed_sex[dataset.s].is_female != dataset.pheno.is_female,
    ...                                      keep=False)

    Notes
    -----

    We have used the same implementation as `PLINK v1.7
    <https://zzz.bwh.harvard.edu/plink/summary.shtml#sexcheck>`__.

    Let `gr` be the the reference genome of the type of the `locus` key (as
    given by :attr:`.tlocus.reference_genome`)

    1. Filter the dataset to loci on the X contig defined by `gr`.

    2. Calculate alternate allele frequency (AAF) for each row from the dataset.

    3. Filter to variants with AAF above `aaf_threshold`.

    4. Remove loci in the pseudoautosomal region, as defined by `gr`, unless
       `include_par` is ``True`` (it defaults to ``False``)

    5. For each row and column with a non-missing genotype call, :math:`E`, the
       expected number of homozygotes (from population AAF), is computed as
       :math:`1.0 - (2.0*\mathrm{maf}*(1.0-\mathrm{maf}))`.

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
        :class:`.tarray` of :obj:`.tstr`. Moreover, the alleles array must have
        exactly two elements (i.e. the variant must be biallelic).
    aaf_threshold : :obj:`float`
        Minimum alternate allele frequency threshold.
    include_par : :obj:`bool`
        Include pseudoautosomal regions.
    female_threshold : :obj:`float`
        Samples are called females if F < female_threshold.
    male_threshold : :obj:`float`
        Samples are called males if F > male_threshold.
    aaf : :class:`str` or :obj:`None`
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
        is_female=hl.if_else(mt.ib.f_stat < female_threshold,
                             True,
                             hl.if_else(mt.ib.f_stat > male_threshold,
                                        False,
                                        hl.missing(tbool))),
        **mt.ib).cols()

    return kt


def _get_regression_row_fields(mt, pass_through, method) -> Dict[str, str]:

    row_fields = dict(zip(mt.row_key.keys(), mt.row_key.keys()))
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
    for k in mt.row_key:
        del row_fields[k]
    return row_fields


@typecheck(y=oneof(expr_float64, sequenceof(expr_float64), sequenceof(sequenceof(expr_float64))),
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           block_size=int,
           pass_through=sequenceof(oneof(str, Expression)),
           weights=nullable(oneof(expr_float64, sequenceof(expr_float64))))
def linear_regression_rows(y, x, covariates, block_size=16, pass_through=(), *, weights=None) -> hail.Table:
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
    :class:`.tarray` of :py:data:`.tfloat64`, with corresponding indexing of
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
    pass_through : :obj:`list` of :class:`str` or :class:`.Expression`
        Additional row fields to include in the resulting table.
    weights : :class:`.Float64Expression` or :obj:`list` of :class:`.Float64Expression`
        Optional column-indexed weighting for doing weighted least squares regression. Specify a single weight if a
        single y or list of ys is specified. If a list of lists of ys is specified, specify one weight per inner list.

    Returns
    -------
    :class:`.Table`
    """
    if not isinstance(Env.backend(), SparkBackend) or weights is not None:
        return _linear_regression_rows_nd(y, x, covariates, block_size, weights, pass_through)

    mt = matrix_table_source('linear_regression_rows/x', x)
    check_entry_indexed('linear_regression_rows/x', x)

    y_is_list = isinstance(y, list)
    if y_is_list and len(y) == 0:
        raise ValueError("'linear_regression_rows': found no values for 'y'")
    is_chained = y_is_list and isinstance(y[0], list)
    if is_chained and any(len(lst) == 0 for lst in y):
        raise ValueError("'linear_regression_rows': found empty inner list for 'y'")

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
        func = 'LinearRegressionRowsChained'

    else:
        y_field_names = list(f'__y_{i}' for i in range(len(y)))
        y_dict = dict(zip(y_field_names, y))
        func = 'LinearRegressionRowsSingle'

    cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))

    row_fields = _get_regression_row_fields(mt, pass_through, 'linear_regression_rows')

    # FIXME: selecting an existing entry field should be emitted as a SelectFields
    mt = mt._select_all(col_exprs=dict(**y_dict,
                                       **dict(zip(cov_field_names, covariates))),
                        row_exprs=row_fields,
                        col_key=[],
                        entry_exprs={x_field_name: x})

    config = {
        'name': func,
        'yFields': y_field_names,
        'xField': x_field_name,
        'covFields': cov_field_names,
        'rowBlockSize': block_size,
        'passThrough': [x for x in row_fields if x not in mt.row_key]
    }
    ht_result = Table(ir.MatrixToTableApply(mt._mir, config))

    if not y_is_list:
        fields = ['y_transpose_x', 'beta', 'standard_error', 't_stat', 'p_value']
        ht_result = ht_result.annotate(**{f: ht_result[f][0] for f in fields})

    return ht_result.persist()


@typecheck(y=oneof(expr_float64, sequenceof(expr_float64), sequenceof(sequenceof(expr_float64))),
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           block_size=int,
           weights=nullable(oneof(expr_float64, sequenceof(expr_float64))),
           pass_through=sequenceof(oneof(str, Expression)))
def _linear_regression_rows_nd(y, x, covariates, block_size=16, weights=None, pass_through=()) -> hail.Table:
    mt = matrix_table_source('linear_regression_rows_nd/x', x)
    check_entry_indexed('linear_regression_rows_nd/x', x)

    y_is_list = isinstance(y, list)
    if y_is_list and len(y) == 0:
        raise ValueError("'linear_regression_rows_nd': found no values for 'y'")
    is_chained = y_is_list and isinstance(y[0], list)

    if is_chained and any(len(lst) == 0 for lst in y):
        raise ValueError("'linear_regression_rows': found empty inner list for 'y'")

    y = wrap_to_list(y)

    if weights is not None:
        if y_is_list and is_chained and not isinstance(weights, list):
            raise ValueError("When y is a list of lists, weights should be a list.")
        elif y_is_list and not is_chained and isinstance(weights, list):
            raise ValueError("When y is a single list, weights should be a single expression.")
        elif not y_is_list and isinstance(weights, list):
            raise ValueError("When y is a single expression, weights should be a single expression.")

    weights = wrap_to_list(weights) if weights is not None else None

    for e in (itertools.chain.from_iterable(y) if is_chained else y):
        analyze('linear_regression_rows_nd/y', e, mt._col_indices)

    for e in covariates:
        analyze('linear_regression_rows_nd/covariates', e, mt._col_indices)

    _warn_if_no_intercept('linear_regression_rows_nd', covariates)

    x_field_name = Env.get_uid()
    if is_chained:
        y_field_name_groups = [[f'__y_{i}_{j}' for j in range(len(y[i]))] for i in range(len(y))]
        y_dict = dict(zip(itertools.chain.from_iterable(y_field_name_groups), itertools.chain.from_iterable(y)))
        if weights is not None and len(weights) != len(y):
            raise ValueError("Must specify same number of weights as groups of phenotypes")
    else:
        y_field_name_groups = list(f'__y_{i}' for i in range(len(y)))
        y_dict = dict(zip(y_field_name_groups, y))
        # Wrapping in a list since the code is written for the more general chained case.
        y_field_name_groups = [y_field_name_groups]
        if weights is not None and len(weights) != 1:
            raise ValueError("Must specify same number of weights as groups of phenotypes")

    cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))
    weight_field_names = list(f'__weight_for_group_{i}' for i in range(len(weights))) if weights is not None else None
    weight_dict = dict(zip(weight_field_names, weights)) if weights is not None else {}

    row_field_names = _get_regression_row_fields(mt, pass_through, 'linear_regression_rows_nd')

    # FIXME: selecting an existing entry field should be emitted as a SelectFields
    mt = mt._select_all(col_exprs=dict(**y_dict,
                                       **weight_dict,
                                       **dict(zip(cov_field_names, covariates))),
                        row_exprs=row_field_names,
                        col_key=[],
                        entry_exprs={x_field_name: x})

    entries_field_name = 'ent'
    sample_field_name = "by_sample"

    num_y_lists = len(y_field_name_groups)

    # Given a hail array, get the mean of the nonmissing entries and
    # return new array where the missing entries are the mean.
    def mean_impute(hl_array):
        non_missing_mean = hl.mean(hl_array, filter_missing=True)
        return hl_array.map(lambda entry: hl.if_else(hl.is_defined(entry), entry, non_missing_mean))

    def select_array_indices(hl_array, indices):
        return indices.map(lambda i: hl_array[i])

    def dot_rows_with_themselves(matrix):
        return (matrix * matrix).sum(1)

    def no_missing(hail_array):
        return hail_array.all(lambda element: hl.is_defined(element))

    ht_local = mt._localize_entries(entries_field_name, sample_field_name)

    ht = ht_local.transmute(**{entries_field_name: ht_local[entries_field_name][x_field_name]})

    def setup_globals(ht):
        # cov_arrays is per sample, then per cov.
        if covariates:
            ht = ht.annotate_globals(cov_arrays=ht[sample_field_name].map(lambda sample_struct: [sample_struct[cov_name] for cov_name in cov_field_names]))
        else:
            ht = ht.annotate_globals(cov_arrays=ht[sample_field_name].map(lambda sample_struct: hl.empty_array(hl.tfloat64)))

        y_arrays_per_group = [ht[sample_field_name].map(lambda sample_struct: [sample_struct[y_name] for y_name in one_y_field_name_set]) for one_y_field_name_set in y_field_name_groups]

        if weight_field_names:
            weight_arrays = ht[sample_field_name].map(lambda sample_struct: [sample_struct[weight_name] for weight_name in weight_field_names])
        else:
            weight_arrays = ht[sample_field_name].map(lambda sample_struct: hl.empty_array(hl.tfloat64))

        ht = ht.annotate_globals(
            y_arrays_per_group=y_arrays_per_group,
            weight_arrays=weight_arrays
        )
        ht = ht.annotate_globals(all_covs_defined=ht.cov_arrays.map(lambda sample_covs: no_missing(sample_covs)))

        def get_kept_samples(group_idx, sample_ys):
            # sample_ys is an array of samples, with each element being an array of the y_values
            return hl.enumerate(sample_ys).filter(
                lambda idx_and_y_values: ht.all_covs_defined[idx_and_y_values[0]] & no_missing(idx_and_y_values[1]) & (hl.is_defined(ht.weight_arrays[idx_and_y_values[0]][group_idx]) if weights else True)
            ).map(lambda idx_and_y_values: idx_and_y_values[0])

        ht = ht.annotate_globals(kept_samples=hl.enumerate(ht.y_arrays_per_group).starmap(get_kept_samples))
        ht = ht.annotate_globals(y_nds=hl.zip(ht.kept_samples, ht.y_arrays_per_group).starmap(
            lambda sample_indices, y_arrays: hl.nd.array(sample_indices.map(lambda idx: y_arrays[idx]))))
        ht = ht.annotate_globals(cov_nds=ht.kept_samples.map(lambda group: hl.nd.array(group.map(lambda idx: ht.cov_arrays[idx]))))

        if weights is None:
            ht = ht.annotate_globals(sqrt_weights=hl.missing(hl.tarray(hl.tndarray(hl.tfloat64, 2))))
            ht = ht.annotate_globals(scaled_y_nds=ht.y_nds)
            ht = ht.annotate_globals(scaled_cov_nds=ht.cov_nds)
        else:
            ht = ht.annotate_globals(weight_nds=hl.enumerate(ht.kept_samples).starmap(
                lambda group_idx, group_sample_indices: hl.nd.array(group_sample_indices.map(lambda group_sample_idx: ht.weight_arrays[group_sample_idx][group_idx]))))
            ht = ht.annotate_globals(sqrt_weights=ht.weight_nds.map(lambda weight_nd: weight_nd.map(lambda e: hl.sqrt(e))))
            ht = ht.annotate_globals(scaled_y_nds=hl.zip(ht.y_nds, ht.sqrt_weights).starmap(lambda y, sqrt_weight: y * sqrt_weight.reshape(-1, 1)))
            ht = ht.annotate_globals(scaled_cov_nds=hl.zip(ht.cov_nds, ht.sqrt_weights).starmap(lambda cov, sqrt_weight: cov * sqrt_weight.reshape(-1, 1)))

        k = builtins.len(covariates)
        ht = ht.annotate_globals(ns=ht.kept_samples.map(lambda one_sample_set: hl.len(one_sample_set)))

        def log_message(i):
            if is_chained:
                return "linear regression_rows[" + hl.str(i) + "] running on " + hl.str(ht.ns[i]) + " samples for " + hl.str(ht.scaled_y_nds[i].shape[1]) + f" response variables y, with input variables x, and {len(covariates)} additional covariates..."
            else:
                return "linear_regression_rows running on " + hl.str(ht.ns[0]) + " samples for " + hl.str(ht.scaled_y_nds[i].shape[1]) + f" response variables y, with input variables x, and {len(covariates)} additional covariates..."

        ht = ht.annotate_globals(ns=hl.range(num_y_lists).map(lambda i: hl._console_log(log_message(i), ht.ns[i])))
        ht = ht.annotate_globals(cov_Qts=hl.if_else(k > 0,
                                 ht.scaled_cov_nds.map(lambda one_cov_nd: hl.nd.qr(one_cov_nd)[0].T),
                                 ht.ns.map(lambda n: hl.nd.zeros((0, n)))))
        ht = ht.annotate_globals(Qtys=hl.zip(ht.cov_Qts, ht.scaled_y_nds).starmap(lambda cov_qt, y: cov_qt @ y))

        return ht.select_globals(
            kept_samples=ht.kept_samples,
            __scaled_y_nds=ht.scaled_y_nds,
            __sqrt_weight_nds=ht.sqrt_weights,
            ns=ht.ns,
            ds=ht.ns.map(lambda n: n - k - 1),
            __cov_Qts=ht.cov_Qts,
            __Qtys=ht.Qtys,
            __yyps=hl.range(num_y_lists).map(lambda i: dot_rows_with_themselves(ht.scaled_y_nds[i].T) - dot_rows_with_themselves(ht.Qtys[i].T)))

    ht = setup_globals(ht)

    def process_block(block):
        rows_in_block = hl.len(block)

        # Processes one block group based on given idx. Returns a single struct.
        def process_y_group(idx):
            if weights is not None:
                X = (hl.nd.array(block[entries_field_name].map(lambda row: mean_impute(select_array_indices(row, ht.kept_samples[idx])))) * ht.__sqrt_weight_nds[idx]).T
            else:
                X = hl.nd.array(block[entries_field_name].map(lambda row: mean_impute(select_array_indices(row, ht.kept_samples[idx])))).T
            n = ht.ns[idx]
            sum_x = X.sum(0)
            Qtx = ht.__cov_Qts[idx] @ X
            ytx = ht.__scaled_y_nds[idx].T @ X
            xyp = ytx - (ht.__Qtys[idx].T @ Qtx)
            xxpRec = (dot_rows_with_themselves(X.T) - dot_rows_with_themselves(Qtx.T)).map(lambda entry: 1 / entry)
            b = xyp * xxpRec
            se = ((1.0 / ht.ds[idx]) * (ht.__yyps[idx].reshape((-1, 1)) @ xxpRec.reshape((1, -1)) - (b * b))).map(lambda entry: hl.sqrt(entry))
            t = b / se
            return hl.rbind(t, lambda t:
                            hl.rbind(ht.ds[idx], lambda d:
                                     hl.rbind(t.map(lambda entry: 2 * hl.expr.functions.pT(-hl.abs(entry), d, True, False)), lambda p:
                                              hl.struct(n=hl.range(rows_in_block).map(lambda i: n), sum_x=sum_x._data_array(),
                                                        y_transpose_x=ytx.T._data_array(), beta=b.T._data_array(),
                                                        standard_error=se.T._data_array(), t_stat=t.T._data_array(),
                                                        p_value=p.T._data_array()))))

        per_y_list = hl.range(num_y_lists).map(lambda i: process_y_group(i))

        key_field_names = [key_field for key_field in ht.key]

        def build_row(row_idx):
            # For every field we care about, map across all y's, getting the row_idxth one from each.
            idxth_keys = {field_name: block[field_name][row_idx] for field_name in key_field_names}
            computed_row_field_names = ['n', 'sum_x', 'y_transpose_x', 'beta', 'standard_error', 't_stat', 'p_value']
            computed_row_fields = {
                field_name: per_y_list.map(lambda one_y: one_y[field_name][row_idx]) for field_name in computed_row_field_names
            }
            pass_through_rows = {
                field_name: block[field_name][row_idx] for field_name in row_field_names
            }

            if not is_chained:
                computed_row_fields = {key: value[0] for key, value in computed_row_fields.items()}

            return hl.struct(**{**idxth_keys, **computed_row_fields, **pass_through_rows})

        new_rows = hl.range(rows_in_block).map(build_row)

        return new_rows

    def process_partition(part):
        grouped = part.grouped(block_size)
        return grouped.flatmap(lambda block: process_block(block)._to_stream())

    res = ht._map_partitions(process_partition)

    if not y_is_list:
        fields = ['y_transpose_x', 'beta', 'standard_error', 't_stat', 'p_value']
        res = res.annotate(**{f: res[f][0] for f in fields})

    res = res.select_globals()

    temp_file_name = hl.utils.new_temp_file("_linear_regression_rows_nd", "result")
    res = res.checkpoint(temp_file_name)

    return res


@typecheck(test=enumeration('wald', 'lrt', 'score', 'firth'),
           y=oneof(expr_float64, sequenceof(expr_float64)),
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           pass_through=sequenceof(oneof(str, Expression)),
           max_iterations=nullable(int),
           tolerance=float)
def logistic_regression_rows(test,
                             y,
                             x,
                             covariates,
                             pass_through=(),
                             *,
                             max_iterations: Optional[int] = None,
                             tolerance: float = 1e-6) -> hail.Table:
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

    Run the logistic regression Wald test per variant using a list of binary (0/1)
    phenotypes, intercept and two covariates stored in column-indexed
    fields:

    >>> result_ht = hl.logistic_regression_rows(
    ...     test='wald',
    ...     y=[dataset.pheno.is_case, dataset.pheno.is_case],  # where pheno values are 0, 1, or missing
    ...     x=dataset.GT.n_alt_alleles(),
    ...     covariates=[1, dataset.pheno.age, dataset.pheno.is_female])

    As above but with at most 100 Newton iterations and a stricter-than-default tolerance of 1e-8:

    >>> result_ht = hl.logistic_regression_rows(
    ...     test='wald',
    ...     y=[dataset.pheno.is_case, dataset.pheno.is_case],  # where pheno values are 0, 1, or missing
    ...     x=dataset.GT.n_alt_alleles(),
    ...     covariates=[1, dataset.pheno.age, dataset.pheno.is_female],
    ...     max_iterations=100,
    ...     tolerance=1e-8)

    Warning
    -------
    :func:`.logistic_regression_rows` considers the same set of
    columns (i.e., samples, points) for every row, namely those columns for
    which **all** response variables and covariates are defined. For each row, missing values of
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

        \mathrm{Prob}(\mathrm{is\_case}) =
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
                                                 reaching the max (by default,
                                                 25 for Wald, LRT; 100 for Firth)
    Wald, LRT, Firth `fit.converged`      bool    ``True`` if iteration converged
    Wald, LRT, Firth `fit.exploded`       bool    ``True`` if iteration exploded
    ================ =================== ======= ===============================

    We consider iteration to have converged when every coordinate of
    :math:`\beta` changes by less than :math:`10^{-6}` by default. For Wald and
    LRT, up to 25 iterations are attempted by default; in testing we find 4 or 5
    iterations nearly always suffice. Convergence may also fail due to
    explosion, which refers to low-level numerical linear algebra exceptions
    caused by manipulating ill-conditioned matrices. Explosion may result from
    (nearly) linearly dependent covariates or complete separation_.

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
    invariant prior <https://en.wikipedia.org/wiki/Jeffreys_prior>`__. This test
    is slower, as both the null and full model must be fit per variant, and
    convergence of the modified Newton method is linear rather than
    quadratic. For Firth, 100 iterations are attempted by default for the null
    model and, if that is successful, for the full model as well. In testing we
    find 20 iterations nearly always suffices. If the null model fails to
    converge, then the `logreg.fit` fields reflect the null model; otherwise,
    they reflect the full model.

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
    y : :class:`.Float64Expression` or :obj:`list` of :class:`.Float64Expression`
        One or more column-indexed response expressions.
        All non-missing values must evaluate to 0 or 1.
        Note that a :class:`.BooleanExpression` will be implicitly converted to
        a :class:`.Float64Expression` with this property.
    x : :class:`.Float64Expression`
        Entry-indexed expression for input variable.
    covariates : :obj:`list` of :class:`.Float64Expression`
        Non-empty list of column-indexed covariate expressions.
    pass_through : :obj:`list` of :class:`str` or :class:`.Expression`
        Additional row fields to include in the resulting table.
    max_iterations : :obj:`int`
        The maximum number of iterations.
    tolerance : :obj:`float`
        Convergence is defined by a change in the beta vector of less than
        `tolerance`.

    Returns
    -------
    :class:`.Table`

    """
    if max_iterations is None:
        max_iterations = 25 if test != 'firth' else 100

    if not isinstance(Env.backend(), SparkBackend):
        return _logistic_regression_rows_nd(
            test, y, x, covariates, pass_through, max_iterations=max_iterations)

    if len(covariates) == 0:
        raise ValueError('logistic regression requires at least one covariate expression')

    mt = matrix_table_source('logistic_regresion_rows/x', x)
    check_entry_indexed('logistic_regresion_rows/x', x)

    y_is_list = isinstance(y, list)
    if y_is_list and len(y) == 0:
        raise ValueError("'logistic_regression_rows': found no values for 'y'")
    y = wrap_to_list(y)

    for e in covariates:
        analyze('logistic_regression_rows/covariates', e, mt._col_indices)

    _warn_if_no_intercept('logistic_regression_rows', covariates)

    x_field_name = Env.get_uid()
    y_field = [f'__y_{i}' for i in range(len(y))]

    y_dict = dict(zip(y_field, y))

    cov_field_names = [f'__cov{i}' for i in range(len(covariates))]
    row_fields = _get_regression_row_fields(mt, pass_through, 'logistic_regression_rows')

    # FIXME: selecting an existing entry field should be emitted as a SelectFields
    mt = mt._select_all(col_exprs=dict(**y_dict,
                                       **dict(zip(cov_field_names, covariates))),
                        row_exprs=row_fields,
                        col_key=[],
                        entry_exprs={x_field_name: x})

    config = {
        'name': 'LogisticRegression',
        'test': test,
        'yFields': y_field,
        'xField': x_field_name,
        'covFields': cov_field_names,
        'passThrough': [x for x in row_fields if x not in mt.row_key],
        'maxIterations': max_iterations,
        'tolerance': tolerance
    }

    result = Table(ir.MatrixToTableApply(mt._mir, config))

    if not y_is_list:
        result = result.transmute(**result.logistic_regression[0])

    return result.persist()


# Helpers for logreg:
def mean_impute(hl_array):
    non_missing_mean = hl.mean(hl_array, filter_missing=True)
    return hl_array.map(lambda entry: hl.coalesce(entry, non_missing_mean))


sigmoid = hl.expit


def nd_max(hl_nd):
    return hl.max(hl.array(hl_nd.reshape(-1)))


def logreg_fit(X: hl.NDArrayNumericExpression, # (K,)
               y: hl.NDArrayNumericExpression, # (N, K)
               null_fit: Optional[hl.StructExpression],
               max_iterations: int,
               tolerance: float
               ) -> hl.StructExpression:
    """Iteratively reweighted least squares to fit the model y ~ Bernoulli(logit(X \beta))

    When fitting the null model, K=n_covariates, otherwise K=n_covariates + 1.
    """
    assert max_iterations >= 0
    assert X.ndim == 2
    assert y.ndim == 1
    # X is samples by covs.
    # y is length num samples, for one cov.
    n = X.shape[0]
    m = X.shape[1]

    if null_fit is None:
        avg = y.sum() / n
        logit_avg = hl.log(avg / (1 - avg))
        b = hl.nd.hstack([hl.nd.array([logit_avg]), hl.nd.zeros((hl.int32(m - 1)))])
        mu = sigmoid(X @ b)
        score = X.T @ (y - mu)
        # Reshape so we do a rowwise multiply
        fisher = X.T @ (X * (mu * (1 - mu)).reshape(-1, 1))
    else:
        # num covs used to fit null model.
        m0 = null_fit.b.shape[0]
        m_diff = m - m0

        X0 = X[:, 0:m0]
        X1 = X[:, m0:]

        b = hl.nd.hstack([null_fit.b, hl.nd.zeros((m_diff,))])
        mu = sigmoid(X @ b)
        score = hl.nd.hstack([null_fit.score, X1.T @ (y - mu)])

        fisher00 = null_fit.fisher
        fisher01 = X0.T @ (X1 * (mu * (1 - mu)).reshape(-1, 1))
        fisher10 = fisher01.T
        fisher11 = X1.T @ (X1 * (mu * (1 - mu)).reshape(-1, 1))

        fisher = hl.nd.vstack([
            hl.nd.hstack([fisher00, fisher01]),
            hl.nd.hstack([fisher10, fisher11])
        ])

    dtype = numerical_regression_fit_dtype
    blank_struct = hl.struct(**{k: hl.missing(dtype[k]) for k in dtype})

    def search(recur, iteration, b, mu, score, fisher):
        def cont(exploded, delta_b, max_delta_b):
            log_lkhd = hl.log((y * mu) + (1 - y) * (1 - mu)).sum()

            next_b = b + delta_b
            next_mu = sigmoid(X @ b)
            next_score = X.T @ (y - mu)
            next_fisher = X.T @ (X * (mu * (1 - mu)).reshape(-1, 1))

            return (hl.case()
                    .when(exploded | hl.is_nan(delta_b[0]),
                          blank_struct.annotate(n_iterations=iteration, log_lkhd=log_lkhd, converged=False, exploded=True))
                    .when(max_delta_b < tolerance,
                          hl.struct(b=b, score=score, fisher=fisher, mu=mu, n_iterations=iteration, log_lkhd=log_lkhd, converged=True, exploded=False))
                    .when(iteration == max_iterations,
                          blank_struct.annotate(n_iterations=iteration, log_lkhd=log_lkhd, converged=False, exploded=False))
                    .default(recur(iteration + 1, next_b, next_mu, next_score, next_fisher)))

        delta_b_struct = hl.nd.solve(fisher, score, no_crash=True)
        exploded = delta_b_struct.failed
        delta_b = delta_b_struct.solution
        max_delta_b = nd_max(hl.abs(delta_b))
        return hl.bind(cont, exploded, delta_b, max_delta_b)

    if max_iterations == 0:
        return blank_struct.annotate(n_iterations=0, log_lkhd=0, converged=False, exploded=False)
    return hl.experimental.loop(search, numerical_regression_fit_dtype, 1, b, mu, score, fisher)


def wald_test(X, fit):
    se = hl.sqrt(hl.nd.diagonal(hl.nd.inv(fit.fisher)))
    z = fit.b / se
    p = z.map(lambda e: 2 * hl.pnorm(-hl.abs(e)))
    return hl.struct(
        beta=fit.b[X.shape[1] - 1],
        standard_error=se[X.shape[1] - 1],
        z_stat=z[X.shape[1] - 1],
        p_value=p[X.shape[1] - 1],
        fit=fit.select('n_iterations', 'converged', 'exploded'))


def lrt_test(X, null_fit, fit):
    chi_sq = hl.if_else(~fit.converged, hl.missing(hl.tfloat64), 2 * (fit.log_lkhd - null_fit.log_lkhd))
    p = hl.pchisqtail(chi_sq, X.shape[1] - null_fit.b.shape[0])

    return hl.struct(
        beta=fit.b[X.shape[1] - 1],
        chi_sq_stat=chi_sq,
        p_value=p,
        fit=fit.select('n_iterations', 'converged', 'exploded'))


def logistic_score_test(X, y, null_fit):
    m = X.shape[1]
    m0 = null_fit.b.shape[0]
    b = hl.nd.hstack([null_fit.b, hl.nd.zeros((hl.int32(m - m0)))])

    X0 = X[:, 0:m0]
    X1 = X[:, m0:]

    mu = hl.expit(X @ b)

    score_0 = null_fit.score
    score_1 = X1.T @ (y - mu)
    score = hl.nd.hstack([score_0, score_1])

    fisher00 = null_fit.fisher
    fisher01 = X0.T @ (X1 * (mu * (1 - mu)).reshape(-1, 1))
    fisher10 = fisher01.T
    fisher11 = X1.T @ (X1 * (mu * (1 - mu)).reshape(-1, 1))

    fisher = hl.nd.vstack([
        hl.nd.hstack([fisher00, fisher01]),
        hl.nd.hstack([fisher10, fisher11])
    ])

    solve_attempt = hl.nd.solve(fisher, score, no_crash=True)

    chi_sq = hl.or_missing(
        ~solve_attempt.failed,
        (score * solve_attempt.solution).sum()
    )

    p = hl.pchisqtail(chi_sq, m - m0)

    return hl.struct(chi_sq_stat=chi_sq, p_value=p)


def _firth_fit(b: hl.NDArrayNumericExpression, # (K,)
               X: hl.NDArrayNumericExpression, # (N, K)
               y: hl.NDArrayNumericExpression, # (N,)
               max_iterations: int,
               tolerance: float
               ) -> hl.StructExpression:
    """Iteratively reweighted least squares using Firth's regression to fit the model y ~ Bernoulli(logit(X \beta))

    When fitting the null model, K=n_covariates, otherwise K=n_covariates + 1.
    """
    assert max_iterations >= 0
    assert X.ndim == 2
    assert y.ndim == 1
    assert b.ndim == 1

    dtype = numerical_regression_fit_dtype._drop_fields(['score', 'fisher'])
    blank_struct = hl.struct(**{k: hl.missing(dtype[k]) for k in dtype})
    X_bslice = X[:, :b.shape[0]]

    def fit(recur, iteration, b):
        def cont(exploded, delta_b, max_delta_b, log_lkhd):
            log_lkhd_left = hl.log(y * mu + (hl.literal(1.0) - y) * (1 - mu)).sum()
            log_lkhd_right = hl.log(hl.abs(hl.nd.diagonal(r))).sum()
            log_lkhd = log_lkhd_left + log_lkhd_right

            next_b = b + delta_b

            return (hl.case()
                    .when(exploded | hl.is_nan(delta_b[0]),
                          blank_struct.annotate(n_iterations=iteration, log_lkhd=log_lkhd, converged=False, exploded=True))
                    .when(max_delta_b < tolerance,
                          hl.struct(b=b, mu=mu, n_iterations=iteration, log_lkhd=log_lkhd, converged=True, exploded=False))
                    .when(iteration == max_iterations,
                          blank_struct.annotate(n_iterations=iteration, log_lkhd=log_lkhd, converged=False, exploded=False))
                    .default(recur(iteration + 1, next_b)))

        m = b.shape[0]  # n_covariates or n_covariates + 1, depending on improved null fit vs full fit
        mu = sigmoid(X_bslice @ b)
        sqrtW = hl.sqrt(mu * (1 - mu))
        q, r = hl.nd.qr(X * sqrtW.T.reshape(-1, 1))
        h = (q * q).sum(1)
        coef = r[:m, :m]
        residual = y - mu
        dep = q[:, :m].T @ ((residual + (h * (0.5 - mu))) / sqrtW)
        delta_b_struct = hl.nd.solve_triangular(coef, dep.reshape(-1, 1), no_crash=True)
        exploded = delta_b_struct.failed
        delta_b = delta_b_struct.solution.reshape(-1)

        max_delta_b = nd_max(hl.abs(delta_b))

        return hl.bind(cont, exploded, delta_b, max_delta_b)

    if max_iterations == 0:
        return blank_struct.annotate(n_iterations=0, log_lkhd=0, converged=False, exploded=False)
    return hl.experimental.loop(fit, dtype, 1, b)


def _firth_test(null_fit, X, y, max_iterations, tolerance) -> hl.StructExpression:
    firth_improved_null_fit = _firth_fit(null_fit.b, X, y, max_iterations=max_iterations, tolerance=tolerance)
    dof = 1  # 1 variant

    def cont(firth_improved_null_fit):
        initial_b_full_model = hl.nd.hstack([firth_improved_null_fit.b, hl.nd.array([0.0])])
        firth_fit = _firth_fit(initial_b_full_model, X, y, max_iterations=max_iterations, tolerance=tolerance)
        def cont2(firth_fit):
            firth_chi_sq = 2 * (firth_fit.log_lkhd - firth_improved_null_fit.log_lkhd)
            firth_p = hl.pchisqtail(firth_chi_sq, dof)

            blank_struct = hl.struct(
                beta=hl.missing(hl.tfloat64),
                chi_sq_stat=hl.missing(hl.tfloat64),
                p_value=hl.missing(hl.tfloat64),
                firth_null_fit=hl.missing(firth_improved_null_fit.dtype),
                fit=hl.missing(firth_fit.dtype)
            )
            return (hl.case()
                    .when(firth_improved_null_fit.converged,
                          hl.case()
                          .when(firth_fit.converged,
                                hl.struct(
                                    beta=firth_fit.b[firth_fit.b.shape[0] - 1],
                                    chi_sq_stat=firth_chi_sq,
                                    p_value=firth_p,
                                    firth_null_fit=firth_improved_null_fit,
                                    fit=firth_fit
                                ))
                          .default(blank_struct.annotate(
                              firth_null_fit=firth_improved_null_fit,
                              fit=firth_fit
                          )))
                    .default(blank_struct.annotate(
                        firth_null_fit=firth_improved_null_fit
                    )))
        return hl.bind(cont2, firth_fit)
    return hl.bind(cont, firth_improved_null_fit)


@typecheck(test=enumeration('wald', 'lrt', 'score', 'firth'),
           y=oneof(expr_float64, sequenceof(expr_float64)),
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           pass_through=sequenceof(oneof(str, Expression)),
           max_iterations=nullable(int),
           tolerance=float)
def _logistic_regression_rows_nd(test,
                                 y,
                                 x,
                                 covariates,
                                 pass_through=(),
                                 *,
                                 max_iterations: Optional[int] = None,
                                 tolerance: float = 1e-6) -> hail.Table:
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

    Run the logistic regression Wald test per variant using a list of binary (0/1)
    phenotypes, intercept and two covariates stored in column-indexed
    fields:

    >>> result_ht = hl.logistic_regression_rows(
    ...     test='wald',
    ...     y=[dataset.pheno.is_case, dataset.pheno.is_case],  # where pheno values are 0, 1, or missing
    ...     x=dataset.GT.n_alt_alleles(),
    ...     covariates=[1, dataset.pheno.age, dataset.pheno.is_female])

    Warning
    -------
    :func:`.logistic_regression_rows` considers the same set of
    columns (i.e., samples, points) for every row, namely those columns for
    which **all** response variables and covariates are defined. For each row, missing values of
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

        \mathrm{Prob}(\mathrm{is\_case}) =
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
    y : :class:`.Float64Expression` or :obj:`list` of :class:`.Float64Expression`
        One or more column-indexed response expressions.
        All non-missing values must evaluate to 0 or 1.
        Note that a :class:`.BooleanExpression` will be implicitly converted to
        a :class:`.Float64Expression` with this property.
    x : :class:`.Float64Expression`
        Entry-indexed expression for input variable.
    covariates : :obj:`list` of :class:`.Float64Expression`
        Non-empty list of column-indexed covariate expressions.
    pass_through : :obj:`list` of :class:`str` or :class:`.Expression`
        Additional row fields to include in the resulting table.

    Returns
    -------
    :class:`.Table`
    """
    if max_iterations is None:
        max_iterations = 25 if test != 'firth' else 100

    if len(covariates) == 0:
        raise ValueError('logistic regression requires at least one covariate expression')

    mt = matrix_table_source('logistic_regresion_rows/x', x)
    check_entry_indexed('logistic_regresion_rows/x', x)

    y_is_list = isinstance(y, list)
    if y_is_list and len(y) == 0:
        raise ValueError("'logistic_regression_rows': found no values for 'y'")
    y = wrap_to_list(y)

    for e in covariates:
        analyze('logistic_regression_rows/covariates', e, mt._col_indices)

    # _warn_if_no_intercept('logistic_regression_rows', covariates)

    x_field_name = Env.get_uid()
    y_field_names = [f'__y_{i}' for i in range(len(y))]

    y_dict = dict(zip(y_field_names, y))

    cov_field_names = [f'__cov{i}' for i in range(len(covariates))]
    row_fields = _get_regression_row_fields(mt, pass_through, 'logistic_regression_rows')

    # Handle filtering columns with missing values:
    mt = mt.filter_cols(hl.array(y + covariates).all(hl.is_defined))

    # FIXME: selecting an existing entry field should be emitted as a SelectFields
    mt = mt._select_all(col_exprs=dict(**y_dict,
                                       **dict(zip(cov_field_names, covariates))),
                        row_exprs=row_fields,
                        col_key=[],
                        entry_exprs={x_field_name: x})

    ht = mt._localize_entries('entries', 'samples')

    # covmat rows are samples, columns are the different covariates
    ht = ht.annotate_globals(covmat=hl.nd.array(ht.samples.map(lambda s: [s[cov_name] for cov_name in cov_field_names])))

    # yvecs is a list of sample-length vectors, one for each dependent variable.
    ht = ht.annotate_globals(yvecs=[hl.nd.array(ht.samples[y_name]) for y_name in y_field_names])

    # Fit null models, which means doing a logreg fit with just the covariates for each phenotype.
    def fit_null(yvec):
        def error_if_not_converged(null_fit):
            return (
                hl.case()
                .when(~null_fit.exploded,
                      (hl.case()
                       .when(null_fit.converged, null_fit)
                       .or_error("Failed to fit logistic regression null model (standard MLE with covariates only): "
                                 "Newton iteration failed to converge")))
                .or_error(hl.format("Failed to fit logistic regression null model (standard MLE with covariates only): "
                                    "exploded at Newton iteration %d", null_fit.n_iterations)))

        null_fit = logreg_fit(ht.covmat, yvec, None, max_iterations=max_iterations, tolerance=tolerance)
        return hl.bind(error_if_not_converged, null_fit)
    ht = ht.annotate_globals(null_fits=ht.yvecs.map(fit_null))

    ht = ht.transmute(x=hl.nd.array(mean_impute(ht.entries[x_field_name])))
    ht = ht.annotate(covs_and_x = hl.nd.hstack([ht.covmat, ht.x.reshape((-1, 1))]))

    def run_test(yvec, null_fit):
        if test == 'score':
            return logistic_score_test(ht.covs_and_x, yvec, null_fit)
        if test == 'firth':
            return _firth_test(null_fit, ht.covs_and_x, yvec, max_iterations=max_iterations, tolerance=tolerance)

        test_fit = logreg_fit(ht.covs_and_x, yvec, null_fit, max_iterations=max_iterations, tolerance=tolerance)
        if test == 'wald':
            return wald_test(ht.covs_and_x, test_fit)
        assert test == 'lrt', test
        return lrt_test(ht.covs_and_x, null_fit, test_fit)
    ht = ht.select(logistic_regression=hl.starmap(run_test, hl.zip(ht.yvecs, ht.null_fits)))

    if not y_is_list:
        ht = ht.select_globals(**ht.null_fits[0])
        return ht.transmute(**ht.logistic_regression[0])
    ht = ht.select_globals('null_fits')
    return ht


@typecheck(test=enumeration('wald', 'lrt', 'score'),
           y=expr_float64,
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           pass_through=sequenceof(oneof(str, Expression)),
           max_iterations=int,
           tolerance=nullable(float))
def poisson_regression_rows(test,
                            y,
                            x,
                            covariates,
                            pass_through=(),
                            *,
                            max_iterations: int = 25,
                            tolerance: Optional[float] = None) -> Table:
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
    pass_through : :obj:`list` of :class:`str` or :class:`.Expression`
        Additional row fields to include in the resulting table.
    tolerance : :obj:`int`, optional
        The iterative fit of this model is considered "converged" if the change in the estimated
        beta is smaller than tolerance. By default the tolerance is 1e-6.

    Returns
    -------
    :class:`.Table`

    """
    if hl.current_backend().requires_lowering:
        return _lowered_poisson_regression_rows(test, y, x, covariates, pass_through, max_iterations=max_iterations, tolerance=tolerance)

    if tolerance is None:
        tolerance = 1e-6

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

    config = {
        'name': 'PoissonRegression',
        'test': test,
        'yField': y_field_name,
        'xField': x_field_name,
        'covFields': cov_field_names,
        'passThrough': [x for x in row_fields if x not in mt.row_key],
        'maxIterations': max_iterations,
        'tolerance': tolerance
    }

    return Table(ir.MatrixToTableApply(mt._mir, config)).persist()


@typecheck(test=enumeration('wald', 'lrt', 'score'),
           y=expr_float64,
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           pass_through=sequenceof(oneof(str, Expression)),
           max_iterations=int,
           tolerance=nullable(float))
def _lowered_poisson_regression_rows(test,
                                     y,
                                     x,
                                     covariates,
                                     pass_through=(),
                                     *,
                                     max_iterations: int = 25,
                                     tolerance: Optional[float] = None):
    assert max_iterations > 0

    if tolerance is None:
        tolerance = 1e-8
    assert tolerance > 0

    k = len(covariates)
    if k == 0:
        raise ValueError('_lowered_poisson_regression_rows: at least one covariate is required.')
    _warn_if_no_intercept('_lowered_poisson_regression_rows', covariates)

    mt = matrix_table_source('_lowered_poisson_regression_rows/x', x)
    check_entry_indexed('_lowered_poisson_regression_rows/x', x)

    row_exprs = _get_regression_row_fields(mt, pass_through, '_lowered_poisson_regression_rows')
    mt = mt._select_all(
        row_exprs=dict(
            pass_through=hl.struct(**row_exprs)
        ),
        col_exprs=dict(
            y=y,
            covariates=covariates
        ),
        entry_exprs=dict(
            x=x
        )
    )
    # FIXME: the order of the columns is irrelevant to regression
    mt = mt.key_cols_by()

    mt = mt.filter_cols(
        hl.all(hl.is_defined(mt.y), *[hl.is_defined(mt.covariates[i]) for i in range(k)])
    )

    mt = mt.annotate_globals(**mt.aggregate_cols(hl.struct(
        yvec=hl.agg.collect(hl.float(mt.y)),
        covmat=hl.agg.collect(mt.covariates.map(hl.float)),
        n=hl.agg.count()
    ), _localize=False))
    mt = mt.annotate_globals(
        yvec=(hl.case()
              .when(mt.n - k - 1 >= 1, hl.nd.array(mt.yvec))
              .or_error(hl.format(
                  "_lowered_poisson_regression_rows: insufficient degrees of freedom: n=%s, k=%s",
                  mt.n, k))),
        covmat=hl.nd.array(mt.covmat),
        n_complete_samples=mt.n
    )
    covmat = mt.covmat
    yvec = mt.yvec
    n = mt.n_complete_samples

    logmean = hl.log(yvec.sum() / n)
    b = hl.nd.array([logmean, *[0 for _ in range(k - 1)]])
    mu = hl.exp(covmat @ b)
    residual = yvec - mu
    score = covmat.T @ residual
    fisher = (mu * covmat.T) @ covmat
    mt = mt.annotate_globals(null_fit=_poisson_fit(covmat, yvec, b, mu, score, fisher, max_iterations, tolerance))
    mt = mt.annotate_globals(
        null_fit=hl.case().when(mt.null_fit.converged, mt.null_fit).or_error(
            hl.format('_lowered_poisson_regression_rows: null model did not converge: %s',
                      mt.null_fit.select('n_iterations', 'log_lkhd', 'converged', 'exploded')))
    )
    mt = mt.annotate_rows(mean_x=hl.agg.mean(mt.x))
    mt = mt.annotate_rows(xvec=hl.nd.array(hl.agg.collect(hl.coalesce(mt.x, mt.mean_x))))
    ht = mt.rows()

    covmat = ht.covmat
    null_fit = ht.null_fit
    # FIXME: we should test a whole block of variants at a time not one-by-one
    xvec = ht.xvec
    yvec = ht.yvec

    if test == 'score':
        chi_sq, p = _poisson_score_test(null_fit, covmat, yvec, xvec)
        return ht.select(
            chi_sq_stat=chi_sq,
            p_value=p,
            **ht.pass_through
        )

    X = hl.nd.hstack([covmat, xvec.T.reshape(-1, 1)])
    b = hl.nd.hstack([null_fit.b, hl.nd.array([0.0])])
    mu = sigmoid(X @ b)
    residual = yvec - mu
    score = hl.nd.hstack([null_fit.score, hl.nd.array([xvec @ residual])])

    fisher00 = null_fit.fisher
    fisher01 = ((covmat.T * mu) @ xvec).reshape((-1, 1))
    fisher10 = fisher01.T
    fisher11 = hl.nd.array([[(mu * xvec.T) @ xvec]])
    fisher = hl.nd.vstack([
        hl.nd.hstack([fisher00, fisher01]),
        hl.nd.hstack([fisher10, fisher11])
    ])

    test_fit = _poisson_fit(X, yvec, b, mu, score, fisher, max_iterations, tolerance)
    if test == 'lrt':
        return ht.select(
            test_fit=test_fit,
            **lrt_test(X, null_fit, test_fit),
            **ht.pass_through
        )
    assert test == 'wald'
    return ht.select(
        test_fit=test_fit,
        **wald_test(X, test_fit),
        **ht.pass_through
    )


def _poisson_fit(X: hl.NDArrayNumericExpression,       # (N, K)
                 y: hl.NDArrayNumericExpression,       # (N,)
                 b: hl.NDArrayNumericExpression,       # (K,)
                 mu: hl.NDArrayNumericExpression,      # (N,)
                 score: hl.NDArrayNumericExpression,   # (K,)
                 fisher: hl.NDArrayNumericExpression,  # (K, K)
                 max_iterations: int,
                 tolerance: float
                 ) -> hl.StructExpression:
    """Iteratively reweighted least squares to fit the model y ~ Poisson(exp(X \beta))

    When fitting the null model, K=n_covariates, otherwise K=n_covariates + 1.
    """
    assert max_iterations >= 0
    assert X.ndim == 2
    assert y.ndim == 1
    assert b.ndim == 1
    assert mu.ndim == 1
    assert score.ndim == 1
    assert fisher.ndim == 2

    dtype = numerical_regression_fit_dtype
    blank_struct = hl.struct(**{k: hl.missing(dtype[k]) for k in dtype})

    def fit(recur, iteration, b, mu, score, fisher):
        def cont(exploded, delta_b, max_delta_b):
            log_lkhd = y @ hl.log(mu) - mu.sum()

            next_b = b + delta_b
            next_mu = hl.exp(X @ next_b)
            next_score = X.T @ (y - next_mu)
            next_fisher = (next_mu * X.T) @ X

            return (hl.case()
                    .when(exploded | hl.is_nan(delta_b[0]),
                          blank_struct.annotate(n_iterations=iteration, log_lkhd=log_lkhd, converged=False, exploded=True))
                    .when(max_delta_b < tolerance,
                          hl.struct(b=b, score=score, fisher=fisher, mu=mu, n_iterations=iteration, log_lkhd=log_lkhd, converged=True, exploded=False))
                    .when(iteration == max_iterations,
                          blank_struct.annotate(n_iterations=iteration, log_lkhd=log_lkhd, converged=False, exploded=False))
                    .default(recur(iteration + 1, next_b, next_mu, next_score, next_fisher)))

        delta_b_struct = hl.nd.solve(fisher, score, no_crash=True)

        exploded = delta_b_struct.failed
        delta_b = delta_b_struct.solution
        max_delta_b = nd_max(delta_b.map(lambda e: hl.abs(e)))
        return hl.bind(cont, exploded, delta_b, max_delta_b)

    if max_iterations == 0:
        return blank_struct.select(n_iterations=0, log_lkhd=0, converged=False, exploded=False)
    return hl.experimental.loop(fit, dtype, 1, b, mu, score, fisher)


def _poisson_score_test(null_fit, covmat, y, xvec):
    dof = 1

    X = hl.nd.hstack([covmat, xvec.T.reshape(-1, 1)])
    b = hl.nd.hstack([null_fit.b, hl.nd.array([0.0])])
    mu = hl.exp(X @ b)
    score = hl.nd.hstack([null_fit.score, hl.nd.array([xvec @ (y - mu)])])

    fisher00 = null_fit.fisher
    fisher01 = ((mu * covmat.T) @ xvec).reshape((-1, 1))
    fisher10 = fisher01.T
    fisher11 = hl.nd.array([[(mu * xvec.T) @ xvec]])
    fisher = hl.nd.vstack([
        hl.nd.hstack([fisher00, fisher01]),
        hl.nd.hstack([fisher10, fisher11])
    ])

    fisher_div_score = hl.nd.solve(fisher, score, no_crash=True)
    chi_sq = hl.or_missing(~fisher_div_score.failed,
                           score @ fisher_div_score.solution)
    p = hl.pchisqtail(chi_sq, dof)
    return chi_sq, p


def linear_mixed_model(y,
                       x,
                       z_t=None,
                       k=None,
                       p_path=None,
                       overwrite=False,
                       standardize=True,
                       mean_impute=True):
    r"""Initialize a linear mixed model from a matrix table.

    .. warning::

        This functionality is no longer implemented/supported as of Hail 0.2.94.
    """
    raise NotImplementedError("linear_mixed_model is no longer implemented/supported as of Hail 0.2.94")


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

    .. warning::

        This functionality is no longer implemented/supported as of Hail 0.2.94.
    """
    raise NotImplementedError("linear_mixed_model is no longer implemented/supported as of Hail 0.2.94")


@typecheck(group=expr_any,
           weight=expr_float64,
           y=expr_float64,
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           max_size=int,
           accuracy=numeric,
           iterations=int)
def _linear_skat(group,
                 weight,
                 y,
                 x,
                 covariates,
                 max_size: int = 46340,
                 accuracy: float = 1e-6,
                 iterations: int = 10000):
    r'''The linear sequence kernel association test (SKAT).

    Linear SKAT tests if the phenotype, `y`, is significantly associated with the genotype, `x`. For
    :math:`N` samples, in a group of :math:`M` variants, with :math:`K` covariates, the model is
    given by:

    .. math::

        \begin{align*}
        X &: R^{N \times K} \quad\quad \textrm{covariates} \\
        G &: \{0, 1, 2\}^{N \times M} \textrm{genotypes} \\
        \\
        \varepsilon &\sim N(0, \sigma^2) \\
        y &= \beta_0 X + \beta_1 G + \varepsilon
        \end{align*}

    The usual null hypothesis is :math:`\beta_1 = 0`. SKAT tests for an association, but does not
    provide an effect size or other information about the association.

    Wu et al. argue that, under the null hypothesis, a particular value, :math:`Q`, is distributed
    according to a generalized chi-squared distribution with parameters determined by the genotypes,
    weights, and residual phenotypes. The SKAT p-value is the probability of drawing even larger
    values of :math:`Q`. :math:`Q` is defined by Wu et al. as:

    .. math::

        \begin{align*}
        r &= y - \widehat{\beta_\textrm{null}} X \\
        W_{ii} &= w_i \\
        \\
        Q &= r^T G W G^T r
        \end{align*}

    :math:`\widehat{\beta_\textrm{null}}` is the best-fit beta under the null model:

    .. math::

        y = \beta_\textrm{null} X + \varepsilon \quad\quad \varepsilon \sim N(0, \sigma^2)

    Therefore :math:`r`, the residual phenotype, is the portion of the phenotype unexplained by the
    covariates alone. Also notice:

    1. The residual phenotypes are normally distributed with mean zero and variance
       :math:`\sigma^2`.

    2. :math:`G W G^T`, is a symmetric positive-definite matrix when the weights are non-negative.

    We can transform the residuals into standard normal variables by normalizing by their
    variance. Note that the variance is corrected for the degrees of freedom in the null model:

    .. math::

        \begin{align*}
        \widehat{\sigma} &= \frac{1}{N - K} r^T r \\
        h &= \frac{1}{\widehat{\sigma}} r \\
        h &\sim N(0, 1) \\
        r &= h \widehat{\sigma}
        \end{align*}

    We can rewrite :math:`Q` in terms of a Grammian matrix and these new standard normal random variables:

    .. math::

        \begin{align*}
        Q &= h^T \widehat{\sigma} G W G^T \widehat{\sigma} h \\
        A &= \widehat{\sigma} G W^{1/2} \\
        B &= A A^T \\
        \\
        Q &= h^T B h \\
        \end{align*}

    This expression is a `"quadratic form" <https://en.wikipedia.org/wiki/Quadratic_form>`__ of the
    vector :math:`h`. Because :math:`B` is a real symmetric matrix, we can eigendecompose it into an
    orthogonal matrix and a diagonal matrix of eigenvalues:

    .. math::

        \begin{align*}
        U \Lambda U^T &= B \quad\quad \Lambda \textrm{ diagonal } U \textrm{ orthogonal} \\
        Q &= h^T U \Lambda U^T h
        \end{align*}

    An orthogonal matrix transforms a vector of i.i.d. standard normal variables into a new vector
    of different i.i.d standard normal variables, so we can interpret :math:`Q` as a weighted sum of
    i.i.d. standard normal variables:

    .. math::

        \begin{align*}
        \tilde{h} &= U^T h \\
        Q &= \sum_s \Lambda_{ss} \tilde{h}_s^2
        \end{align*}

    The distribution of such sums (indeed, any quadratic form of i.i.d. standard normal variables)
    is governed by the generalized chi-squared distribution (the CDF is available in Hail as
    :func:`.pgenchisq`):

    .. math::

        \begin{align*}
        \lambda_i &= \Lambda_{ii} \\
        Q &\sim \mathrm{GeneralizedChiSquared}(\lambda, \vec{1}, \vec{0}, 0, 0)
        \end{align*}

    Therefore, we can test the null hypothesis by calculating the probability of receiving values
    larger than :math:`Q`. If that probability is very small, then the residual phenotypes are
    likely not i.i.d. normal variables with variance :math:`\widehat{\sigma}^2`.

    The SKAT method was originally described in:

        Wu MC, Lee S, Cai T, Li Y, Boehnke M, Lin X. *Rare-variant association testing for
        sequencing data with the sequence kernel association test.* Am J Hum Genet. 2011 Jul
        15;89(1):82-93. doi: 10.1016/j.ajhg.2011.05.029. Epub 2011 Jul 7. PMID: 21737059; PMCID:
        PMC3135811. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3135811/

    Examples
    --------

    Generate a dataset with a phenotype noisily computed from the genotypes:

    >>> hl.reset_global_randomness()
    >>> mt = hl.balding_nichols_model(1, n_samples=100, n_variants=20)
    >>> mt = mt.annotate_rows(gene = mt.locus.position // 12)
    >>> mt = mt.annotate_rows(weight = 1)
    >>> mt = mt.annotate_cols(phenotype = hl.agg.sum(mt.GT.n_alt_alleles()) - 20 + hl.rand_norm(0, 1))

    Test if the phenotype is significantly associated with the genotype:

    >>> skat = hl._linear_skat(
    ...     mt.gene,
    ...     mt.weight,
    ...     mt.phenotype,
    ...     mt.GT.n_alt_alleles(),
    ...     covariates=[1.0])
    >>> skat.show()
    +-------+-------+----------+----------+-------+
    | group |  size |   q_stat |  p_value | fault |
    +-------+-------+----------+----------+-------+
    | int32 | int64 |  float64 |  float64 | int32 |
    +-------+-------+----------+----------+-------+
    |     0 |    11 | 8.76e+02 | 1.23e-05 |     0 |
    |     1 |     9 | 8.13e+02 | 3.95e-05 |     0 |
    +-------+-------+----------+----------+-------+

    The same test, but using the original paper's suggested weights which are derived from the
    allele frequency.

    >>> mt = hl.variant_qc(mt)
    >>> skat = hl._linear_skat(
    ...     mt.gene,
    ...     hl.dbeta(mt.variant_qc.AF[0], 1, 25),
    ...     mt.phenotype,
    ...     mt.GT.n_alt_alleles(),
    ...     covariates=[1.0])
    >>> skat.show()
    +-------+-------+----------+----------+-------+
    | group |  size |   q_stat |  p_value | fault |
    +-------+-------+----------+----------+-------+
    | int32 | int64 |  float64 |  float64 | int32 |
    +-------+-------+----------+----------+-------+
    |     0 |    11 | 2.39e+01 | 4.32e-01 |     0 |
    |     1 |     9 | 1.69e+01 | 7.82e-02 |     0 |
    +-------+-------+----------+----------+-------+

    Our simulated data was unweighted, so the null hypothesis appears true. In real datasets, we
    expect the allele frequency to correlate with effect size.

    Notice that, in the second group, the fault flag is set to 1. This indicates that the numerical
    integration to calculate the p-value failed to achieve the required accuracy (by default,
    1e-6). In this particular case, the null hypothesis is likely true and the numerical integration
    returned a (nonsensical) value greater than one.

    The `max_size` parameter allows us to skip large genes that would cause "out of memory" errors:

    >>> skat = hl._linear_skat(
    ...     mt.gene,
    ...     mt.weight,
    ...     mt.phenotype,
    ...     mt.GT.n_alt_alleles(),
    ...     covariates=[1.0],
    ...     max_size=10)
    >>> skat.show()
    +-------+-------+----------+----------+-------+
    | group |  size |   q_stat |  p_value | fault |
    +-------+-------+----------+----------+-------+
    | int32 | int64 |  float64 |  float64 | int32 |
    +-------+-------+----------+----------+-------+
    |     0 |    11 |       NA |       NA |    NA |
    |     1 |     9 | 8.13e+02 | 3.95e-05 |     0 |
    +-------+-------+----------+----------+-------+

    Notes
    -----

    In the SKAT R package, the "weights" are actually the *square root* of the weight expression
    from the paper. This method uses the definition from the paper.

    The paper includes an explicit intercept term but this method expects the user to specify the
    intercept as an extra covariate with the value 1.

    This method does not perform small sample size correction.

    The `q_stat` return value is *not* the :math:`Q` statistic from the paper. We match the output
    of the SKAT R package which returns :math:`\tilde{Q}`:

    .. math::

        \tilde{Q} = \frac{Q}{2 \widehat{\sigma}^2}

    Parameters
    ----------
    group : :class:`.Expression`
        Row-indexed expression indicating to which group a variant belongs. This is typically a gene
        name or an interval.
    weight : :class:`.Float64Expression`
        Row-indexed expression for weights. Must be non-negative.
    y : :class:`.Float64Expression`
        Column-indexed response (dependent variable) expression.
    x : :class:`.Float64Expression`
        Entry-indexed expression for input (independent variable).
    covariates : :obj:`list` of :class:`.Float64Expression`
        List of column-indexed covariate expressions. You must explicitly provide an intercept term
        if desired. You must provide at least one covariate.
    max_size : :obj:`int`
        Maximum size of group on which to run the test. Groups which exceed this size will have a
        missing p-value and missing q statistic. Defaults to 46340.
    accuracy : :obj:`float`
        The accuracy of the p-value if fault value is zero. Defaults to 1e-6.
    iterations : :obj:`int`
        The maximum number of iterations used to calculate the p-value (which has no closed
        form). Defaults to 1e5.

    Returns
    -------
    :class:`.Table`
        One row per-group. The key is `group`. The row fields are:

        - group : the `group` parameter.

        - size : :obj:`.tint64`, the number of variants in this group.

        - q_stat : :obj:`.tfloat64`, the :math:`Q` statistic, see Notes for why this differs from the paper.

        - p_value : :obj:`.tfloat64`, the test p-value for the null hypothesis that the genotypes
          have no linear influence on the phenotypes.

        - fault : :obj:`.tint32`, the fault flag from :func:`.pgenchisq`.

        The global fields are:

        - n_complete_samples : :obj:`.tint32`, the number of samples with neither a missing
          phenotype nor a missing covariate.

        - y_residual : :obj:`.tint32`, the residual phenotype from the null model. This may be
          interpreted as the component of the phenotype not explained by the covariates alone.

        - s2 : :obj:`.tfloat64`, the variance of the residuals, :math:`\sigma^2` in the paper.

    '''
    mt = matrix_table_source('skat/x', x)
    k = len(covariates)
    if k == 0:
        raise ValueError('_linear_skat: at least one covariate is required.')
    _warn_if_no_intercept('_linear_skat', covariates)
    mt = mt._select_all(
        row_exprs=dict(
            group=group,
            weight=weight
        ),
        col_exprs=dict(
            y=y,
            covariates=covariates
        ),
        entry_exprs=dict(
            x=x
        )
    )
    mt = mt.filter_cols(
        hl.all(hl.is_defined(mt.y), *[hl.is_defined(mt.covariates[i]) for i in range(k)])
    )
    yvec, covmat, n = mt.aggregate_cols((
        hl.agg.collect(hl.float(mt.y)),
        hl.agg.collect(mt.covariates.map(hl.float)),
        hl.agg.count()
    ), _localize=False)
    mt = mt.annotate_globals(
        yvec=hl.nd.array(yvec),
        covmat=hl.nd.array(covmat),
        n_complete_samples=n
    )
    # Instead of finding the best-fit beta, we go directly to the best-predicted value using the
    # reduced QR decomposition:
    #
    #     Q @ R = X
    #     y = X beta
    #     X^T y = X^T X beta
    #     (X^T X)^-1 X^T y = beta
    #     (R^T Q^T Q R)^-1 R^T Q^T y = beta
    #     (R^T R)^-1 R^T Q^T y = beta
    #     R^-1 R^T^-1 R^T Q^T y = beta
    #     R^-1 Q^T y = beta
    #
    #     X beta = X R^-1 Q^T y
    #            = Q R R^-1 Q^T y
    #            = Q Q^T y
    #
    covmat_Q, _ = hl.nd.qr(mt.covmat)
    mt = mt.annotate_globals(
        covmat_Q=covmat_Q
    )
    null_mu = mt.covmat_Q @ (mt.covmat_Q.T @ mt.yvec)
    y_residual = mt.yvec - null_mu
    mt = mt.annotate_globals(
        y_residual=y_residual,
        s2=y_residual @ y_residual.T / (n - k)
    )
    mt = mt.annotate_rows(
        G_row_mean=hl.agg.mean(mt.x)
    )
    mt = mt.annotate_rows(
        G_row=hl.agg.collect(hl.coalesce(mt.x, mt.G_row_mean))
    )
    ht = mt.rows()
    ht = ht.filter(hl.all(hl.is_defined(ht.group), hl.is_defined(ht.weight)))
    ht = ht.group_by(
        'group'
    ).aggregate(
        weight_take=hl.agg.take(ht.weight, n=max_size + 1),
        G_take=hl.agg.take(ht.G_row, n=max_size + 1),
        size=hl.agg.count()
    )
    ht = ht.annotate(
        weight=hl.nd.array(hl.or_missing(hl.len(ht.weight_take) <= max_size, ht.weight_take)),
        G=hl.nd.array(hl.or_missing(hl.len(ht.G_take) <= max_size, ht.G_take)).T
    )
    ht = ht.annotate(
        Q=((ht.y_residual @ ht.G).map(lambda x: x**2) * ht.weight).sum(0)
    )

    # Null model:
    #
    #     y = X b + e,    e ~ N(0, \sigma^2)
    #
    # We can find a best-fit b, bhat, and a best-fit y, yhat:
    #
    #     bhat = (X.T X).inv X.T y
    #
    #     Q R = X                     (reduced QR decomposition)
    #     bhat = R.inv Q.T y
    #
    #     yhat = X bhat
    #          = Q R R.inv Q.T y
    #          = Q Q.T y
    #
    # The residual phenotype not captured by the covariates alone is r:
    #
    #     r = y - yhat
    #       = (I - Q Q.T) y
    #
    # We can factor the Q-statistic (note there are two Qs: the Q from the QR decomposition and the
    # Q-statistic from the paper):
    #
    #     Q = r.T G diag(w) G.T r
    #     Z = r.T G diag(sqrt(w))
    #     Q = Z Z.T
    #
    # Plugging in our expresion for r:
    #
    #     Z = y.T (I - Q Q.T) G diag(sqrt(w))
    #
    # Notice that I - Q Q.T is symmetric (ergo X = X.T) because each summand is symmetric and sums
    # of symmetric matrices are symmetric matrices.
    #
    # We have asserted that
    #
    #     y ~ N(0, \sigma^2)
    #
    # It will soon be apparent that the distribution of Q is easier to characterize if our random
    # variables are standard normals:
    #
    #     h ~ N(0, 1)
    #     y = \sigma h
    #
    # We set \sigma^2 to the sample variance of the residual vectors.
    #
    # Returning to Z:
    #
    #     Z = h.T \sigma (I - Q Q.T) G diag(sqrt(w))
    #     Q = Z Z.T
    #
    # Which we can factor into a symmetric matrix and a standard normal:
    #
    #     A = \sigma (I - Q Q.T) G diag(sqrt(w))
    #     B = A A.T
    #     Q = h.T B h
    #
    # This is called a "quadratic form". It is a weighted sum of products of pairs of entries of h,
    # which we have asserted are i.i.d. standard normal variables. The distribution of such sums is
    # given by the generalized chi-squared distribution:
    #
    #     U L U.T = B                    B is symmetric and thus has an eigendecomposition
    #     h.T B h = Q ~ GeneralizedChiSquare(L, 1, 0, 0, 0)
    #
    # The orthogonal matrix U remixes the vector of i.i.d. normal variables into a new vector of
    # different i.i.d. normal variables. The L matrix is diagonal and scales each squared normal
    # variable.
    #
    # Since B = A A.T is symmetric, its eigenvalues are the square of the singular values of A or
    # A.T:
    #
    #     W S V = A
    #     U L U.T = B
    #             = A A.T
    #             = W S V V.T S W
    #             = W S S W           V is orthogonal so V V.T = I
    #             = W S^2 W

    weights_arr = hl.array(ht.weight)
    A = hl.case().when(
        hl.all(weights_arr.map(lambda x: x >= 0)),
        (ht.G - ht.covmat_Q @ (ht.covmat_Q.T @ ht.G)) * hl.sqrt(ht.weight)
    ).or_error(hl.format('hl._linear_skat: every weight must be positive, in group %s, the weights were: %s',
                         ht.group, weights_arr))
    singular_values = hl.nd.svd(A, compute_uv=False)

    # SVD(M) = U S V. U and V are unitary, therefore SVD(k M) = U (k S) V.
    eigenvalues = ht.s2 * singular_values.map(lambda x: x**2)

    # The R implementation of SKAT, Function.R, Get_Lambda_Approx filters the eigenvalues,
    # presumably because a good estimate of the Generalized Chi-Sqaured CDF is not significantly
    # affected by chi-squared components with very tiny weights.
    threshold = 1e-5 * eigenvalues.sum() / eigenvalues.shape[0]
    w = hl.array(eigenvalues).filter(lambda y: y >= threshold)
    genchisq_data = hl.pgenchisq(
        ht.Q,
        w=w,
        k=hl.nd.ones(hl.len(w), dtype=hl.tint32),
        lam=hl.nd.zeros(hl.len(w)),
        mu=0,
        sigma=0,
        min_accuracy=accuracy,
        max_iterations=iterations
    )
    ht = ht.select(
        'size',
        # for reasons unknown, the R implementation calls this expression the Q statistic (which is
        # *not* what they write in the paper)
        q_stat=ht.Q / 2 / ht.s2,
        # The reasoning for taking the complement of the CDF value is:
        #
        # 1. Q is a measure of variance and thus positive.
        #
        # 2. We want to know the probability of obtaining a variance even larger ("more extreme")
        #
        # Ergo, we want to check the right-tail of the distribution.
        p_value=1.0 - genchisq_data.value,
        fault=genchisq_data.fault
    )
    return ht.select_globals('y_residual', 's2', 'n_complete_samples')


@typecheck(group=expr_any,
           weight=expr_float64,
           y=expr_float64,
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           max_size=int,
           null_max_iterations=int,
           null_tolerance=float,
           accuracy=numeric,
           iterations=int)
def _logistic_skat(group,
                   weight,
                   y,
                   x,
                   covariates,
                   max_size: int = 46340,
                   null_max_iterations: int = 25,
                   null_tolerance: float = 1e-6,
                   accuracy: float = 1e-6,
                   iterations: int = 10000):
    r'''The logistic sequence kernel association test (SKAT).

    Logistic SKAT tests if the phenotype, `y`, is significantly associated with the genotype,
    `x`. For :math:`N` samples, in a group of :math:`M` variants, with :math:`K` covariates, the
    model is given by:

    .. math::

        \begin{align*}
        X &: R^{N \times K} \\
        G &: \{0, 1, 2\}^{N \times M} \\
        \\
        Y &\sim \textrm{Bernoulli}(\textrm{logit}^{-1}(\beta_0 X + \beta_1 G))
        \end{align*}

    The usual null hypothesis is :math:`\beta_1 = 0`. SKAT tests for an association, but does not
    provide an effect size or other information about the association.

    Wu et al. argue that, under the null hypothesis, a particular value, :math:`Q`, is distributed
    according to a generalized chi-squared distribution with parameters determined by the genotypes,
    weights, and residual phenotypes. The SKAT p-value is the probability of drawing even larger
    values of :math:`Q`. If :math:`\widehat{\beta_\textrm{null}}` is the best-fit beta under the
    null model:

    .. math::

        Y \sim \textrm{Bernoulli}(\textrm{logit}^{-1}(\beta_\textrm{null} X))

    Then :math:`Q` is defined by Wu et al. as:

    .. math::

        \begin{align*}
        p_i &= \textrm{logit}^{-1}(\widehat{\beta_\textrm{null}} X) \\
        r_i &= y_i - p_i \\
        W_{ii} &= w_i \\
        \\
        Q &= r^T G W G^T r
        \end{align*}

    Therefore :math:`r_i`, the residual phenotype, is the portion of the phenotype unexplained by
    the covariates alone. Also notice:

    1. Each sample's phenotype is Bernoulli distributed with mean :math:`p_i` and variance
       :math:`\sigma^2_i = p_i(1 - p_i)`, the binomial variance.

    2. :math:`G W G^T`, is a symmetric positive-definite matrix when the weights are non-negative.

    We describe below our interpretation of the mathematics as described in the main body and
    appendix of Wu, et al. According to the paper, the distribution of :math:`Q` is given by a
    generalized chi-squared distribution whose weights are the eigenvalues of a symmetric matrix
    which we call :math:`Z Z^T`:

    .. math::

        \begin{align*}
        V_{ii} &= \sigma^2_i \\
        W_{ii} &= w_i \quad\quad \textrm{the weight for variant } i \\
        \\
        P_0 &= V - V X (X^T V X)^{-1} X^T V \\
        Z Z^T &= P_0^{1/2} G W G^T P_0^{1/2}
        \end{align*}

    The eigenvalues of :math:`Z Z^T` and :math:`Z^T Z` are the squared singular values of :math:`Z`;
    therefore, we instead focus on :math:`Z^T Z`. In the expressions below, we elide transpositions
    of symmetric matrices:

    .. math::

        \begin{align*}
        Z Z^T &= P_0^{1/2} G W G^T P_0^{1/2} \\
        Z &= P_0^{1/2} G W^{1/2} \\
        Z^T Z &= W^{1/2} G^T P_0 G W^{1/2}
        \end{align*}

    Before substituting the definition of :math:`P_0`, simplify it using the reduced QR
    decomposition:

    .. math::

        \begin{align*}
        Q R &= V^{1/2} X \\
        R^T Q^T &= X^T V^{1/2} \\
        \\
        P_0 &= V - V X (X^T V X)^{-1} X^T V \\
            &= V - V X (R^T Q^T Q R)^{-1} X^T V \\
            &= V - V X (R^T R)^{-1} X^T V \\
            &= V - V X R^{-1} (R^T)^{-1} X^T V \\
            &= V - V^{1/2} Q (R^T)^{-1} X^T V^{1/2} \\
            &= V - V^{1/2} Q Q^T V^{1/2} \\
            &= V^{1/2} (I - Q Q^T) V^{1/2} \\
        \end{align*}

    Substitute this simplified expression into :math:`Z`:

    .. math::

        \begin{align*}
        Z^T Z &= W^{1/2} G^T V^{1/2} (I - Q Q^T) V^{1/2} G W^{1/2} \\
        \end{align*}

    Split this symmetric matrix by observing that :math:`I - Q Q^T` is idempotent:

    .. math::

        \begin{align*}
        I - Q Q^T &= (I - Q Q^T)(I - Q Q^T)^T \\
        \\
        Z &= (I - Q Q^T) V^{1/2} G W^{1/2} \\
        Z &= (G - Q Q^T G) V^{1/2} W^{1/2}
        \end{align*}

    Finally, the squared singular values of :math:`Z` are the eigenvalues of :math:`Z^T Z`, so
    :math:`Q` should be distributed as follows:

    .. math::

        \begin{align*}
        U S V^T &= Z \quad\quad \textrm{the singular value decomposition} \\
        \lambda_s &= S_{ss}^2 \\
        \\
        Q &\sim \textrm{GeneralizedChiSquared}(\lambda, \vec{1}, \vec{0}, 0, 0)
        \end{align*}

    The null hypothesis test tests for the probability of observing even larger values of :math:`Q`.

    The SKAT method was originally described in:

        Wu MC, Lee S, Cai T, Li Y, Boehnke M, Lin X. *Rare-variant association testing for
        sequencing data with the sequence kernel association test.* Am J Hum Genet. 2011 Jul
        15;89(1):82-93. doi: 10.1016/j.ajhg.2011.05.029. Epub 2011 Jul 7. PMID: 21737059; PMCID:
        PMC3135811. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3135811/

    Examples
    --------

    Generate a dataset with a phenotype noisily computed from the genotypes:

    >>> hl.reset_global_randomness()
    >>> mt = hl.balding_nichols_model(1, n_samples=100, n_variants=20)
    >>> mt = mt.annotate_rows(gene = mt.locus.position // 12)
    >>> mt = mt.annotate_rows(weight = 1)
    >>> mt = mt.annotate_cols(phenotype = (hl.agg.sum(mt.GT.n_alt_alleles()) - 20 + hl.rand_norm(0, 1)) > 0.5)

    Test if the phenotype is significantly associated with the genotype:

    >>> skat = hl._logistic_skat(
    ...     mt.gene,
    ...     mt.weight,
    ...     mt.phenotype,
    ...     mt.GT.n_alt_alleles(),
    ...     covariates=[1.0])
    >>> skat.show()
    +-------+-------+----------+----------+-------+
    | group |  size |   q_stat |  p_value | fault |
    +-------+-------+----------+----------+-------+
    | int32 | int64 |  float64 |  float64 | int32 |
    +-------+-------+----------+----------+-------+
    |     0 |    11 | 1.78e+02 | 1.68e-04 |     0 |
    |     1 |     9 | 1.39e+02 | 1.82e-03 |     0 |
    +-------+-------+----------+----------+-------+

    The same test, but using the original paper's suggested weights which are derived from the
    allele frequency.

    >>> mt = hl.variant_qc(mt)
    >>> skat = hl._logistic_skat(
    ...     mt.gene,
    ...     hl.dbeta(mt.variant_qc.AF[0], 1, 25),
    ...     mt.phenotype,
    ...     mt.GT.n_alt_alleles(),
    ...     covariates=[1.0])
    >>> skat.show()
    +-------+-------+----------+----------+-------+
    | group |  size |   q_stat |  p_value | fault |
    +-------+-------+----------+----------+-------+
    | int32 | int64 |  float64 |  float64 | int32 |
    +-------+-------+----------+----------+-------+
    |     0 |    11 | 8.04e+00 | 3.50e-01 |     0 |
    |     1 |     9 | 1.22e+00 | 5.04e-01 |     0 |
    +-------+-------+----------+----------+-------+

    Our simulated data was unweighted, so the null hypothesis appears true. In real datasets, we
    expect the allele frequency to correlate with effect size.

    Notice that, in the second group, the fault flag is set to 1. This indicates that the numerical
    integration to calculate the p-value failed to achieve the required accuracy (by default,
    1e-6). In this particular case, the null hypothesis is likely true and the numerical integration
    returned a (nonsensical) value greater than one.

    The `max_size` parameter allows us to skip large genes that would cause "out of memory" errors:

    >>> skat = hl._logistic_skat(
    ...     mt.gene,
    ...     mt.weight,
    ...     mt.phenotype,
    ...     mt.GT.n_alt_alleles(),
    ...     covariates=[1.0],
    ...     max_size=10)
    >>> skat.show()
    +-------+-------+----------+----------+-------+
    | group |  size |   q_stat |  p_value | fault |
    +-------+-------+----------+----------+-------+
    | int32 | int64 |  float64 |  float64 | int32 |
    +-------+-------+----------+----------+-------+
    |     0 |    11 |       NA |       NA |    NA |
    |     1 |     9 | 1.39e+02 | 1.82e-03 |     0 |
    +-------+-------+----------+----------+-------+

    Notes
    -----

    In the SKAT R package, the "weights" are actually the *square root* of the weight expression
    from the paper. This method uses the definition from the paper.

    The paper includes an explicit intercept term but this method expects the user to specify the
    intercept as an extra covariate with the value 1.

    This method does not perform small sample size correction.

    The `q_stat` return value is *not* the :math:`Q` statistic from the paper. We match the output
    of the SKAT R package which returns :math:`\tilde{Q}`:

    .. math::

        \tilde{Q} = \frac{Q}{2}

    Parameters
    ----------
    group : :class:`.Expression`
        Row-indexed expression indicating to which group a variant belongs. This is typically a gene
        name or an interval.
    weight : :class:`.Float64Expression`
        Row-indexed expression for weights. Must be non-negative.
    y : :class:`.Float64Expression`
        Column-indexed response (dependent variable) expression.
    x : :class:`.Float64Expression`
        Entry-indexed expression for input (independent variable).
    covariates : :obj:`list` of :class:`.Float64Expression`
        List of column-indexed covariate expressions. You must explicitly provide an intercept term
        if desired. You must provide at least one covariate.
    max_size : :obj:`int`
        Maximum size of group on which to run the test. Groups which exceed this size will have a
        missing p-value and missing q statistic. Defaults to 46340.
    null_max_iterations : :obj:`int`
        The maximum number of iterations when fitting the logistic null model. Defaults to 25.
    null_tolerance : :obj:`float`
        The null model logisitic regression converges when the errors is less than this. Defaults to
        1e-6.
    accuracy : :obj:`float`
        The accuracy of the p-value if fault value is zero. Defaults to 1e-6.
    iterations : :obj:`int`
        The maximum number of iterations used to calculate the p-value (which has no closed
        form). Defaults to 1e5.

    Returns
    -------
    :class:`.Table`
        One row per-group. The key is `group`. The row fields are:

        - group : the `group` parameter.

        - size : :obj:`.tint64`, the number of variants in this group.

        - q_stat : :obj:`.tfloat64`, the :math:`Q` statistic, see Notes for why this differs from the paper.

        - p_value : :obj:`.tfloat64`, the test p-value for the null hypothesis that the genotypes
          have no linear influence on the phenotypes.

        - fault : :obj:`.tint32`, the fault flag from :func:`.pgenchisq`.

        The global fields are:

        - n_complete_samples : :obj:`.tint32`, the number of samples with neither a missing
          phenotype nor a missing covariate.

        - y_residual : :obj:`.tint32`, the residual phenotype from the null model. This may be
          interpreted as the component of the phenotype not explained by the covariates alone.

        - s2 : :obj:`.tfloat64`, the variance of the residuals, :math:`\sigma^2` in the paper.

        - null_fit:

          - b : :obj:`.tndarray` vector of coefficients.

          - score : :obj:`.tndarray` vector of score statistics.

          - fisher : :obj:`.tndarray` matrix of fisher statistics.

          - mu : :obj:`.tndarray` the expected value under the null model.

          - n_iterations : :obj:`.tint32` the number of iterations before termination.

          - log_lkhd : :obj:`.tfloat64` the log-likelihood of the final iteration.

          - converged : :obj:`.tbool` True if the null model converged.

          - exploded : :obj:`.tbool` True if the null model failed to converge due to numerical
            explosion.

    '''
    mt = matrix_table_source('skat/x', x)
    k = len(covariates)
    if k == 0:
        raise ValueError('_logistic_skat: at least one covariate is required.')
    _warn_if_no_intercept('_logistic_skat', covariates)
    mt = mt._select_all(
        row_exprs=dict(
            group=group,
            weight=weight
        ),
        col_exprs=dict(
            y=y,
            covariates=covariates
        ),
        entry_exprs=dict(
            x=x
        )
    )
    mt = mt.filter_cols(
        hl.all(hl.is_defined(mt.y), *[hl.is_defined(mt.covariates[i]) for i in range(k)])
    )
    if mt.y.dtype != hl.tbool:
        mt = mt.annotate_cols(
            y=(hl.case()
               .when(hl.any(mt.y == 0, mt.y == 1), hl.bool(mt.y))
               .or_error(hl.format(
                   f'hl._logistic_skat: phenotypes must either be True, False, 0, or 1, found: %s of type {mt.y.dtype}', mt.y)))
        )
    yvec, covmat, n = mt.aggregate_cols((
        hl.agg.collect(hl.float(mt.y)),
        hl.agg.collect(mt.covariates.map(hl.float)),
        hl.agg.count()
    ), _localize=False)
    mt = mt.annotate_globals(
        yvec=hl.nd.array(yvec),
        covmat=hl.nd.array(covmat),
        n_complete_samples=n
    )
    null_fit = logreg_fit(mt.covmat, mt.yvec, None, max_iterations=null_max_iterations, tolerance=null_tolerance)
    mt = mt.annotate_globals(
        null_fit=hl.case().when(null_fit.converged, null_fit).or_error(
            hl.format('hl._logistic_skat: null model did not converge: %s', null_fit))
    )
    null_mu = mt.null_fit.mu
    y_residual = mt.yvec - null_mu
    mt = mt.annotate_globals(
        y_residual=y_residual,
        s2=null_mu * (1 - null_mu)
    )
    mt = mt.annotate_rows(
        G_row_mean=hl.agg.mean(mt.x)
    )
    mt = mt.annotate_rows(
        G_row=hl.agg.collect(hl.coalesce(mt.x, mt.G_row_mean))
    )
    ht = mt.rows()
    ht = ht.filter(hl.all(hl.is_defined(ht.group), hl.is_defined(ht.weight)))
    ht = ht.group_by(
        'group'
    ).aggregate(
        weight_take=hl.agg.take(ht.weight, n=max_size + 1),
        G_take=hl.agg.take(ht.G_row, n=max_size + 1),
        size=hl.agg.count()
    )
    ht = ht.annotate(
        weight=hl.nd.array(hl.or_missing(hl.len(ht.weight_take) <= max_size, ht.weight_take)),
        G=hl.nd.array(hl.or_missing(hl.len(ht.G_take) <= max_size, ht.G_take)).T
    )
    ht = ht.annotate(
        # Q=ht.y_residual @ (ht.G * ht.weight) @ ht.G.T @ ht.y_residual.T
        Q=((ht.y_residual @ ht.G).map(lambda x: x**2) * ht.weight).sum(0)
    )

    # See linear SKAT code comment for an extensive description of the mathematics here.

    sqrtv = hl.sqrt(ht.s2)
    Q, _ = hl.nd.qr(ht.covmat * sqrtv.reshape(-1, 1))
    weights_arr = hl.array(ht.weight)
    G_scaled = ht.G * sqrtv.reshape(-1, 1)
    A = hl.case().when(
        hl.all(weights_arr.map(lambda x: x >= 0)),
        (G_scaled - Q @ (Q.T @ G_scaled)) * hl.sqrt(ht.weight)
    ).or_error(hl.format('hl._logistic_skat: every weight must be positive, in group %s, the weights were: %s',
                         ht.group, weights_arr))
    singular_values = hl.nd.svd(A, compute_uv=False)
    eigenvalues = singular_values.map(lambda x: x**2)

    # The R implementation of SKAT, Function.R, Get_Lambda_Approx filters the eigenvalues,
    # presumably because a good estimate of the Generalized Chi-Sqaured CDF is not significantly
    # affected by chi-squared components with very tiny weights.
    threshold = 1e-5 * eigenvalues.sum() / eigenvalues.shape[0]
    w = hl.array(eigenvalues).filter(lambda y: y >= threshold)
    genchisq_data = hl.pgenchisq(
        ht.Q,
        w=w,
        k=hl.nd.ones(hl.len(w), dtype=hl.tint32),
        lam=hl.nd.zeros(hl.len(w)),
        mu=0,
        sigma=0,
        min_accuracy=accuracy,
        max_iterations=iterations
    )
    ht = ht.select(
        'size',
        # for reasons unknown, the R implementation calls this expression the Q statistic (which is
        # *not* what they write in the paper)
        q_stat=ht.Q / 2,
        # The reasoning for taking the complement of the CDF value is:
        #
        # 1. Q is a measure of variance and thus positive.
        #
        # 2. We want to know the probability of obtaining a variance even larger ("more extreme")
        #
        # Ergo, we want to check the right-tail of the distribution.
        p_value=1.0 - genchisq_data.value,
        fault=genchisq_data.fault
    )
    return ht.select_globals('y_residual', 's2', 'n_complete_samples', 'null_fit')


@typecheck(key_expr=expr_any,
           weight_expr=expr_float64,
           y=expr_float64,
           x=expr_float64,
           covariates=sequenceof(expr_float64),
           logistic=oneof(bool, sized_tupleof(nullable(int), nullable(float))),
           max_size=int,
           accuracy=numeric,
           iterations=int)
def skat(key_expr,
         weight_expr,
         y,
         x,
         covariates,
         logistic: Union[bool, Tuple[int, float]] = False,
         max_size: int = 46340,
         accuracy: float = 1e-6,
         iterations: int = 10000) -> Table:
    r"""Test each keyed group of rows for association by linear or logistic
    SKAT test.

    Examples
    --------

    Test each gene for association using the linear sequence kernel association
    test:

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
    logistic : :obj:`bool` or :obj:`tuple` of :obj:`int` and :obj:`float`
        If false, use the linear test. If true, use the logistic test with no
        more than 25 logistic iterations and a convergence tolerance of 1e-6. If
        a tuple is given, use the logistic test with the tuple elements as the
        maximum nubmer of iterations and convergence tolerance, respectively.
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
    if hl.current_backend().requires_lowering:
        if logistic:
            kwargs = {
                'accuracy': accuracy,
                'iterations': iterations
            }
            if logistic is not True:
                null_max_iterations, null_tolerance = logistic
                kwargs['null_max_iterations'] = null_max_iterations
                kwargs['null_tolerance'] = null_tolerance
            ht = hl._logistic_skat(key_expr, weight_expr, y, x, covariates, max_size, **kwargs)
        else:
            ht = hl._linear_skat(key_expr, weight_expr, y, x, covariates, max_size, accuracy, iterations)
        ht = ht.select_globals()
        return ht
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

    mt = mt._select_all(col_exprs=dict(**{y_field_name: y},
                                       **dict(zip(cov_field_names, covariates))),
                        row_exprs={weight_field_name: weight_expr,
                                   key_field_name: key_expr},
                        entry_exprs=entry_expr)

    if logistic is True:
        use_logistic = True
        max_iterations = 25
        tolerance = 1e-6
    elif logistic is False:
        use_logistic = False
        max_iterations = 0
        tolerance = 0.0
    else:
        assert isinstance(logistic, tuple) and len(logistic) == 2
        use_logistic = True
        max_iterations, tolerance = logistic

    config = {
        'name': 'Skat',
        'keyField': key_field_name,
        'weightField': weight_field_name,
        'xField': x_field_name,
        'yField': y_field_name,
        'covFields': cov_field_names,
        'logistic': use_logistic,
        'maxSize': max_size,
        'accuracy': accuracy,
        'iterations': iterations,
        'logistic_max_iterations': max_iterations,
        'logistic_tolerance': tolerance
    }

    return Table(ir.MatrixToTableApply(mt._mir, config)).persist()


@typecheck(p_value=expr_numeric,
           approximate=bool)
def lambda_gc(p_value, approximate=True):
    """
    Compute genomic inflation factor (lambda GC) from an Expression of p-values.

    .. include:: ../_templates/experimental.rst

    Parameters
    ----------
    p_value : :class:`.NumericExpression`
        Row-indexed numeric expression of p-values.
    approximate : :obj:`bool`
        If False, computes exact lambda GC (slower and uses more memory).

    Returns
    -------
    :obj:`float`
        Genomic inflation factor (lambda genomic control).
    """
    check_row_indexed('lambda_gc', p_value)
    t = table_source('lambda_gc', p_value)
    med_chisq = _lambda_gc_agg(p_value, approximate)
    return t.aggregate(med_chisq)


@typecheck(p_value=expr_numeric,
           approximate=bool)
def _lambda_gc_agg(p_value, approximate=True):
    chisq = hl.qchisqtail(p_value, 1)
    if approximate:
        med_chisq = hl.agg.filter(~hl.is_nan(p_value), hl.agg.approx_quantiles(chisq, 0.5))
    else:
        med_chisq = hl.agg.filter(~hl.is_nan(p_value), hl.median(hl.agg.collect(chisq)))
    return med_chisq / hl.qchisqtail(0.5, 1)


@typecheck(ds=oneof(Table, MatrixTable),
           keep_star=bool,
           left_aligned=bool,
           permit_shuffle=bool)
def split_multi(ds, keep_star=False, left_aligned=False, *, permit_shuffle=False):
    """Split multiallelic variants.

    Warning
    -------
    In order to support a wide variety of data types, this function splits only
    the variants on a :class:`.MatrixTable`, but **not the genotypes**. Use
    :func:`.split_multi_hts` if possible, or split the genotypes yourself using
    one of the entry modification methods: :meth:`.MatrixTable.annotate_entries`,
    :meth:`.MatrixTable.select_entries`, :meth:`.MatrixTable.transmute_entries`.

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

    Warning
    -------
    This method assumes `ds` contains at most one non-split variant per locus. This assumption permits the
    most efficient implementation of the splitting algorithm. If your queries involving `split_multi`
    crash with errors about out-of-order keys, this assumption may be violated. Otherwise, this
    warning likely does not apply to your dataset.

    If each locus in `ds` contains one multiallelic variant and one or more biallelic variants, you
    can filter to the multiallelic variants, split those, and then combine the split variants with
    the original biallelic variants.

    For example, the following code splits a dataset `mt` which contains a mixture of split and
    non-split variants.

    >>> bi = mt.filter_rows(hl.len(mt.alleles) == 2)
    >>> bi = bi.annotate_rows(a_index=1, was_split=False, old_locus=bi.locus, old_alleles=bi.alleles)
    >>> multi = mt.filter_rows(hl.len(mt.alleles) > 2)
    >>> split = hl.split_multi(multi)
    >>> mt = split.union_rows(bi)

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
    permit_shuffle : :obj:`bool`
        If ``True``, permit a data shuffle to sort out-of-order split results.
        This will only be required if input data has duplicate loci, one of
        which contains more than one alternate allele.

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
                return MatrixTable(ir.MatrixKeyRowsBy(mt._mir, ['locus', 'alleles'], is_sorted=True))
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
                return Table(ir.TableKeyBy(ht._tir, ['locus', 'alleles'], is_sorted=True))

    if left_aligned:
        def make_struct(i):
            def error_on_moved(v):
                return (hl.case()
                        .when(v.locus == old_row.locus, new_struct(v, i))
                        .or_error("Found non-left-aligned variant in split_multi"))
            return hl.bind(error_on_moved,
                           hl.min_rep(old_row.locus, [old_row.alleles[0], old_row.alleles[i]]))
        return split_rows(hl.sorted(kept_alleles.map(make_struct)), permit_shuffle)
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

        left = split_rows(make_array(lambda locus: locus == ds['locus']), permit_shuffle)
        moved = split_rows(make_array(lambda locus: locus != ds['locus']), True)
    return left.union(moved) if is_table else left.union_rows(moved, _check_cols=False)


@typecheck(ds=oneof(Table, MatrixTable),
           keep_star=bool,
           left_aligned=bool,
           vep_root=str,
           permit_shuffle=bool)
def split_multi_hts(ds, keep_star=False, left_aligned=False, vep_root='vep', *, permit_shuffle=False):
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

    >>> hl.split_multi_hts(dataset).write('output/split.mt')

    Warning
    -------
    This method assumes `ds` contains at most one non-split variant per locus. This assumption permits the
    most efficient implementation of the splitting algorithm. If your queries involving `split_multi_hts`
    crash with errors about out-of-order keys, this assumption may be violated. Otherwise, this
    warning likely does not apply to your dataset.

    If each locus in `ds` contains one multiallelic variant and one or more biallelic variants, you
    can filter to the multiallelic variants, split those, and then combine the split variants with
    the original biallelic variants.

    For example, the following code splits a dataset `mt` which contains a mixture of split and
    non-split variants.

    >>> bi = mt.filter_rows(hl.len(mt.alleles) == 2)
    >>> bi = bi.annotate_rows(a_index=1, was_split=False)
    >>> multi = mt.filter_rows(hl.len(mt.alleles) > 2)
    >>> split = hl.split_multi_hts(multi)
    >>> mt = split.union_rows(bi)

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

    `GQ` is recomputed from `PL` if `PL` is provided and is not
    missing. If not, it is copied from the original GQ.

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
    >>> split_ds = split_ds.annotate_rows(info = split_ds.info.annotate(AC = split_ds.info.AC[split_ds.a_index - 1]))
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
    vep_root : :class:`str`
        Top-level location of vep data. All variable-length VEP fields
        (intergenic_consequences, motif_feature_consequences,
        regulatory_feature_consequences, and transcript_consequences)
        will be split properly (i.e. a_index corresponding to the VEP allele_num).
    permit_shuffle : :obj:`bool`
        If ``True``, permit a data shuffle to sort out-of-order split results.
        This will only be required if input data has duplicate loci, one of
        which contains more than one alternate allele.

    Returns
    -------
    :class:`.MatrixTable` or :class:`.Table`
        A biallelic variant dataset.

    """

    split = split_multi(ds, keep_star=keep_star, left_aligned=left_aligned, permit_shuffle=permit_shuffle)

    row_fields = set(ds.row)
    update_rows_expression = {}
    if vep_root in row_fields:
        update_rows_expression[vep_root] = split[vep_root].annotate(**{
            x: split[vep_root][x].filter(lambda csq: csq.allele_num == split.a_index)
            for x in ('intergenic_consequences', 'motif_feature_consequences',
                      'regulatory_feature_consequences', 'transcript_consequences')})

    if isinstance(ds, Table):
        return split.annotate(**update_rows_expression).drop('old_locus', 'old_alleles')

    split = split.annotate_rows(**update_rows_expression)
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
                                                                      split.a_index).unphased_diploid_gt_index() == i
                                                ).map(lambda j: split.PL[j]))))))
        if 'GQ' in entry_fields:
            update_entries_expression['PL'] = pl
            update_entries_expression['GQ'] = hl.or_else(hl.gq_from_pl(pl), split.GQ)
        else:
            update_entries_expression['PL'] = pl
    else:
        if 'GQ' in entry_fields:
            update_entries_expression['GQ'] = split.GQ

    if 'PGT' in entry_fields:
        update_entries_expression['PGT'] = hl.downcode(split.PGT, split.a_index)
    if 'PID' in entry_fields:
        update_entries_expression['PID'] = split.PID
    return split.annotate_entries(**update_entries_expression).drop('old_locus', 'old_alleles')


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

    This method drops variants with :math:`p_j = 0` or :math:`p_j = 1` before
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

    mt = mt.select_entries(__gt=call_expr.n_alt_alleles()).unfilter_entries()
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

    mt = mt.select_entries(__gt=call_expr.n_alt_alleles()).unfilter_entries()
    mt = mt.select_rows(__AC=agg.sum(mt.__gt),
                        __ACsq=agg.sum(mt.__gt * mt.__gt),
                        __n_called=agg.count_where(hl.is_defined(mt.__gt)))
    mt = mt.select_rows(__mean_gt=mt.__AC / mt.__n_called,
                        __centered_length=hl.sqrt(mt.__ACsq - (mt.__AC ** 2) / mt.__n_called))
    fmt = mt.filter_rows(mt.__centered_length > 0.1)  # truly non-zero values are at least sqrt(0.5)

    normalized_gt = hl.or_else((fmt.__gt - fmt.__mean_gt) / fmt.__centered_length, 0.0)

    try:
        bm = BlockMatrix.from_entry_expr(normalized_gt)
        return (bm.T @ bm) / (bm.n_rows / bm.n_cols)
    except FatalError as fe:
        raise FatalError("Could not convert MatrixTable to BlockMatrix. It's possible all variants were dropped by variance filter.\n"
                         "Check that the input MatrixTable has at least two samples in it:  mt.count_cols().") from fe


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
    ...         {'v': '1:3:C:G', 's': 'd', 'GT': hl.missing(hl.tcall)}]
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
    see :meth:`~hail.MatrixTable.add_row_index`).

    The correlation of two vectors is defined as the
    `Pearson correlation coeffecient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__
    between the corresponding empirical distributions of elements,
    or equivalently as the cosine of the angle between the vectors.

    This method has two stages:

    - writing the row-normalized block matrix to a temporary file on persistent
      disk with :meth:`.BlockMatrix.from_entry_expr`. The parallelism is
      ``n_rows / block_size``.

    - reading and multiplying this block matrix by its transpose. The
      parallelism is ``(n_rows / block_size)^2`` if all blocks are computed.

    Warning
    -------
    See all warnings on :meth:`.BlockMatrix.from_entry_expr`. In particular,
    for large matrices, it may be preferable to run the two stages separately,
    saving the row-normalized block matrix to a file on external storage with
    :meth:`.BlockMatrix.write_from_entry_expr`.

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
    ...         {'v': '2:1:C:G',       'cm': 0.2, 's': 'd', 'GT': hl.missing(hl.tcall)}]
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
    (see :meth:`~hail.MatrixTable.add_row_index`). Each variant is regarded as a vector of
    elements defined by `entry_expr`, typically the number of alternate alleles
    or genotype dosage. Missing values are mean-imputed within variant.

    The method produces a symmetric block-sparse matrix supported in a
    neighborhood of the diagonal. If variants :math:`i` and :math:`j` are on the
    same contig and within `radius` base pairs (inclusive) then the
    :math:`(i, j)` element is their
    `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__.
    Otherwise, the :math:`(i, j)` element is ``0.0``.

    Rows with a constant value (i.e., zero variance) will result in ``nan``
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
    starts_and_stops = hl.linalg.utils.locus_windows(locus_expr, radius, coord_expr, _localize=False)
    starts_and_stops = hl.tuple([starts_and_stops[0].map(lambda i: hl.int64(i)), starts_and_stops[1].map(lambda i: hl.int64(i))])
    ld = hl.row_correlation(entry_expr, block_size)
    return ld._sparsify_row_intervals_expr(starts_and_stops, blocks_only=False)


@typecheck(n_populations=int,
           n_samples=int,
           n_variants=int,
           n_partitions=nullable(int),
           pop_dist=nullable(sequenceof(numeric)),
           fst=nullable(sequenceof(numeric)),
           af_dist=nullable(expr_any),
           reference_genome=reference_genome_type,
           mixture=bool,
           phased=bool)
def balding_nichols_model(n_populations: int,
                          n_samples: int,
                          n_variants: int,
                          n_partitions: Optional[int] = None,
                          pop_dist: Optional[List[int]] = None,
                          fst: Optional[List[Union[float, int]]] = None,
                          af_dist: Optional[hl.Expression] = None,
                          reference_genome: str = 'default',
                          mixture: bool = False,
                          *,
                          phased: bool = False
                          ) -> MatrixTable:
    r"""Generate a matrix table of variants, samples, and genotypes using the
    Balding-Nichols or Pritchard-Stephens-Donnelly model.

    Examples
    --------
    Generate a matrix table of genotypes with 1000 variants and 100 samples
    across 3 populations:

    >>> hl.reset_global_randomness()
    >>> bn_ds = hl.balding_nichols_model(3, 100, 1000)
    >>> bn_ds.show(n_rows=5, n_cols=5)
    +---------------+------------+------+------+------+------+------+
    | locus         | alleles    | 0.GT | 1.GT | 2.GT | 3.GT | 4.GT |
    +---------------+------------+------+------+------+------+------+
    | locus<GRCh37> | array<str> | call | call | call | call | call |
    +---------------+------------+------+------+------+------+------+
    | 1:1           | ["A","C"]  | 0/1  | 0/0  | 0/1  | 0/0  | 0/0  |
    | 1:2           | ["A","C"]  | 1/1  | 1/1  | 1/1  | 1/1  | 0/1  |
    | 1:3           | ["A","C"]  | 0/1  | 0/1  | 1/1  | 0/1  | 1/1  |
    | 1:4           | ["A","C"]  | 0/1  | 0/0  | 0/1  | 0/0  | 0/1  |
    | 1:5           | ["A","C"]  | 0/1  | 0/1  | 0/1  | 0/0  | 0/0  |
    +---------------+------------+------+------+------+------+------+
    showing top 5 rows
    showing the first 5 of 100 columns

    Generate a dataset as above but with phased genotypes:

    >>> hl.reset_global_randomness()
    >>> bn_ds = hl.balding_nichols_model(3, 100, 1000, phased=True)
    >>> bn_ds.show(n_rows=5, n_cols=5)
    +---------------+------------+------+------+------+------+------+
    | locus         | alleles    | 0.GT | 1.GT | 2.GT | 3.GT | 4.GT |
    +---------------+------------+------+------+------+------+------+
    | locus<GRCh37> | array<str> | call | call | call | call | call |
    +---------------+------------+------+------+------+------+------+
    | 1:1           | ["A","C"]  | 0|0  | 0|0  | 0|0  | 0|0  | 1|0  |
    | 1:2           | ["A","C"]  | 1|1  | 1|1  | 1|1  | 1|1  | 1|1  |
    | 1:3           | ["A","C"]  | 1|1  | 1|1  | 0|1  | 1|1  | 1|1  |
    | 1:4           | ["A","C"]  | 0|0  | 1|0  | 0|0  | 1|0  | 0|0  |
    | 1:5           | ["A","C"]  | 0|0  | 0|1  | 0|0  | 0|0  | 0|0  |
    +---------------+------------+------+------+------+------+------+
    showing top 5 rows
    showing the first 5 of 100 columns

    Generate a matrix table using 4 populations, 40 samples, 150 variants, 3
    partitions, population distribution ``[0.1, 0.2, 0.3, 0.4]``,
    :math:`F_{ST}` values ``[.02, .06, .04, .12]``, ancestral allele
    frequencies drawn from a truncated beta distribution with ``a = 0.01`` and
    ``b = 0.05`` over the interval ``[0.05, 1]``, and random seed 1:

    >>> hl.reset_global_randomness()
    >>> bn_ds = hl.balding_nichols_model(4, 40, 150, 3,
    ...          pop_dist=[0.1, 0.2, 0.3, 0.4],
    ...          fst=[.02, .06, .04, .12],
    ...          af_dist=hl.rand_beta(a=0.01, b=2.0, lower=0.05, upper=1.0))

    To guarantee reproducibility, we set the Hail global seed with
    :func:`.set_global_seed` immediately prior to generating the dataset.

    Notes
    -----
    This method simulates a matrix table of variants, samples, and genotypes
    using the Balding-Nichols model, which we now define.

    - :math:`K` populations are labeled by integers :math:`0, 1, \dots, K - 1`.
    - :math:`N` samples are labeled by strings :math:`0, 1, \dots, N - 1`.
    - :math:`M` variants are defined as ``1:1:A:C``, ``1:2:A:C``, ...,
      ``1:M:A:C``.
    - The default distribution for population assignment :math:`\pi` is uniform.
    - The default ancestral frequency distribution :math:`P_0` is uniform on
      :math:`[0.1, 0.9]`.
      All three classes are located in ``hail.stats``.
    - The default :math:`F_{ST}` values are all :math:`0.1`.

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
        \begin{aligned}
            k_n \,&\sim\, \pi \\
            p_m \,&\sim\, P_0 \\
            p_{k,m} \mid p_m\,&\sim\, \mathrm{Beta}(\mu = p_m,\, \sigma^2 = F_k p_m (1 - p_m)) \\
            g_{n,m} \mid k_n, p_{k, m} \,&\sim\, \mathrm{Binomial}(2, p_{k_n, m})
        \end{aligned}

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

    For the `Pritchard-Stephens-Donnelly model <http://www.genetics.org/content/155/2/945.long>`__,
    set the `mixture` to true to treat `pop_dist` as the parameters of the
    Dirichlet distribution describing admixture between the modern populations.
    In this case, the type of `pop` is :class:`.tarray` of
    :py:data:`.tfloat64` and the value is the mixture proportions.

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
        `n_populations` with non-negative values.
        Default is ``[1, ..., 1]``.
    fst : :obj:`list` of :obj:`float`, optional
        :math:`F_{ST}` values, a list of length `n_populations` with values
        in (0, 1). Default is ``[0.1, ..., 0.1]``.
    af_dist : :class:`.Float64Expression`, optional
        Representing a random function.  Ancestral allele frequency
        distribution.  Default is :func:`.rand_unif` over the range
        `[0.1, 0.9]` with seed 0.
    reference_genome : :class:`str` or :class:`.ReferenceGenome`
        Reference genome to use.
    mixture : :obj:`bool`
        Treat `pop_dist` as the parameters of a Dirichlet distribution,
        as in the Prichard-Stevens-Donnelly model.
    phased : :obj:`bool`
        Generate phased genotypes.

    Returns
    -------
    :class:`.MatrixTable`
        Simulated matrix table of variants, samples, and genotypes.

    """
    if pop_dist is None:
        pop_dist = [1 for _ in range(n_populations)]

    if fst is None:
        fst = [0.1 for _ in range(n_populations)]

    if af_dist is None:
        af_dist = hl.rand_unif(0.1, 0.9, seed=0)

    if n_partitions is None:
        n_partitions = max(8, int(n_samples * n_variants / (128 * 1024 * 1024)))

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
        raise ExpressionException('balding_nichols_model expects af_dist to '
                                  + 'have scalar arguments: found expression '
                                  + 'from source {}'
                                  .format(af_dist._indices.source))

    if af_dist.dtype != tfloat64:
        raise ValueError("af_dist must be a hail function with return type tfloat64.")

    info("balding_nichols_model: generating genotypes for {} populations, {} samples, and {} variants..."
         .format(n_populations, n_samples, n_variants))

    # generate matrix table
    from numpy import linspace
    n_partitions = min(n_partitions, n_variants)
    start_idxs = [int(x) for x in linspace(0, n_variants, n_partitions + 1)]
    idx_bounds = list(zip(start_idxs, start_idxs[1:]))

    pop_f = hl.rand_dirichlet if mixture else hl.rand_cat

    bn = hl.Table._generate(
        contexts=idx_bounds,
        globals=hl.struct(
            bn=hl.struct(
                n_populations=n_populations,
                n_samples=n_samples,
                n_variants=n_variants,
                n_partitions=n_partitions,
                pop_dist=pop_dist,
                fst=fst,
                mixture=mixture
            ),
            cols=hl.range(n_samples).map(
                lambda idx: hl.struct(sample_idx=idx, pop=pop_f(pop_dist))
            )
        ),
        partitions=[
            hl.Interval(**{
                endpoint: hl.Struct(
                    locus=reference_genome.locus_from_global_position(idx),
                    alleles=['A', 'C']
                ) for endpoint, idx in [('start', lo), ('end', hi)]
            })
            for (lo, hi) in idx_bounds
        ],
        rowfn=lambda idx_range, _: hl.range(idx_range[0], idx_range[1]).map(
            lambda idx: hl.bind(
                lambda ancestral: hl.struct(
                    locus=hl.locus_from_global_position(idx, reference_genome),
                    alleles=['A', 'C'],
                    ancestral_af=ancestral,
                    af=hl.array([(1 - x) / x for x in fst]).map(
                        lambda x: hl.rand_beta(ancestral * x, (1 - ancestral) * x)
                    ),
                    entries=hl.repeat(hl.struct(), n_samples),
                ),
                af_dist
            )
        )
    )

    bn = bn._unlocalize_entries('entries', 'cols', ['sample_idx'])

    # entry info
    p = hl.sum(bn.pop * bn.af) if mixture else bn.af[bn.pop]
    q = 1 - p

    if phased:
        mom = hl.rand_bool(p)
        dad = hl.rand_bool(p)
        return bn.select_entries(GT=hl.call(mom, dad, phased=True))

    idx = hl.rand_cat([q ** 2, 2 * p * q, p ** 2])
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
     - `old_to_new` (``array<int32>``) -- An array that maps old allele index to
       new allele index. Its length is the same as `old_alleles`. Alleles that
       are filtered are missing.
     - `new_to_old` (``array<int32>``) -- An array that maps new allele index to
       the old allele index. Its length is the same as the modified `alleles`
       field.

    If all alternate alleles of a variant are filtered out, the variant itself
    is filtered out.

    **Using** `f`

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
        Function from (allele: :class:`.StringExpression`, allele_index:
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
    new_to_old = (hl.enumerate(mt.__allele_inclusion)
                  .filter(lambda elt: elt[1])
                  .map(lambda elt: elt[0]))
    old_to_new_dict = (hl.dict(hl.enumerate(hl.enumerate(mt.alleles)
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
    return left.union_rows(right, _check_cols=False).drop('__allele_inclusion', '__new_locus', '__new_alleles')


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
    For usage of the `f` argument, see the :func:`.filter_alleles`
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
     - `old_to_new` (``array<int32>``) -- An array that maps old allele index to
       new allele index. Its length is the same as `old_alleles`. Alleles that
       are filtered are missing.
     - `new_to_old` (``array<int32>``) -- An array that maps new allele index to
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
        Function from (allele: :class:`.StringExpression`, allele_index:
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
                             mt.entry.dtype, hl.hts_entry_schema))

    mt = filter_alleles(mt, f)

    if subset:
        newPL = hl.if_else(
            hl.is_defined(mt.PL),
            hl.bind(
                lambda unnorm: unnorm - hl.min(unnorm),
                hl.range(0, hl.triangle(mt.alleles.length())).map(
                    lambda newi: hl.bind(
                        lambda newc: mt.PL[hl.call(mt.new_to_old[newc[0]],
                                                   mt.new_to_old[newc[1]]).unphased_diploid_gt_index()],
                        hl.unphased_diploid_gt_index_call(newi)))),
            hl.missing(tarray(tint32)))
        return mt.annotate_entries(
            GT=hl.unphased_diploid_gt_index_call(hl.argmin(newPL, unique=True)),
            AD=hl.if_else(
                hl.is_defined(mt.AD),
                hl.range(0, mt.alleles.length()).map(
                    lambda newi: mt.AD[mt.new_to_old[newi]]),
                hl.missing(tarray(tint32))),
            # DP unchanged
            GQ=hl.gq_from_pl(newPL),
            PL=newPL)
    # otherwise downcode
    else:
        mt = mt.annotate_rows(__old_to_new_no_na=mt.old_to_new.map(lambda x: hl.or_else(x, 0)))
        newPL = hl.if_else(
            hl.is_defined(mt.PL),
            (hl.range(0, hl.triangle(hl.len(mt.alleles)))
             .map(lambda newi: hl.min(hl.range(0, hl.triangle(hl.len(mt.old_alleles)))
                                      .filter(lambda oldi: hl.bind(
                                          lambda oldc: hl.call(mt.__old_to_new_no_na[oldc[0]],
                                                               mt.__old_to_new_no_na[oldc[1]]) == hl.unphased_diploid_gt_index_call(newi),
                                          hl.unphased_diploid_gt_index_call(oldi)))
                                      .map(lambda oldi: mt.PL[oldi])))),
            hl.missing(tarray(tint32)))
        return mt.annotate_entries(
            GT=hl.call(mt.__old_to_new_no_na[mt.GT[0]],
                       mt.__old_to_new_no_na[mt.GT[1]]),
            AD=hl.if_else(
                hl.is_defined(mt.AD),
                (hl.range(0, hl.len(mt.alleles))
                 .map(lambda newi: hl.sum(hl.range(0, hl.len(mt.old_alleles))
                                          .filter(lambda oldi: mt.__old_to_new_no_na[oldi] == newi)
                                          .map(lambda oldi: mt.AD[oldi])))),
                hl.missing(tarray(tint32))),
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

    return Table(ir.MatrixToTableApply(mt._mir, {
        'name': 'LocalLDPrune',
        'callField': call_field,
        'r2Threshold': float(r2),
        'windowSize': bp_window_size,
        'maxQueueSize': max_queue_size
    })).persist()


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

    mt = mt.annotate_rows(info=locally_pruned_table[mt.row_key])
    mt = mt.filter_rows(hl.is_defined(mt.info)).unfilter_entries()

    std_gt_bm = BlockMatrix.from_entry_expr(
        hl.or_else(
            (mt[field].n_alt_alleles() - mt.info.mean) * mt.info.centered_length_rec,
            0.0),
        block_size=block_size)
    r2_bm = (std_gt_bm @ std_gt_bm.T) ** 2

    _, stops = hl.linalg.utils.locus_windows(locally_pruned_table.locus, bp_window_size)

    entries = r2_bm.sparsify_row_intervals(range(stops.size), stops, blocks_only=True).entries(keyed=False)
    entries = entries.filter((entries.entry >= r2) & (entries.i < entries.j))
    entries = entries.select(i=hl.int32(entries.i), j=hl.int32(entries.j))

    if keep_higher_maf:
        fields = ['mean', 'locus']
    else:
        fields = ['locus']

    info = locally_pruned_table.aggregate(
        hl.agg.collect(locally_pruned_table.row.select('idx', *fields)), _localize=False)
    info = hl.sorted(info, key=lambda x: x.idx)

    entries = entries.annotate_globals(info=info)

    entries = entries.filter(
        (entries.info[entries.i].locus.contig == entries.info[entries.j].locus.contig)
        & (entries.info[entries.j].locus.position - entries.info[entries.i].locus.position <= bp_window_size))

    if keep_higher_maf:
        entries = entries.annotate(
            i=hl.struct(idx=entries.i,
                        twice_maf=hl.min(entries.info[entries.i].mean, 2.0 - entries.info[entries.i].mean)),
            j=hl.struct(idx=entries.j,
                        twice_maf=hl.min(entries.info[entries.j].mean, 2.0 - entries.info[entries.j].mean)))

        def tie_breaker(left, right):
            return hl.sign(right.twice_maf - left.twice_maf)
    else:
        tie_breaker = None

    variants_to_remove = hl.maximal_independent_set(
        entries.i, entries.j, keep=False, tie_breaker=tie_breaker, keyed=False)

    locally_pruned_table = locally_pruned_table.annotate_globals(
        variants_to_remove=variants_to_remove.aggregate(
            hl.agg.collect_as_set(variants_to_remove.node.idx), _localize=False))
    return locally_pruned_table.filter(
        locally_pruned_table.variants_to_remove.contains(hl.int32(locally_pruned_table.idx)),
        keep=False
    ).select().persist()


def _warn_if_no_intercept(caller, covariates):
    if all([e._indices.axes for e in covariates]):
        warning(f'{caller}: model appears to have no intercept covariate.'
                '\n    To include an intercept, add 1.0 to the list of covariates.')
        return True
    return False
