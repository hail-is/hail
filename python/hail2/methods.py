from decorator import decorator

from hail2.matrixtable import MatrixTable
from hail2.table import Table
import hail2.expr.functions as f
from hail2.expr.expression import *
from hail.typ import *
from hail.typecheck import *
from hail.java import *
from hail.representation import Interval, Pedigree, Variant, GenomeReference, Struct
from hail.utils import Summary, wrap_to_list, hadoop_read
from hail.history import *


@typecheck(matrix=MatrixTable,
           ys=oneof(Expression, listof(Expression)),
           x=Expression,
           covariates=listof(Expression),
           root=strlike,
           block_size=integral)
def linreg(matrix, ys, x, covariates=[], root='linreg', block_size=16):
    """Test each variant for association with multiple phenotypes using linear regression.

    .. warning::

        :py:meth:`.linreg` uses the same set of samples for each phenotype,
        namely the set of samples for which **all** phenotypes and covariates are defined.

    **Annotations**

    With the default root, the following four variant annotations are added.
    The indexing of the array annotations corresponds to that of ``y``.

    - **va.linreg.nCompleteSamples** (*Int*) -- number of samples used
    - **va.linreg.AC** (*Double*) -- sum of input values ``x``
    - **va.linreg.ytx** (*Array[Double]*) -- array of dot products of each response vector ``y`` with the input vector ``x``
    - **va.linreg.beta** (*Array[Double]*) -- array of fit effect coefficients, :math:`\hat\beta_1`
    - **va.linreg.se** (*Array[Double]*) -- array of estimated standard errors, :math:`\widehat{\mathrm{se}}`
    - **va.linreg.tstat** (*Array[Double]*) -- array of :math:`t`-statistics, equal to :math:`\hat\beta_1 / \widehat{\mathrm{se}}`
    - **va.linreg.pval** (*Array[Double]*) -- array of :math:`p`-values

    :param ys: list of one or more response expressions.
    :type ys: list of str

    :param str x: expression for input variable

    :param covariates: list of covariate expressions.
    :type covariates: list of str

    :param str root: Variant annotation path to store result of linear regression.

    :param int variant_block_size: Number of variant regressions to perform simultaneously.  Larger block size requires more memmory.

    :return: Variant dataset with linear regression variant annotations.
    :rtype: :py:class:`.VariantDataset`

    """
    all_exprs = [x]

    ys = wrap_to_list(ys)

    # x is entry-indexed
    analyze(x, matrix._entry_indices, set(), set(matrix._fields.keys()))

    # ys and covariates are col-indexed
    for e in (tuple(wrap_to_list(ys)) + tuple(covariates)):
        all_exprs.append(e)
        analyze(e, matrix._col_indices, set(), set(matrix._fields.keys()))

    base, cleanup = matrix._process_joins(*all_exprs)

    jm = base._jvds.linreg(
        jarray(Env.jvm().java.lang.String, [y._ast.to_hql() for y in ys]),
        x._ast.to_hql(),
        jarray(Env.jvm().java.lang.String, [cov._ast.to_hql() for cov in covariates]),
        'va.`{}`'.format(root),
        block_size
    )

    return cleanup(MatrixTable(matrix._hc, jm))
