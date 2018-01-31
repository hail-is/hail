import sys

from hail.context import init, stop, default_reference
from hail.expr.types import Type, TInt32, TInt64, TFloat32, TFloat64, TSet, TString, TBoolean, TArray, TDict, TLocus, \
    TVariant, TAltAllele, TCall, TInterval, TStruct
from hail.utils import hadoop_read, hadoop_write, hadoop_copy, Struct

from hail.table import Table, GroupedTable, asc, desc
from hail.matrixtable import MatrixTable, GroupedMatrixTable
import hail.expr.functions as functions
import hail.expr.aggregators as agg
import hail.methods as methods
from hail.expr import Type, TInt32, TInt64, TFloat32, TFloat64, TString, TBoolean, TArray, TSet, TDict, TStruct, \
    TLocus, TVariant, TAltAllele, TCall, TInterval, eval_expr, eval_expr_typed
from hail.genetics import LDMatrix, KinshipMatrix, Variant, Locus, AltAllele, Interval, Call, Pedigree, Trio, \
    GenomeReference
from hail.utils import Struct, hadoop_write, hadoop_read, hadoop_copy

from pyspark import SparkContext
from pyspark.sql import SQLContext


if sys.version_info >= (3, 0) or sys.version_info <= (2, 6):
    raise EnvironmentError('Hail requires Python 2.7, found {}.{}'.format(
        sys.version_info.major, sys.version_info.minor))


def default_reference():
    """Return the default reference genome.

    Returns
    -------
    :class:`.GenomeReference`
    """
    return Env.hc().default_reference

__all__ = ['init',
           'stop',
           'default_reference',
           'Table',
           'GroupedTable',
           'MatrixTable',
           'GroupedMatrixTable',
           'asc',
           'desc',
           'hadoop_read',
           'hadoop_write',
           'hadoop_copy',
           'functions',
           'agg',
           'methods',
           'eval_expr',
           'eval_expr_typed',
           'Variant',
           'GenomeReference',
           'Locus',
           'AltAllele',
           'Interval',
           'Struct',
           'Call',
           'Pedigree',
           'Trio',
           'Type',
           'TInt32',
           'TInt64',
           'TFloat32',
           'TFloat64',
           'TString',
           'TBoolean',
           'TArray',
           'TSet',
           'TDict',
           'TStruct',
           'TLocus',
           'TVariant',
           'TAltAllele',
           'TCall',
           'TInterval',
           'KinshipMatrix',
           'LDMatrix']
