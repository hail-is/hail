import sys

from hail.api1 import HailContext, VariantDataset, KeyTable
from hail.expr.types import Type, TInt32, TInt64, TFloat32, TFloat64, TSet, TString, TBoolean, TArray, TDict, TLocus, \
    TVariant, TAltAllele, TCall, TInterval, TStruct
from hail.genetics import *
from hail.utils import hadoop_read, hadoop_write, hadoop_copy, Struct

if sys.version_info >= (3, 0) or sys.version_info <= (2, 6):
    raise EnvironmentError('Hail requires Python 2.7, found {}.{}'.format(
        sys.version_info.major, sys.version_info.minor))

__all__ = ['HailContext',
           'VariantDataset',
           'KeyTable',
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
           'hadoop_read',
           'hadoop_write',
           'hadoop_copy',
           'KinshipMatrix',
           'LDMatrix',
           ]
