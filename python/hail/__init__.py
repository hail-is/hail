from hail.representation import *
from hail.context import HailContext
from hail.dataset import VariantDataset
from hail.typ import TInt32, TInt64, TFloat32, TFloat64, TSet, TString, TBoolean, TArray, TDict, TLocus, TVariant, TAltAllele, TCall, TInterval
from hail.keytable import KeyTable
from hail.kinshipMatrix import KinshipMatrix
from hail.ldMatrix import LDMatrix
from hail.utils import hadoop_read, hadoop_write, hadoop_copy

import sys
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
           'typ',
           'representation'
           ]
