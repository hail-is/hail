import sys
if sys.version_info >= (3, 0) or sys.version_info <= (2, 6):
    raise EnvironmentError('Hail requires Python 2.7, found {}.{}'.format(
        sys.version_info.major, sys.version_info.minor))
del sys

__doc__ = """
    __  __     <>__ 
   / /_/ /__  __/ / 
  / __  / _ `/ / /  
 /_/ /_/\_,_/_/_/
===================
    
For API documentation, visit the website: www.hail.is

For help, visit either:
 - the forum (discuss.hail.is) 
 - or the gitter channel: https://gitter.im/hail-is/hail/
 
To report a bug, please open an issue: https://github.com/hail-is/hail/issues
"""

from .context import init, stop, default_reference
from .table import Table, GroupedTable, asc, desc
from .matrixtable import MatrixTable, GroupedMatrixTable
from .expr import *
from .genetics import *
from .methods import *
from hail.expr import aggregators as agg
from hail.utils import Struct, hadoop_write, hadoop_read, hadoop_copy


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
           'Struct',
           'agg']


__all__.extend(genetics.__all__)
__all__.extend(methods.__all__)

# don't import the following with `from hail import *`
builtins = {'str', 'any', 'all', 'set', 'dict', 'sum', 'min', 'max', 'len', 'sorted', 'map', 'filter', 'range'}
__all__.extend([x for x in expr.__all__ if x not in builtins])
del builtins
