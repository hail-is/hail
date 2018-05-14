import sys
if sys.version_info < (3, 6):
    raise EnvironmentError('Hail requires Python 3.6, found {}.{}'.format(
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

from .context import init, stop, spark_context, default_reference, get_reference
from .table import Table, GroupedTable, asc, desc
from .matrixtable import MatrixTable, GroupedMatrixTable
from .expr import *
from .genetics import *
from . import genetics as genetics
from .methods import *
from . import methods as methods
from . import linalg as linalg
from hail.expr import aggregators as agg
from hail.utils import Struct, Interval, hadoop_copy, hadoop_open


__all__ = ['init',
           'stop',
           'spark_context',
           'default_reference',
           'get_reference',
           'Table',
           'GroupedTable',
           'MatrixTable',
           'GroupedMatrixTable',
           'asc',
           'desc',
           'hadoop_open',
           'hadoop_copy',
           'Struct',
           'Interval',
           'agg',
           'genetics',
           'methods',
           'linalg']


__all__.extend(genetics.__all__)
__all__.extend(methods.__all__)

# don't overwrite builtins in `from hail import *`
import builtins
__all__.extend([x for x in expr.__all__ if not hasattr(builtins, x)])
del builtins

__version__ = None  # set in hail.init()

import warnings
warnings.filterwarnings('once', append=True)
del warnings
