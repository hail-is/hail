from typing import Optional
import pkg_resources
import sys
import asyncio
import nest_asyncio

if sys.version_info < (3, 6):
    raise EnvironmentError('Hail requires Python 3.6 or later, found {}.{}'.format(
        sys.version_info.major, sys.version_info.minor))

if sys.version_info[:2] == (3, 6):
    if asyncio._get_running_loop() is not None:
        nest_asyncio.apply()
else:
    try:
        asyncio.get_running_loop()
        nest_asyncio.apply()
    except RuntimeError as err:
        assert 'no running event loop' in err.args[0]


__pip_version__ = pkg_resources.resource_string(__name__, 'hail_pip_version').decode().strip()
del pkg_resources
del sys

__doc__ = r"""
    __  __     <>__
   / /_/ /__  __/ /
  / __  / _ `/ / /
 /_/ /_/\_,_/_/_/
===================

For API documentation, visit the website: https://www.hail.is

For help, visit either:
 - the forum: https://discuss.hail.is
 - or our Zulip chatroom: https://hail.zulipchat.com

To report a bug, please open an issue: https://github.com/hail-is/hail/issues
"""

# F403 'from .expr import *' used; unable to detect undefined names
# F401 '.expr.*' imported but unused
# E402 module level import not at top of file
from .table import Table, GroupedTable, asc, desc  # noqa: E402
from .matrixtable import MatrixTable, GroupedMatrixTable  # noqa: E402
from .expr import *  # noqa: F401,F403,E402
from .genetics import *  # noqa: F401,F403,E402
from .methods import *  # noqa: F401,F403,E402
from . import expr  # noqa: E402
from . import genetics  # noqa: E402
from . import methods  # noqa: E402
from . import stats  # noqa: E402
from . import linalg  # noqa: E402
from . import plot  # noqa: E402
from . import ggplot  # noqa: E402
from . import experimental  # noqa: E402
from . import ir  # noqa: E402
from . import backend  # noqa: E402
from . import nd  # noqa: E402
from . import vds  # noqa: E402
from hail.expr import aggregators as agg  # noqa: E402
from hail.utils import (Struct, Interval, hadoop_copy, hadoop_open, hadoop_ls,  # noqa: E402
                        hadoop_stat, hadoop_exists, hadoop_is_file,
                        hadoop_is_dir, hadoop_scheme_supported, copy_log)

from .context import (init, init_local, init_batch, stop, spark_context, tmp_dir,  # noqa: E402
                      default_reference, get_reference, set_global_seed, _set_flags, _get_flags, _with_flags,
                      _async_current_backend, current_backend, debug_info, citation, cite_hail,
                      cite_hail_bibtex, version, TemporaryFilename, TemporaryDirectory)

scan = agg.aggregators.ScanFunctions({name: getattr(agg, name) for name in agg.__all__})

__all__ = [
    'init',
    'init_local',
    'init_batch',
    'stop',
    'spark_context',
    'tmp_dir',
    'TemporaryFilename',
    'TemporaryDirectory',
    'default_reference',
    'get_reference',
    'set_global_seed',
    '_set_flags',
    '_get_flags',
    '_with_flags',
    'Table',
    'GroupedTable',
    'MatrixTable',
    'GroupedMatrixTable',
    'asc',
    'desc',
    'hadoop_open',
    'hadoop_copy',
    'hadoop_is_dir',
    'hadoop_is_file',
    'hadoop_stat',
    'hadoop_exists',
    'hadoop_ls',
    'hadoop_scheme_supported',
    'copy_log',
    'Struct',
    'Interval',
    'agg',
    'scan',
    'genetics',
    'methods',
    'stats',
    'linalg',
    'nd',
    'plot',
    'ggplot',
    'experimental',
    'ir',
    'vds',
    'backend',
    '_async_current_backend',
    'current_backend',
    'debug_info',
    'citation',
    'cite_hail',
    'cite_hail_bibtex',
    'version'
]

__all__.extend(genetics.__all__)
__all__.extend(methods.__all__)

# don't overwrite builtins in `from hail import *`
import builtins  # noqa: E402

__all__.extend([x for x in expr.__all__ if not hasattr(builtins, x)])
del builtins

ir.register_functions()
ir.register_aggregators()

__version__: Optional[str] = None  # set by hail.version()
__revision__: Optional[str] = None  # set by hail.revision()

import warnings  # noqa: E402

warnings.filterwarnings('once', append=True)
del warnings
