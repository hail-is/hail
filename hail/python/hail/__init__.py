import importlib.resources
from pathlib import Path
from sys import version_info
from typing import Optional

if version_info < (3, 9):
    raise EnvironmentError(f'Hail requires Python 3.9 or later, found {version_info.major}.{version_info.minor}')


def __resource(name: str) -> Path:
    return importlib.resources.files(__name__) / name


def __resource_str(name: str) -> str:
    with __resource(name).open('r', encoding='utf-8') as fp:
        return fp.read()


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

# ruff: noqa: E402
from hail.utils import (
    ANY_REGION,
    Interval,
    Struct,
    copy_log,
    hadoop_copy,
    hadoop_exists,
    hadoop_is_dir,
    hadoop_is_file,
    hadoop_ls,
    hadoop_open,
    hadoop_scheme_supported,
    hadoop_stat,
)

from . import (
    backend,
    experimental,
    expr,
    genetics,
    ggplot,
    ir,
    linalg,
    methods,
    nd,
    plot,
    stats,
    utils,
    vds,
)
from .context import (
    TemporaryDirectory,
    TemporaryFilename,
    _async_current_backend,
    _get_flags,
    _set_flags,
    _with_flags,
    citation,
    cite_hail,
    cite_hail_bibtex,
    current_backend,
    debug_info,
    default_reference,
    get_reference,
    init,
    init_batch,
    init_local,
    init_spark,
    reset_global_randomness,
    set_global_seed,
    spark_context,
    stop,
    tmp_dir,
    version,
)
from .expr import *  # noqa: F403
from .expr import aggregators
from .genetics import *  # noqa: F403
from .matrixtable import GroupedMatrixTable, MatrixTable
from .methods import *  # noqa: F403
from .table import GroupedTable, Table, asc, desc

agg = aggregators
scan = aggregators.aggregators.ScanFunctions({name: getattr(agg, name) for name in agg.__all__})

__all__ = [
    'init',
    'init_local',
    'init_batch',
    'init_spark',
    'stop',
    'spark_context',
    'tmp_dir',
    'TemporaryFilename',
    'TemporaryDirectory',
    'default_reference',
    'get_reference',
    'set_global_seed',
    'reset_global_randomness',
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
    'utils',
    'version',
    'ANY_REGION',
]

__all__.extend(genetics.__all__)
__all__.extend(methods.__all__)

# don't overwrite builtins in `from hail import *`
import builtins

__all__.extend([x for x in expr.__all__ if not hasattr(builtins, x)])
del builtins

ir.register_functions()
ir.register_aggregators()

__pip_version__ = __resource_str('hail_pip_version').strip()
__version__: Optional[str] = None  # set by hail.version()
__revision__: Optional[str] = None  # set by hail.revision()

import warnings

warnings.filterwarnings('once', append=True)
del warnings
