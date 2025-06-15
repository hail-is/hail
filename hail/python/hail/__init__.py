import importlib.resources
from pathlib import Path
from sys import version_info

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
# ruff: noqa: F403
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
)
from .expr import *
from .expr import aggregators
from .genetics import *
from .matrixtable import GroupedMatrixTable, MatrixTable
from .methods import *
from .table import GroupedTable, Table, asc, desc
from .version import __pip_version__, __revision__, __version__

agg = aggregators
scan = aggregators.aggregators.ScanFunctions({name: getattr(agg, name) for name in agg.__all__})


def version() -> str:
    return __version__


__all__ = [
    'ANY_REGION',
    'GroupedMatrixTable',
    'GroupedTable',
    'Interval',
    'MatrixTable',
    'Struct',
    'Table',
    'TemporaryDirectory',
    'TemporaryFilename',
    '__pip_version__',
    '__revision__',
    '__version__',
    '_async_current_backend',
    '_get_flags',
    '_set_flags',
    '_with_flags',
    'agg',
    'asc',
    'backend',
    'citation',
    'cite_hail',
    'cite_hail_bibtex',
    'copy_log',
    'current_backend',
    'debug_info',
    'default_reference',
    'desc',
    'experimental',
    'genetics',
    'get_reference',
    'ggplot',
    'hadoop_copy',
    'hadoop_exists',
    'hadoop_is_dir',
    'hadoop_is_file',
    'hadoop_ls',
    'hadoop_open',
    'hadoop_scheme_supported',
    'hadoop_stat',
    'init',
    'init_batch',
    'init_local',
    'init_spark',
    'ir',
    'linalg',
    'methods',
    'nd',
    'plot',
    'reset_global_randomness',
    'scan',
    'set_global_seed',
    'spark_context',
    'stats',
    'stop',
    'tmp_dir',
    'utils',
    'vds',
    'version',
]

__all__.extend(genetics.__all__)
__all__.extend(methods.__all__)

# don't overwrite builtins in `from hail import *`
import builtins

__all__.extend([x for x in expr.__all__ if not hasattr(builtins, x)])
del builtins

ir.register_functions()
ir.register_aggregators()

import warnings

warnings.filterwarnings('once', append=True)
del warnings
