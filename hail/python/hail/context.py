from typing import Optional, Union, Tuple, List, Dict
import warnings
import sys
import os
from contextlib import contextmanager
from urllib.parse import urlparse, urlunparse
from random import Random

import pkg_resources
from pyspark import SparkContext

import hail
from hail.genetics.reference_genome import ReferenceGenome
from hail.typecheck import (nullable, typecheck, typecheck_method, enumeration, dictof, oneof,
                            sized_tupleof, sequenceof)
from hail.utils import get_env_or_default
from hail.utils.java import Env, warning, choose_backend
from hail.backend import Backend
from hailtop.utils import secret_alnum_string
from hailtop.fs.fs import FS
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration, get_gcs_requester_pays_configuration
from .builtin_references import BUILTIN_REFERENCES
from .backend.backend import local_jar_information


def _get_tmpdir(tmpdir):
    if tmpdir is None:
        tmpdir = '/tmp'
    return tmpdir


def _get_local_tmpdir(local_tmpdir):
    local_tmpdir = get_env_or_default(local_tmpdir, 'TMPDIR', 'file:///tmp')
    r = urlparse(local_tmpdir)
    if not r.scheme:
        r = r._replace(scheme='file')
    elif r.scheme != 'file':
        raise ValueError('invalid local_tmpfile: must use scheme file, got scheme {r.scheme}')
    return urlunparse(r)


def _get_log(log):
    if log is None:
        py_version = version()
        log_dir = os.environ.get('HAIL_LOG_DIR')
        if log_dir is None:
            log_dir = os.getcwd()
        log = hail.utils.timestamp_path(os.path.join(log_dir, 'hail'),
                                        suffix=f'-{py_version}.log')
    return log


def convert_gcs_requester_pays_configuration_to_hadoop_conf_style(
    x: Optional[Union[str, Tuple[str, List[str]]]]
) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(x, str):
        return x, None
    if isinstance(x, tuple):
        return x[0], ",".join(x[1])
    return None, None


class HailContext(object):
    @staticmethod
    def create(log: str,
               quiet: bool,
               append: bool,
               tmpdir: str,
               local_tmpdir: str,
               default_reference: str,
               global_seed: Optional[int],
               backend: Backend):
        hc = HailContext(log=log,
                         quiet=quiet,
                         append=append,
                         tmpdir=tmpdir,
                         local_tmpdir=local_tmpdir,
                         global_seed=global_seed,
                         backend=backend)
        hc.initialize_references(default_reference)
        return hc

    @typecheck_method(log=str,
                      quiet=bool,
                      append=bool,
                      tmpdir=str,
                      local_tmpdir=str,
                      global_seed=nullable(int),
                      backend=Backend)
    def __init__(self, log, quiet, append, tmpdir, local_tmpdir, global_seed, backend):
        assert not Env._hc

        self._log = log

        self._tmpdir = tmpdir
        self._local_tmpdir = local_tmpdir

        self._backend = backend

        self._warn_cols_order = True
        self._warn_entries_order = True

        self._default_ref: Optional[ReferenceGenome] = None

        if not quiet:
            py_version = version()
            sys.stderr.write(
                'Welcome to\n'
                '     __  __     <>__\n'
                '    / /_/ /__  __/ /\n'
                '   / __  / _ `/ / /\n'
                '  /_/ /_/\\_,_/_/_/   version {}\n'.format(py_version))

            if py_version.startswith('devel'):
                sys.stderr.write('NOTE: This is a beta version. Interfaces may change\n'
                                 '  during the beta period. We recommend pulling\n'
                                 '  the latest changes weekly.\n')
            sys.stderr.write(f'LOGGING: writing to {log}\n')

        self._user_specified_rng_nonce = True
        if global_seed is None:
            if 'rng_nonce' not in backend.get_flags('rng_nonce'):
                backend.set_flags(rng_nonce=hex(Random().randrange(-2**63, 2**63 - 1)))
                self._user_specified_rng_nonce = False
        else:
            backend.set_flags(rng_nonce=hex(global_seed))
        Env._hc = self

    def initialize_references(self, default_reference):
        assert self._backend
        self._backend.initialize_references()
        if default_reference in BUILTIN_REFERENCES:
            self._default_ref = self._backend.get_reference(default_reference)
        else:
            self._default_ref = ReferenceGenome.read(default_reference)

    @property
    def default_reference(self) -> ReferenceGenome:
        assert self._default_ref is not None, '_default_ref should have been initialized in HailContext.create'
        return self._default_ref

    def stop(self):
        assert self._backend
        self._backend.stop()
        self._backend = None
        Env._hc = None
        Env._dummy_table = None
        Env._seed_generator = None
        hail.ir.clear_session_functions()


@typecheck(sc=nullable(SparkContext),
           app_name=nullable(str),
           master=nullable(str),
           local=str,
           log=nullable(str),
           quiet=bool,
           append=bool,
           min_block_size=int,
           branching_factor=int,
           tmp_dir=nullable(str),
           default_reference=enumeration(*BUILTIN_REFERENCES),
           idempotent=bool,
           global_seed=nullable(int),
           spark_conf=nullable(dictof(str, str)),
           skip_logging_configuration=bool,
           local_tmpdir=nullable(str),
           _optimizer_iterations=nullable(int),
           backend=nullable(str),
           driver_cores=nullable(oneof(str, int)),
           driver_memory=nullable(str),
           worker_cores=nullable(oneof(str, int)),
           worker_memory=nullable(str),
           gcs_requester_pays_configuration=nullable(oneof(str, sized_tupleof(str, sequenceof(str)))),
           regions=nullable(sequenceof(str)),
           gcs_bucket_allow_list=nullable(dictof(str, sequenceof(str))))
def init(sc=None,
         app_name=None,
         master=None,
         local='local[*]',
         log=None,
         quiet=False,
         append=False,
         min_block_size=0,
         branching_factor=50,
         tmp_dir=None,
         default_reference='GRCh37',
         idempotent=False,
         global_seed=None,
         spark_conf=None,
         skip_logging_configuration=False,
         local_tmpdir=None,
         _optimizer_iterations=None,
         *,
         backend=None,
         driver_cores=None,
         driver_memory=None,
         worker_cores=None,
         worker_memory=None,
         gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None,
         regions: Optional[List[str]] = None,
         gcs_bucket_allow_list: Optional[Dict[str, List[str]]] = None):
    """Initialize and configure Hail.

    This function will be called with default arguments if any Hail functionality is used. If you
    need custom configuration, you must explicitly call this function before using Hail. For
    example, to set the default reference genome to GRCh38, import Hail and immediately call
    :func:`.init`:

    >>> import hail as hl
    >>> hl.init(default_reference='GRCh38')  # doctest: +SKIP

    Hail has two backends, ``spark`` and ``batch``. Hail selects a backend by consulting, in order,
    these configuration locations:

    1. The ``backend`` parameter of this function.
    2. The ``HAIL_QUERY_BACKEND`` environment variable.
    3. The value of ``hailctl config get query/backend``.

    If no configuration is found, Hail will select the Spark backend.

    Examples
    --------
    Configure Hail to use the Batch backend:

    >>> import hail as hl
    >>> hl.init(backend='batch')  # doctest: +SKIP

    If a :class:`pyspark.SparkContext` is already running, then Hail must be
    initialized with it as an argument:

    >>> hl.init(sc=sc)  # doctest: +SKIP

    Configure Hail to bill to `my_project` when accessing any Google Cloud Storage bucket that has
    requester pays enabled:

    >>> hl.init(gcs_requester_pays_configuration='my-project')  # doctest: +SKIP

    Configure Hail to bill to `my_project` when accessing the Google Cloud Storage buckets named
    `bucket_of_fish` and `bucket_of_eels`:

    >>> hl.init(
    ...     gcs_requester_pays_configuration=('my-project', ['bucket_of_fish', 'bucket_of_eels'])
    ... )  # doctest: +SKIP

    You may also use `hailctl config set gcs_requester_pays/project` and `hailctl config set
    gcs_requester_pays/buckets` to achieve the same effect.

    See Also
    --------
    :func:`.stop`

    Parameters
    ----------
    sc : pyspark.SparkContext, optional
        Spark Backend only. Spark context. If not specified, the Spark backend will create a new
        Spark context.
    app_name : :class:`str`
        A name for this pipeline. In the Spark backend, this becomes the Spark application name. In
        the Batch backend, this is a prefix for the name of every Batch.
    master : :class:`str`, optional
        Spark Backend only. URL identifying the Spark leader (master) node or `local[N]` for local
        clusters.
    local : :class:`str`
        Spark Backend only. Local-mode core limit indicator. Must either be `local[N]` where N is a
        positive integer or `local[*]`. The latter indicates Spark should use all cores
        available. `local[*]` does not respect most containerization CPU limits. This option is only
        used if `master` is unset and `spark.master` is not set in the Spark configuration.
    log : :class:`str`
        Local path for Hail log file. Does not currently support distributed file systems like
        Google Storage, S3, or HDFS.
    quiet : :obj:`bool`
        Print fewer log messages.
    append : :obj:`bool`
        Append to the end of the log file.
    min_block_size : :obj:`int`
        Minimum file block size in MB.
    branching_factor : :obj:`int`
        Branching factor for tree aggregation.
    tmp_dir : :class:`str`, optional
        Networked temporary directory.  Must be a network-visible file
        path.  Defaults to /tmp in the default scheme.
    default_reference : :class:`str`
        Default reference genome. Either ``'GRCh37'``, ``'GRCh38'``,
        ``'GRCm38'``, or ``'CanFam3'``.
    idempotent : :obj:`bool`
        If ``True``, calling this function is a no-op if Hail has already been initialized.
    global_seed : :obj:`int`, optional
        Global random seed.
    spark_conf : :obj:`dict` of :class:`str` to :class`str`, optional
        Spark backend only. Spark configuration parameters.
    skip_logging_configuration : :obj:`bool`
        Spark Backend only. Skip logging configuration in java and python.
    local_tmpdir : :class:`str`, optional
        Local temporary directory.  Used on driver and executor nodes.
        Must use the file scheme.  Defaults to TMPDIR, or /tmp.
    driver_cores : :class:`str` or :class:`int`, optional
        Batch backend only. Number of cores to use for the driver process. May be 1, 2, 4, or 8. Default is
        1.
    driver_memory : :class:`str`, optional
        Batch backend only. Memory tier to use for the driver process. May be standard or
        highmem. Default is standard.
    worker_cores : :class:`str` or :class:`int`, optional
        Batch backend only. Number of cores to use for the worker processes. May be 1, 2, 4, or 8. Default is
        1.
    worker_memory : :class:`str`, optional
        Batch backend only. Memory tier to use for the worker processes. May be standard or
        highmem. Default is standard.
    gcs_requester_pays_configuration : either :class:`str` or :class:`tuple` of :class:`str` and :class:`list` of :class:`str`, optional
        If a string is provided, configure the Google Cloud Storage file system to bill usage to the
        project identified by that string. If a tuple is provided, configure the Google Cloud
        Storage file system to bill usage to the specified project for buckets specified in the
        list. See examples above.
    regions : :obj:`list` of :class:`str`, optional
        List of regions to run jobs in when using the Batch backend. Use :data:`.ANY_REGION` to specify any region is allowed
        or use `None` to use the underlying default regions from the hailctl environment configuration. For example, use
        `hailctl config set batch/regions region1,region2` to set the default regions to use.
    gcs_bucket_allow_list:
        A list of buckets that Hail should be permitted to read from or write to, even if their default policy is to
        use "cold" storage. Should look like ``["bucket1", "bucket2"]``.
    """
    if Env._hc:
        if idempotent:
            return
        else:
            warning('Hail has already been initialized. If this call was intended to change configuration,'
                    ' close the session with hl.stop() first.')

    backend = choose_backend(backend)

    if backend == 'service':
        warnings.warn(
            'The "service" backend is now called the "batch" backend. Support for "service" will be removed in a '
            'future release.'
        )
        backend = 'batch'

    if backend == 'batch':
        import nest_asyncio
        nest_asyncio.apply()
        import asyncio
        return asyncio.get_event_loop().run_until_complete(init_batch(
            log=log,
            quiet=quiet,
            append=append,
            tmpdir=tmp_dir,
            local_tmpdir=local_tmpdir,
            default_reference=default_reference,
            global_seed=global_seed,
            driver_cores=driver_cores,
            driver_memory=driver_memory,
            worker_cores=worker_cores,
            worker_memory=worker_memory,
            name_prefix=app_name,
            gcs_requester_pays_configuration=gcs_requester_pays_configuration,
            regions=regions,
            gcs_bucket_allow_list=gcs_bucket_allow_list
        ))
    if backend == 'spark':
        return init_spark(
            sc=sc,
            app_name=app_name,
            master=master,
            local=local,
            min_block_size=min_block_size,
            branching_factor=branching_factor,
            spark_conf=spark_conf,
            _optimizer_iterations=_optimizer_iterations,
            log=log,
            quiet=quiet,
            append=append,
            tmp_dir=tmp_dir,
            local_tmpdir=local_tmpdir,
            default_reference=default_reference,
            global_seed=global_seed,
            skip_logging_configuration=skip_logging_configuration,
            gcs_requester_pays_configuration=gcs_requester_pays_configuration
        )
    if backend == 'local':
        return init_local(
            log=log,
            quiet=quiet,
            append=append,
            tmpdir=tmp_dir,
            default_reference=default_reference,
            global_seed=global_seed,
            skip_logging_configuration=skip_logging_configuration,
            gcs_requester_pays_configuration=gcs_requester_pays_configuration
        )
    raise ValueError(f'unknown Hail Query backend: {backend}')


@typecheck(sc=nullable(SparkContext),
           app_name=nullable(str),
           master=nullable(str),
           local=str,
           log=nullable(str),
           quiet=bool,
           append=bool,
           min_block_size=int,
           branching_factor=int,
           tmp_dir=nullable(str),
           default_reference=enumeration(*BUILTIN_REFERENCES),
           idempotent=bool,
           global_seed=nullable(int),
           spark_conf=nullable(dictof(str, str)),
           skip_logging_configuration=bool,
           local_tmpdir=nullable(str),
           _optimizer_iterations=nullable(int),
           gcs_requester_pays_configuration=nullable(oneof(str, sized_tupleof(str, sequenceof(str)))))
def init_spark(sc=None,
               app_name=None,
               master=None,
               local='local[*]',
               log=None,
               quiet=False,
               append=False,
               min_block_size=0,
               branching_factor=50,
               tmp_dir=None,
               default_reference='GRCh37',
               idempotent=False,
               global_seed=None,
               spark_conf=None,
               skip_logging_configuration=False,
               local_tmpdir=None,
               _optimizer_iterations=None,
               gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None
               ):
    from hail.backend.spark_backend import SparkBackend, connect_logger

    log = _get_log(log)
    tmpdir = _get_tmpdir(tmp_dir)
    local_tmpdir = _get_local_tmpdir(local_tmpdir)
    optimizer_iterations = get_env_or_default(_optimizer_iterations, 'HAIL_OPTIMIZER_ITERATIONS', 3)

    app_name = app_name or 'Hail'
    gcs_requester_pays_project, gcs_requester_pays_buckets = convert_gcs_requester_pays_configuration_to_hadoop_conf_style(
        get_gcs_requester_pays_configuration(
            gcs_requester_pays_configuration=gcs_requester_pays_configuration,
        )
    )
    backend = SparkBackend(
        idempotent, sc, spark_conf, app_name, master, local, log,
        quiet, append, min_block_size, branching_factor, tmpdir, local_tmpdir,
        skip_logging_configuration, optimizer_iterations,
        gcs_requester_pays_project=gcs_requester_pays_project,
        gcs_requester_pays_buckets=gcs_requester_pays_buckets
    )
    if not backend.fs.exists(tmpdir):
        backend.fs.mkdir(tmpdir)

    HailContext.create(
        log, quiet, append, tmpdir, local_tmpdir, default_reference,
        global_seed, backend)
    if not quiet:
        connect_logger(backend._utils_package_object, 'localhost', 12888)


@typecheck(
    billing_project=nullable(str),
    remote_tmpdir=nullable(str),
    jar_url=nullable(str),
    log=nullable(str),
    quiet=bool,
    append=bool,
    tmpdir=nullable(str),
    local_tmpdir=nullable(str),
    default_reference=enumeration(*BUILTIN_REFERENCES),
    global_seed=nullable(int),
    disable_progress_bar=nullable(bool),
    driver_cores=nullable(oneof(str, int)),
    driver_memory=nullable(str),
    worker_cores=nullable(oneof(str, int)),
    worker_memory=nullable(str),
    name_prefix=nullable(str),
    token=nullable(str),
    gcs_requester_pays_configuration=nullable(oneof(str, sized_tupleof(str, sequenceof(str)))),
    regions=nullable(sequenceof(str)),
    gcs_bucket_allow_list=nullable(sequenceof(str))
)
async def init_batch(
        *,
        billing_project: Optional[str] = None,
        remote_tmpdir: Optional[str] = None,
        jar_url: Optional[str] = None,
        log: Optional[str] = None,
        quiet: bool = False,
        append: bool = False,
        tmpdir: Optional[str] = None,
        local_tmpdir: Optional[str] = None,
        default_reference: str = 'GRCh37',
        global_seed: Optional[int] = None,
        disable_progress_bar: Optional[bool] = None,
        driver_cores: Optional[Union[str, int]] = None,
        driver_memory: Optional[str] = None,
        worker_cores: Optional[Union[str, int]] = None,
        worker_memory: Optional[str] = None,
        name_prefix: Optional[str] = None,
        token: Optional[str] = None,
        gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None,
        regions: Optional[List[str]] = None,
        gcs_bucket_allow_list: Optional[List[str]] = None
):
    from hail.backend.service_backend import ServiceBackend
    # FIXME: pass local_tmpdir and use on worker and driver
    backend = await ServiceBackend.create(billing_project=billing_project,
                                          remote_tmpdir=remote_tmpdir,
                                          disable_progress_bar=disable_progress_bar,
                                          jar_url=jar_url,
                                          driver_cores=driver_cores,
                                          driver_memory=driver_memory,
                                          worker_cores=worker_cores,
                                          worker_memory=worker_memory,
                                          name_prefix=name_prefix,
                                          credentials_token=token,
                                          regions=regions,
                                          gcs_requester_pays_configuration=gcs_requester_pays_configuration,
                                          gcs_bucket_allow_list=gcs_bucket_allow_list)

    log = _get_log(log)
    if tmpdir is None:
        tmpdir = backend.remote_tmpdir + 'tmp/hail/' + secret_alnum_string()
    local_tmpdir = _get_local_tmpdir(local_tmpdir)

    HailContext.create(
        log, quiet, append, tmpdir, local_tmpdir, default_reference,
        global_seed, backend)


@typecheck(
    log=nullable(str),
    quiet=bool,
    append=bool,
    branching_factor=int,
    tmpdir=nullable(str),
    default_reference=enumeration(*BUILTIN_REFERENCES),
    global_seed=nullable(int),
    skip_logging_configuration=bool,
    jvm_heap_size=nullable(str),
    _optimizer_iterations=nullable(int),
    gcs_requester_pays_configuration=nullable(oneof(str, sized_tupleof(str, sequenceof(str))))
)
def init_local(
        log=None,
        quiet=False,
        append=False,
        branching_factor=50,
        tmpdir=None,
        default_reference='GRCh37',
        global_seed=None,
        skip_logging_configuration=False,
        jvm_heap_size=None,
        _optimizer_iterations=None,
        gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None
):
    from hail.backend.local_backend import LocalBackend, connect_logger

    log = _get_log(log)
    tmpdir = _get_tmpdir(tmpdir)
    optimizer_iterations = get_env_or_default(_optimizer_iterations, 'HAIL_OPTIMIZER_ITERATIONS', 3)

    jvm_heap_size = get_env_or_default(jvm_heap_size, 'HAIL_LOCAL_BACKEND_HEAP_SIZE', None)
    gcs_requester_pays_project, gcs_requester_pays_buckets = convert_gcs_requester_pays_configuration_to_hadoop_conf_style(
        get_gcs_requester_pays_configuration(
            gcs_requester_pays_configuration=gcs_requester_pays_configuration,
        )
    )
    backend = LocalBackend(
        tmpdir, log, quiet, append, branching_factor,
        skip_logging_configuration, optimizer_iterations,
        jvm_heap_size,
        gcs_requester_pays_project=gcs_requester_pays_project,
        gcs_requester_pays_buckets=gcs_requester_pays_buckets
    )

    if not backend.fs.exists(tmpdir):
        backend.fs.mkdir(tmpdir)

    HailContext.create(
        log, quiet, append, tmpdir, tmpdir, default_reference,
        global_seed, backend)
    if not quiet:
        connect_logger(backend._utils_package_object, 'localhost', 12888)


def version() -> str:
    """Get the installed Hail version.

    Returns
    -------
    str
    """
    if hail.__version__ is None:
        # https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package
        hail.__version__ = pkg_resources.resource_string(__name__, 'hail_version').decode().strip()
    return hail.__version__


def revision() -> str:
    """Get the installed Hail git revision.

    Returns
    -------
    str
    """
    if hail.__revision__ is None:
        # https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package
        hail.__revision__ = pkg_resources.resource_string(__name__, 'hail_revision').decode().strip()
    return hail.__revision__


def _hail_cite_url():
    v = version()
    [tag, sha_prefix] = v.split("-")
    if not local_jar_information().development_mode:
        # pip installed
        return f"https://github.com/hail-is/hail/releases/tag/{tag}"
    return f"https://github.com/hail-is/hail/commit/{sha_prefix}"


def citation(*, bibtex=False):
    """Generate a Hail citation.

    Parameters
    ----------
    bibtex : bool
        Generate a citation in BibTeX form.

    Returns
    -------
    str
    """
    if bibtex:
        return f"@misc{{Hail," \
            f"  author = {{Hail Team}}," \
            f"  title = {{Hail}}," \
            f"  howpublished = {{\\url{{{_hail_cite_url()}}}}}" \
            f"}}"
    return f"Hail Team. Hail {version()}. {_hail_cite_url()}."


def cite_hail():
    return citation(bibtex=False)


def cite_hail_bibtex():
    return citation(bibtex=True)


def stop():
    """Stop the currently running Hail session."""
    if Env._hc:
        Env.hc().stop()


def spark_context():
    """Returns the active Spark context.

    Returns
    -------
    :class:`pyspark.SparkContext`
    """
    return Env.spark_backend('spark_context').sc


def tmp_dir() -> str:
    """Returns the Hail shared temporary directory.

    Returns
    -------
    :class:`str`
    """
    return Env.hc()._tmpdir


class _TemporaryFilenameManager:
    def __init__(self, fs: FS, name: str):
        self.fs = fs
        self.name = name

    def __enter__(self):
        return self.name

    def __exit__(self, type, value, traceback):
        try:
            return self.fs.remove(self.name)
        except FileNotFoundError:
            pass


def TemporaryFilename(*,
                      prefix: str = '',
                      suffix: str = '',
                      dir: Optional[str] = None
                      ) -> _TemporaryFilenameManager:
    """A context manager which produces a temporary filename that is deleted when the context manager exits.

    Warning
    -------

    The filename is generated randomly and is extraordinarly unlikely to already exist, but this
    function does not satisfy the strict requirements of Python's :class:`.TemporaryFilename`.

    Examples
    --------

    >>> with TemporaryFilename() as f:  # doctest: +SKIP
    ...     open(f, 'w').write('hello hail')
    ...     print(open(f).read())
    hello hail

    Returns
    -------
    :class:`._TemporaryFilenameManager`

    """
    if dir is None:
        dir = tmp_dir()
    if not dir.endswith('/'):
        dir = dir + '/'
    return _TemporaryFilenameManager(
        current_backend().fs,
        dir + prefix + secret_alnum_string(10) + suffix)


class _TemporaryDirectoryManager:
    def __init__(self, fs: FS, name: str):
        self.fs = fs
        self.name = name

    def __enter__(self):
        return self.name

    def __exit__(self, type, value, traceback):
        try:
            return self.fs.rmtree(self.name)
        except FileNotFoundError:
            pass


def TemporaryDirectory(*,
                       prefix: str = '',
                       suffix: str = '',
                       dir: Optional[str] = None,
                       ensure_exists: bool = True
                       ) -> _TemporaryDirectoryManager:
    """A context manager which produces a temporary directory name that is recursively deleted when the context manager exits.

    If the filesystem has a notion of directories, then we ensure the directory exists.

    Warning
    -------

    The directory name is generated randomly and is extraordinarly unlikely to already exist, but
    this function does not satisfy the strict requirements of Python's :class:`.TemporaryDirectory`.

    Examples
    --------

    >>> with TemporaryDirectory() as dir:  # doctest: +SKIP
    ...     open(f'{dir}/hello', 'w').write('hello hail')
    ...     print(open(f'{dir}/hello').read())
    hello hail

    Returns
    -------
    :class:`._TemporaryDirectoryManager`

    """
    if dir is None:
        dir = tmp_dir()
    if not dir.endswith('/'):
        dir = dir + '/'
    dirname = dir + prefix + secret_alnum_string(10) + suffix
    fs = current_backend().fs
    if ensure_exists:
        fs.mkdir(dirname)
    return _TemporaryDirectoryManager(fs, dirname)


def current_backend() -> Backend:
    return Env.hc()._backend


async def _async_current_backend() -> Backend:
    return (await Env._async_hc())._backend


def default_reference():
    """Returns the default reference genome ``'GRCh37'``.

    Returns
    -------
    :class:`.ReferenceGenome`
    """
    return Env.hc().default_reference


def get_reference(name) -> ReferenceGenome:
    """Returns the reference genome corresponding to `name`.

    Notes
    -----

    Hail's built-in references are ``'GRCh37'``, ``GRCh38'``, ``'GRCm38'``, and
    ``'CanFam3'``.
    The contig names and lengths come from the GATK resource bundle:
    `human_g1k_v37.dict
    <ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/b37/human_g1k_v37.dict>`__
    and `Homo_sapiens_assembly38.dict
    <ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/hg38/Homo_sapiens_assembly38.dict>`__.


    If ``name='default'``, the value of :func:`.default_reference` is returned.

    Parameters
    ----------
    name : :class:`str`
        Name of a previously loaded reference genome or one of Hail's built-in
        references: ``'GRCh37'``, ``'GRCh38'``, ``'GRCm38'``, ``'CanFam3'``, and
        ``'default'``.

    Returns
    -------
    :class:`.ReferenceGenome`
    """
    Env.hc()
    if name == 'default':
        return default_reference()
    else:
        return Env.backend().get_reference(name)


@typecheck(seed=int)
def set_global_seed(seed):
    """Deprecated.

    Has no effect. To ensure reproducible randomness, use the `global_seed`
    argument to :func:`.init` and :func:`.reset_global_randomness`.

    See the :ref:`random functions <sec-random-functions>` reference docs for more.

    Parameters
    ----------
    seed : :obj:`int`
        Integer used to seed Hail's random number generator
    """

    warning('hl.set_global_seed has no effect. See '
            'https://hail.is/docs/0.2/functions/random.html for details on '
            'ensuring reproducibility of randomness.')
    pass


@typecheck()
def reset_global_randomness():
    """Restore global randomness to initial state for test reproducibility.
    """

    Env.reset_global_randomness()


def _set_flags(**flags):
    Env.backend().set_flags(**flags)


def _get_flags(*flags):
    return Env.backend().get_flags(*flags)


@contextmanager
def _with_flags(**flags):
    before = _get_flags(*flags)
    try:
        _set_flags(**flags)
        yield
    finally:
        _set_flags(**before)


def debug_info():
    from hail.backend.spark_backend import SparkBackend
    from hail.backend.backend import local_jar_information
    spark_conf = None
    if isinstance(Env.backend(), SparkBackend):
        spark_conf = spark_context()._conf.getAll()
    return {
        'spark_conf': spark_conf,
        'local_jar_information': local_jar_information(),
        'version': version()
    }
