from typing import Optional
import sys
import os
from urllib.parse import urlparse, urlunparse

import pkg_resources
from pyspark import SparkContext

import hail
from hail.genetics.reference_genome import ReferenceGenome
from hail.typecheck import nullable, typecheck, typecheck_method, enumeration, dictof
from hail.utils import get_env_or_default
from hail.utils.java import Env, FatalError, warning
from hail.backend import Backend
from hailtop.utils import secret_alnum_string
from .fs.fs import FS


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
        log = hail.utils.timestamp_path(os.path.join(os.getcwd(), 'hail'),
                                        suffix=f'-{py_version}.log')
    return log


class HailContext(object):
    @typecheck_method(log=str,
                      quiet=bool,
                      append=bool,
                      tmpdir=str,
                      local_tmpdir=str,
                      default_reference=str,
                      global_seed=nullable(int),
                      backend=Backend)
    def __init__(self, log, quiet, append, tmpdir, local_tmpdir,
                 default_reference, global_seed, backend):
        assert not Env._hc

        super(HailContext, self).__init__()

        self._log = log

        self._tmpdir = tmpdir
        self._local_tmpdir = local_tmpdir

        self._backend = backend

        self._warn_cols_order = True
        self._warn_entries_order = True

        Env._hc = self

        ReferenceGenome._from_config(self._backend.get_reference('GRCh37'), True)
        ReferenceGenome._from_config(self._backend.get_reference('GRCh38'), True)
        ReferenceGenome._from_config(self._backend.get_reference('GRCm38'), True)
        ReferenceGenome._from_config(self._backend.get_reference('CanFam3'), True)

        if default_reference in ReferenceGenome._references:
            self._default_ref = ReferenceGenome._references[default_reference]
        else:
            self._default_ref = ReferenceGenome.read(default_reference)

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

        if global_seed is None:
            global_seed = 6348563392232659379
        Env.set_seed(global_seed)

    @property
    def default_reference(self):
        return self._default_ref

    def stop(self):
        self._backend.stop()
        self._backend = None
        Env._hc = None
        Env._dummy_table = None
        Env._seed_generator = None
        hail.ir.clear_session_functions()
        ReferenceGenome._references = {}


@typecheck(sc=nullable(SparkContext),
           app_name=str,
           master=nullable(str),
           local=str,
           log=nullable(str),
           quiet=bool,
           append=bool,
           min_block_size=int,
           branching_factor=int,
           tmp_dir=nullable(str),
           default_reference=enumeration('GRCh37', 'GRCh38', 'GRCm38', 'CanFam3'),
           idempotent=bool,
           global_seed=nullable(int),
           spark_conf=nullable(dictof(str, str)),
           skip_logging_configuration=bool,
           local_tmpdir=nullable(str),
           _optimizer_iterations=nullable(int))
def init(sc=None, app_name='Hail', master=None, local='local[*]',
         log=None, quiet=False, append=False,
         min_block_size=0, branching_factor=50, tmp_dir=None,
         default_reference='GRCh37', idempotent=False,
         global_seed=6348563392232659379,
         spark_conf=None,
         skip_logging_configuration=False,
         local_tmpdir=None,
         _optimizer_iterations=None):
    """Initialize Hail and Spark.

    Examples
    --------
    Import and initialize Hail using GRCh38 as the default reference genome:

    >>> import hail as hl
    >>> hl.init(default_reference='GRCh38')  # doctest: +SKIP

    Notes
    -----
    Hail is not only a Python library; most of Hail is written in Java/Scala
    and runs together with Apache Spark in the Java Virtual Machine (JVM).
    In order to use Hail, a JVM needs to run as well. The :func:`.init`
    function is used to initialize Hail and Spark.

    This function also sets global configuration parameters used for the Hail
    session, like the default reference genome and log file location.

    This function will be called automatically (with default parameters) if
    any Hail functionality requiring the backend (most of the libary!) is used.
    To initialize Hail explicitly with non-default arguments, be sure to do so
    directly after importing the module, as in the above example.

    To facilitate the migration from Spark to the ServiceBackend, this method
    calls init_service when the environment variable HAIL_QUERY_BACKEND is set
    to "service".

    Note
    ----
    If a :class:`pyspark.SparkContext` is already running, then Hail must be
    initialized with it as an argument:

    >>> hl.init(sc=sc)  # doctest: +SKIP

    See Also
    --------
    :func:`.stop`

    Parameters
    ----------
    sc : pyspark.SparkContext, optional
        Spark context. By default, a Spark context will be created.
    app_name : :class:`str`
        Spark application name.
    master : :class:`str`, optional
        URL identifying the Spark leader (master) node or `local[N]` for local clusters.
    local : :class:`str`
       Local-mode core limit indicator. Must either be `local[N]` where N is a
       positive integer or `local[*]`. The latter indicates Spark should use all
       cores available. `local[*]` does not respect most containerization CPU
       limits. This option is only used if `master` is unset and `spark.master`
       is not set in the Spark configuration.
    log : :class:`str`
        Local path for Hail log file. Does not currently support distributed
        file systems like Google Storage, S3, or HDFS.
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
        Spark configuration parameters.
    skip_logging_configuration : :obj:`bool`
        Skip logging configuration in java and python.
    local_tmpdir : :class:`str`, optional
        Local temporary directory.  Used on driver and executor nodes.
        Must use the file scheme.  Defaults to TMPDIR, or /tmp.
    """
    if Env._hc:
        if idempotent:
            return
        else:
            warning('Hail has already been initialized. If this call was intended to change configuration,'
                    ' close the session with hl.stop() first.')

    if os.environ.get('HAIL_QUERY_BACKEND') == 'service':
        return init_service(
            log=log,
            quiet=quiet,
            append=append,
            tmpdir=tmp_dir,
            local_tmpdir=local_tmpdir,
            default_reference=default_reference,
            global_seed=global_seed,
            skip_logging_configuration=skip_logging_configuration)

    from hail.backend.spark_backend import SparkBackend

    log = _get_log(log)
    tmpdir = _get_tmpdir(tmp_dir)
    local_tmpdir = _get_local_tmpdir(local_tmpdir)
    optimizer_iterations = get_env_or_default(_optimizer_iterations, 'HAIL_OPTIMIZER_ITERATIONS', 3)

    backend = SparkBackend(
        idempotent, sc, spark_conf, app_name, master, local, log,
        quiet, append, min_block_size, branching_factor, tmpdir, local_tmpdir,
        skip_logging_configuration, optimizer_iterations)

    if not backend.fs.exists(tmpdir):
        backend.fs.mkdir(tmpdir)

    HailContext(
        log, quiet, append, tmpdir, local_tmpdir, default_reference,
        global_seed, backend)


@typecheck(
    billing_project=nullable(str),
    bucket=nullable(str),
    log=nullable(str),
    quiet=bool,
    append=bool,
    tmpdir=nullable(str),
    local_tmpdir=nullable(str),
    default_reference=enumeration('GRCh37', 'GRCh38', 'GRCm38', 'CanFam3'),
    global_seed=nullable(int),
    skip_logging_configuration=bool)
def init_service(
        billing_project: str = None,
        bucket: str = None,
        log=None,
        quiet=False,
        append=False,
        tmpdir=None,
        local_tmpdir=None,
        default_reference='GRCh37',
        global_seed=6348563392232659379,
        skip_logging_configuration=False):
    from hail.backend.service_backend import ServiceBackend
    backend = ServiceBackend(billing_project, bucket, skip_logging_configuration=skip_logging_configuration)

    log = _get_log(log)
    if tmpdir is None:
        tmpdir = 'gs://' + backend._bucket + '/tmp/hail/' + secret_alnum_string()
    assert tmpdir.startswith('gs://')
    local_tmpdir = _get_local_tmpdir(local_tmpdir)

    HailContext(
        log, quiet, append, tmpdir, local_tmpdir, default_reference,
        global_seed, backend)


@typecheck(
    log=nullable(str),
    quiet=bool,
    append=bool,
    branching_factor=int,
    tmpdir=nullable(str),
    default_reference=enumeration('GRCh37', 'GRCh38', 'GRCm38', 'CanFam3'),
    global_seed=nullable(int),
    skip_logging_configuration=bool,
    _optimizer_iterations=nullable(int))
def init_local(
        log=None,
        quiet=False,
        append=False,
        branching_factor=50,
        tmpdir=None,
        default_reference='GRCh37',
        global_seed=6348563392232659379,
        skip_logging_configuration=False,
        _optimizer_iterations=None):
    from hail.backend.local_backend import LocalBackend

    log = _get_log(log)
    tmpdir = _get_tmpdir(tmpdir)
    optimizer_iterations = get_env_or_default(_optimizer_iterations, 'HAIL_OPTIMIZER_ITERATIONS', 3)

    backend = LocalBackend(
        tmpdir, log, quiet, append, branching_factor,
        skip_logging_configuration, optimizer_iterations)

    if not backend.fs.exists(tmpdir):
        backend.fs.mkdir(tmpdir)

    HailContext(
        log, quiet, append, tmpdir, tmpdir, default_reference,
        global_seed, backend)


def version():
    """Get the installed hail version.

    Returns
    -------
    str
    """
    if hail.__version__ is None:
        # https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package
        hail.__version__ = pkg_resources.resource_string(__name__, 'hail_version').decode().strip()
    return hail.__version__


def _hail_cite_url():
    v = version()
    [tag, sha_prefix] = v.split("-")
    if pkg_resources.resource_exists(__name__, "hail-all-spark.jar"):
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
        return self.fs.remove(self.name)


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
    :class:`.DeletingFile` or :class:`.DeletingDirectory`

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
        return self.fs.rmtree(self.name)


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
    :class:`.DeletingFile` or :class:`.DeletingDirectory`

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
        return ReferenceGenome._references[name]


@typecheck(seed=int)
def set_global_seed(seed):
    """Sets Hail's global seed to `seed`.

    Parameters
    ----------
    seed : :obj:`int`
        Integer used to seed Hail's random number generator
    """

    Env.set_seed(seed)


def _set_flags(**flags):
    available = set(Env.backend()._jhc.flags().available())
    invalid = []
    for flag, value in flags.items():
        if flag in available:
            Env.backend()._jhc.flags().set(flag, value)
        else:
            invalid.append(flag)
    if len(invalid) != 0:
        raise FatalError("Flags {} not valid. Valid flags: \n    {}"
                         .format(', '.join(invalid), '\n    '.join(available)))


def _get_flags(*flags):
    return {flag: Env.backend()._jhc.flags().get(flag) for flag in flags}


def debug_info():
    hail_jar_path = None
    if pkg_resources.resource_exists(__name__, "hail-all-spark.jar"):
        hail_jar_path = pkg_resources.resource_filename(__name__, "hail-all-spark.jar")
    return {
        'spark_conf': spark_context()._conf.getAll(),
        'hail_jar_path': hail_jar_path,
        'version': version()
    }
