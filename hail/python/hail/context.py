import pkg_resources
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

import hail
from hail.genetics.reference_genome import ReferenceGenome
from hail.typecheck import nullable, typecheck, typecheck_method, enumeration, dictof
from hail.utils import get_env_or_default
from hail.utils.java import Env, joption, FatalError, connect_logger, install_exception_handler, uninstall_exception_handler
from hail.backend import Backend, ServiceBackend, SparkBackend

import sys
import os


class HailContext(object):
    @typecheck_method(sc=nullable(SparkContext),
                      app_name=str,
                      master=nullable(str),
                      local=str,
                      log=nullable(str),
                      quiet=bool,
                      append=bool,
                      min_block_size=int,
                      branching_factor=int,
                      tmp_dir=nullable(str),
                      default_reference=str,
                      idempotent=bool,
                      global_seed=nullable(int),
                      spark_conf=nullable(dictof(str, str)),
                      optimizer_iterations=nullable(int),
                      _backend=nullable(Backend))
    def __init__(self, sc=None, app_name="Hail", master=None, local='local[*]',
                 log=None, quiet=False, append=False,
                 min_block_size=1, branching_factor=50, tmp_dir=None,
                 default_reference="GRCh37", idempotent=False,
                 global_seed=6348563392232659379, spark_conf=None,
                 optimizer_iterations=None, _backend=None):

        if Env._hc:
            if idempotent:
                return
            else:
                raise FatalError('Hail has already been initialized, restart session '
                                 'or stop Hail to change configuration.')

        if pkg_resources.resource_exists(__name__, "hail-all-spark.jar"):
            hail_jar_path = pkg_resources.resource_filename(__name__, "hail-all-spark.jar")
            assert os.path.exists(hail_jar_path), f'{hail_jar_path} does not exist'
            conf = SparkConf()

            base_conf = spark_conf or {}
            for k, v in base_conf.items():
                conf.set(k, v)

            jars = [hail_jar_path]

            if os.environ.get('HAIL_SPARK_MONITOR'):
                import sparkmonitor
                jars.append(os.path.join(os.path.dirname(sparkmonitor.__file__), 'listener.jar'))
                conf.set("spark.extraListeners", "sparkmonitor.listener.JupyterSparkMonitorListener")

            conf.set('spark.jars', ','.join(jars))
            conf.set('spark.driver.extraClassPath', ','.join(jars))
            conf.set('spark.executor.extraClassPath', './hail-all-spark.jar')
            if sc is None:
                SparkContext._ensure_initialized(conf=conf)
            else:
                import warnings
                warnings.warn(
                    'pip-installed Hail requires additional configuration options in Spark referring\n'
                    '  to the path to the Hail Python module directory HAIL_DIR,\n'
                    '  e.g. /path/to/python/site-packages/hail:\n'
                    '    spark.jars=HAIL_DIR/hail-all-spark.jar\n'
                    '    spark.driver.extraClassPath=HAIL_DIR/hail-all-spark.jar\n'
                    '    spark.executor.extraClassPath=./hail-all-spark.jar')
        else:
            SparkContext._ensure_initialized()

        self._gateway = SparkContext._gateway
        self._jvm = SparkContext._jvm

        # hail package
        self._hail = getattr(self._jvm, 'is').hail

        self._warn_cols_order = True
        self._warn_entries_order = True

        Env._jvm = self._jvm
        Env._gateway = self._gateway

        jsc = sc._jsc.sc() if sc else None

        if _backend is None:
            if os.environ.get('HAIL_APISERVER_URL') is not None:
                _backend = ServiceBackend()
            else:
                _backend = SparkBackend()
        self._backend = _backend

        tmp_dir = get_env_or_default(tmp_dir, 'TMPDIR', '/tmp')
        optimizer_iterations = get_env_or_default(optimizer_iterations, 'HAIL_OPTIMIZER_ITERATIONS', 3)

        py_version = version()

        if log is None:
            log = hail.utils.timestamp_path(os.path.join(os.getcwd(), 'hail'),
                                            suffix=f'-{py_version}.log')
        self._log = log

        # we always pass 'quiet' to the JVM because stderr output needs
        # to be routed through Python separately.
        # if idempotent:
        if idempotent:
            self._jhc = self._hail.HailContext.getOrCreate(
                jsc, app_name, joption(master), local, log, True, append,
                min_block_size, branching_factor, tmp_dir, optimizer_iterations)
        else:
            self._jhc = self._hail.HailContext.apply(
                jsc, app_name, joption(master), local, log, True, append,
                min_block_size, branching_factor, tmp_dir, optimizer_iterations)

        self._jsc = self._jhc.sc()
        self.sc = sc if sc else SparkContext(gateway=self._gateway, jsc=self._jvm.JavaSparkContext(self._jsc))
        self._jspark_session = self._jhc.sparkSession()
        self._spark_session = SparkSession(self.sc, self._jhc.sparkSession())

        super(HailContext, self).__init__()

        # do this at the end in case something errors, so we don't raise the above error without a real HC
        Env._hc = self

        ReferenceGenome._from_config(_backend.get_reference('GRCh37'), True)
        ReferenceGenome._from_config(_backend.get_reference('GRCh38'), True)
        ReferenceGenome._from_config(_backend.get_reference('GRCm38'), True)

        if default_reference in ReferenceGenome._references:
            self._default_ref = ReferenceGenome._references[default_reference]
        else:
            self._default_ref = ReferenceGenome.read(default_reference)

        jar_version = self._jhc.version()

        if jar_version != py_version:
            raise RuntimeError(f"Hail version mismatch between JAR and Python library\n"
                               f"  JAR:    {jar_version}\n"
                               f"  Python: {py_version}")

        if not quiet:
            sys.stderr.write('Running on Apache Spark version {}\n'.format(self.sc.version))
            if self._jsc.uiWebUrl().isDefined():
                sys.stderr.write('SparkUI available at {}\n'.format(self._jsc.uiWebUrl().get()))

            connect_logger('localhost', 12888)

            self._hail.HailContext.startProgressBar(self._jsc)

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

        install_exception_handler()
        Env.set_seed(global_seed)


    @property
    def default_reference(self):
        return self._default_ref

    def stop(self):
        Env.hail().HailContext.clear()
        self.sc.stop()
        self.sc = None
        Env._jvm = None
        Env._gateway = None
        Env._hc = None
        uninstall_exception_handler()
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
           tmp_dir=str,
           default_reference=enumeration('GRCh37', 'GRCh38', 'GRCm38'),
           idempotent=bool,
           global_seed=nullable(int),
           spark_conf=nullable(dictof(str, str)),
           _optimizer_iterations=nullable(int),
           _backend=nullable(Backend))
def init(sc=None, app_name='Hail', master=None, local='local[*]',
         log=None, quiet=False, append=False,
         min_block_size=0, branching_factor=50, tmp_dir='/tmp',
         default_reference='GRCh37', idempotent=False,
         global_seed=6348563392232659379,
         spark_conf=None,
         _optimizer_iterations=None,
         _backend=None):
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
    app_name : :obj:`str`
        Spark application name.
    master : :obj:`str`, optional
        Spark master.
    local : :obj:`str`
       Local-mode master, used if `master` is not defined here or in the
       Spark configuration.
    log : :obj:`str`
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
    tmp_dir : :obj:`str`
        Temporary directory for Hail files. Must be a network-visible
        file path.
    default_reference : :obj:`str`
        Default reference genome. Either ``'GRCh37'``, ``'GRCh38'``,
        or ``'GRCm38'``.
    idempotent : :obj:`bool`
        If ``True``, calling this function is a no-op if Hail has already been initialized.
    global_seed : :obj:`int`, optional
        Global random seed.
    spark_conf : :obj:`dict[str, str]`, optional
        Spark configuration parameters.
    """
    HailContext(sc, app_name, master, local, log, quiet, append,
                min_block_size, branching_factor, tmp_dir,
                default_reference, idempotent, global_seed, spark_conf,
                _optimizer_iterations,_backend)


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
        Env._hc = None

def spark_context():
    """Returns the active Spark context.

    Returns
    -------
    :class:`pyspark.SparkContext`
    """
    return Env.hc().sc

def current_backend():
    return Env.hc()._backend

def default_reference():
    """Returns the default reference genome ``'GRCh37'``.

    Returns
    -------
    :class:`.ReferenceGenome`
    """
    return Env.hc().default_reference

def get_reference(name) -> 'hail.ReferenceGenome':
    """Returns the reference genome corresponding to `name`.

    Notes
    -----

    Hail's built-in references are ``'GRCh37'``, ``GRCh38'``, and ``'GRCm38'``.
    The contig names and lengths come from the GATK resource bundle:
    `human_g1k_v37.dict
    <ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/b37/human_g1k_v37.dict>`__
    and `Homo_sapiens_assembly38.dict
    <ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/hg38/Homo_sapiens_assembly38.dict>`__.


    If ``name='default'``, the value of :func:`.default_reference` is returned.

    Parameters
    ----------
    name : :obj:`str`
        Name of a previously loaded reference genome or one of Hail's built-in
        references: ``'GRCh37'``, ``'GRCh38'``, ``'GRCm38'``, and ``'default'``.

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
    available = set(Env.hc()._jhc.flags().available())
    invalid = []
    for flag, value in flags.items():
        if flag in available:
            Env.hc()._jhc.flags().set(flag, value)
        else:
            invalid.append(flag)
    if len(invalid) != 0:
        raise FatalError("Flags {} not valid. Valid flags: \n    {}"
                         .format(', '.join(invalid), '\n    '.join(available)))


def _get_flags(*flags):
    return {flag: Env.hc()._jhc.flags().get(flag) for flag in flags}


def debug_info():
    hail_jar_path = None
    if pkg_resources.resource_exists(__name__, "hail-all-spark.jar"):
        hail_jar_path = pkg_resources.resource_filename(__name__, "hail-all-spark.jar")
    return {
        'spark_conf': spark_context()._conf.getAll(),
        'hail_jar_path': hail_jar_path,
        'version': version()
    }
