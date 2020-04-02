import pkg_resources
from pyspark import SparkContext

import hail
from hail.genetics.reference_genome import ReferenceGenome
from hail.typecheck import nullable, typecheck, typecheck_method, enumeration, dictof
from hail.utils import get_env_or_default
from hail.utils.java import Env, joption, FatalError, connect_logger, install_exception_handler, uninstall_exception_handler, warn
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
                warn('Hail has already been initialized. If this call was intended to change configuration,'
                     ' close the session with hl.stop() first.')

        py_version = version()
        
        if log is None:
            log = hail.utils.timestamp_path(os.path.join(os.getcwd(), 'hail'),
                                            suffix=f'-{py_version}.log')
        self._log = log

        tmp_dir = get_env_or_default(tmp_dir, 'TMPDIR', '/tmp')
        self.tmp_dir = tmp_dir

        optimizer_iterations = get_env_or_default(optimizer_iterations, 'HAIL_OPTIMIZER_ITERATIONS', 3)

        if _backend is None:
            if os.environ.get('HAIL_APISERVER_URL') is not None:
                _backend = ServiceBackend()
            else:
                _backend = SparkBackend(
                    idempotent, sc, spark_conf, app_name, master, local, log,
                    quiet, append, min_block_size, branching_factor, tmp_dir,
                    optimizer_iterations)
        self._backend = _backend

        self._warn_cols_order = True
        self._warn_entries_order = True

        super(HailContext, self).__init__()

        # do this at the end in case something errors, so we don't raise the above error without a real HC
        Env._hc = self

        ReferenceGenome._from_config(_backend.get_reference('GRCh37'), True)
        ReferenceGenome._from_config(_backend.get_reference('GRCh38'), True)
        ReferenceGenome._from_config(_backend.get_reference('GRCm38'), True)
        ReferenceGenome._from_config(_backend.get_reference('CanFam3'), True)

        if default_reference in ReferenceGenome._references:
            self._default_ref = ReferenceGenome._references[default_reference]
        else:
            self._default_ref = ReferenceGenome.read(default_reference)

        if not quiet:
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
        self._backend.stop()
        self._backend = None
        Env._jvm = None
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
           default_reference=enumeration('GRCh37', 'GRCh38', 'GRCm38', 'CanFam3'),
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
        ``'GRCm38'``, or ``'CanFam3'``.
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

def spark_context():
    """Returns the active Spark context.

    Returns
    -------
    :class:`pyspark.SparkContext`
    """
    return Env.backend().sc

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
    name : :obj:`str`
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
