from pyspark import SparkContext
from pyspark.sql import SQLContext

from hail.genetics.genomeref import GenomeReference
from hail.typecheck import nullable, strlike, integral, typecheck, typecheck_method
from hail.utils import wrap_to_list, get_env_or_default
from hail.utils.java import Env, joption, FatalError, connect_logger

import sys


class HailContext(object):
    @typecheck_method(sc=nullable(SparkContext),
                      app_name=strlike,
                      master=nullable(strlike),
                      local=strlike,
                      log=strlike,
                      quiet=bool,
                      append=bool,
                      min_block_size=integral,
                      branching_factor=integral,
                      tmp_dir=nullable(strlike),
                      default_reference=strlike)
    def __init__(self, sc=None, app_name="Hail", master=None, local='local[*]',
                 log='hail.log', quiet=False, append=False,
                 min_block_size=1, branching_factor=50, tmp_dir=None,
                 default_reference="GRCh37"):

        if Env._hc:
            raise FatalError('Hail Context has already been created, restart session '
                             'or stop Hail context to change configuration.')

        SparkContext._ensure_initialized()

        self._gateway = SparkContext._gateway
        self._jvm = SparkContext._jvm

        # hail package
        self._hail = getattr(self._jvm, 'is').hail

        Env._jvm = self._jvm
        Env._gateway = self._gateway

        jsc = sc._jsc.sc() if sc else None

        tmp_dir = get_env_or_default(tmp_dir, 'TMPDIR', '/tmp')

        # we always pass 'quiet' to the JVM because stderr output needs
        # to be routed through Python separately.
        self._jhc = self._hail.HailContext.apply(
            jsc, app_name, joption(master), local, log, True, append,
            min_block_size, branching_factor, tmp_dir)

        self._jsc = self._jhc.sc()
        self.sc = sc if sc else SparkContext(gateway=self._gateway, jsc=self._jvm.JavaSparkContext(self._jsc))
        self._jsql_context = self._jhc.sqlContext()
        self._sql_context = SQLContext(self.sc, self._jsql_context)
        self._counter = 1

        super(HailContext, self).__init__()

        # do this at the end in case something errors, so we don't raise the above error without a real HC
        Env._hc = self

        self._default_ref = None
        Env.hail().variant.GenomeReference.setDefaultReference(self._jhc, default_reference)

        sys.stderr.write('Running on Apache Spark version {}\n'.format(self.sc.version))
        if self._jsc.uiWebUrl().isDefined():
            sys.stderr.write('SparkUI available at {}\n'.format(self._jsc.uiWebUrl().get()))

        if not quiet:
            connect_logger('localhost', 12888)

        sys.stderr.write(
            'Welcome to\n'
            '     __  __     <>__\n'
            '    / /_/ /__  __/ /\n'
            '   / __  / _ `/ / /\n'
            '  /_/ /_/\_,_/_/_/   version {}\n'.format(self.version))

        if self.version.startswith('devel'):
            sys.stderr.write('WARNING: This is an unstable development build.\n')

    @property
    def version(self):
        return self._jhc.version()

    @property
    def default_reference(self):
        if not self._default_ref:
            self._default_ref = GenomeReference._from_java(Env.hail().variant.GenomeReference.defaultReference())
        return self._default_ref

    def stop(self):
        self.sc.stop()
        self.sc = None
        Env._jvm = None
        Env._gateway = None
        Env._hc = None

@typecheck(sc=nullable(SparkContext),
           app_name=strlike,
           master=nullable(strlike),
           local=strlike,
           log=strlike,
           quiet=bool,
           append=bool,
           min_block_size=integral,
           branching_factor=integral,
           tmp_dir=strlike,
           default_reference=strlike)
def init(sc=None, app_name="Hail", master=None, local='local[*]',
             log='hail.log', quiet=False, append=False,
             min_block_size=1, branching_factor=50, tmp_dir='/tmp',
             default_reference="GRCh37"):
    HailContext(sc, app_name, master, local, log, quiet, append, min_block_size, branching_factor, tmp_dir, default_reference)

def stop():
    """Stop the currently running HailContext."""
    Env.hc().stop()

def default_reference():
    """Return the default reference genome.

    Returns
    -------
    :class:`.GenomeReference`
    """
    return Env.hc().default_reference
