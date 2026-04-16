import os
import sys
import warnings
from collections.abc import Callable

import orjson
import pyspark
import pyspark.sql
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from hail.expr.table_type import ttable
from hail.table import Table
from hail.utils import maybe
from hail.utils.java import scala_object
from hail.version import __version__
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration
from hailtop.aiotools.validators import validate_file
from hailtop.config import ConfigVariable, configuration_of
from hailtop.fs.fs import FS
from hailtop.fs.router_fs import RouterFS
from hailtop.utils import async_to_blocking

from ..fs.hadoop_fs import HadoopFS
from .backend import local_jar_information
from .py4j_backend import Py4JBackend, raise_when_mismatched_hail_versions


def _modify_spark_conf(conf: pyspark.SparkConf, key: str, update: Callable[[str | None], str]):
    old = conf.get(key, None)
    conf.set(key, update(old))


def _append_delimited(*xs: str, sep: str) -> Callable[[str | None], str]:
    return lambda csv: sep.join(xs if csv is None else (csv, *xs))


def _configure_spark_classpath(conf: SparkConf):
    info = local_jar_information()
    classpath = [info.hail_jar, *info.extra_classpath]

    if os.environ.get('HAIL_SPARK_MONITOR') or os.environ.get('AZURE_SPARK') == '1':
        import sparkmonitor

        classpath.append(os.path.join(os.path.dirname(sparkmonitor.__file__), 'listener.jar'))
        _modify_spark_conf(
            conf,
            'spark.extraListeners',
            _append_delimited('sparkmonitor.listener.JupyterSparkMonitorListener', sep=','),
        )

    spark_jars = [path for path in classpath if os.path.splitext(path)[1] == '.jar']
    _modify_spark_conf(conf, 'spark.jars', _append_delimited(*spark_jars, sep=','))
    if os.environ.get('AZURE_SPARK') == '1':
        print('AZURE_SPARK environment variable is set to "1", assuming you are in HDInsight.')
        # Setting extraClassPath in HDInsight overrides the classpath entirely so you can't
        # load the Scala standard library. Interestingly, setting extraClassPath is not
        # necessary in HDInsight.
    else:
        _modify_spark_conf(conf, 'spark.driver.extraClassPath', _append_delimited(*classpath, sep=':'))
        _modify_spark_conf(conf, 'spark.executor.extraClassPath', _append_delimited(*classpath, sep=':'))


def _get_or_create_pyspark_session(
    sc: SparkContext | None,
    *,
    spark_conf: dict[str, str] | None = None,
    app_name: str | None = None,
    master: str | None = None,
    local_tmpdir: str | None = None,
    min_block_size: int | None = None,
    show_progress: bool | None = None,
) -> SparkSession:
    if sc is not None:
        # We can't modify many conf parameters in an existing SparkContext
        warnings.warn(
            'Hail requires additional configuration options in Spark referring\n'
            '  to the path to the Hail Python module directory HAIL_DIR,\n'
            '  e.g. /path/to/python/site-packages/hail:\n'
            '    spark.jars=HAIL_DIR/backend/hail-all-spark.jar\n'
            '    spark.driver.extraClassPath=HAIL_DIR/backend/hail-all-spark.jar\n'
            '    spark.executor.extraClassPath=./hail-all-spark.jar'
        )
        return SparkSession(sc)

    # How we source and apply sparkconf depends on the execution environment.
    #
    # If hail is running in a python shell then we need to configure the driver
    # and worker classpath before starting the jvm. Note that defaults are read
    # after jvm initialisation and so specifying `loadDefaults=True` here has
    # no effect.
    #
    # If hail is running as a spark-submit job within a managed spark service
    # like dataproc, then the jvm has already been started and conf passed to
    # `_ensure_initialized` is ignored.
    #
    # Once the jvm has been initialised, we can load default values and apply
    # configuration defined in python and scala. While we don't ship a
    # spark-defaults.conf file with hail, users are free to supply and edit
    # one. This is the mechanism used by the install_gcs_connector script to
    # configure hadoop fs auth.
    conf = SparkConf(loadDefaults=False).setAll(list((spark_conf or dict()).items()))
    _configure_spark_classpath(conf)
    SparkContext._ensure_initialized(conf=conf)

    jvm = SparkContext._jvm
    raise_when_mismatched_hail_versions(jvm)
    JBackend = scala_object(getattr(jvm, 'is').hail.backend.spark, 'SparkBackend')

    conf = (
        SparkConf(loadDefaults=True)
        .setAll(SparkConf(_jconf=JBackend.pySparkConf()).getAll())
        .setAll(conf.getAll())
        .setAppName(app_name if app_name is not None else 'Hail')
    )

    # It's important that we allow users to overwrite master, but we should use
    # the default when it exists. For example, pyspark has no default so we
    # should use 'local[*]'. Dataproc defines 'yarn' as the default master; we
    # should use this.
    conf.setMaster(master if master is not None else conf.get('spark.master', 'local[*]'))

    if local_tmpdir is not None:
        conf.set('spark.local.dir', local_tmpdir.removeprefix('file://'))

    if show_progress is not None:
        conf.set('spark.ui.showConsoleProgress', str(show_progress).lower())

    if min_block_size is not None:
        if min_block_size < 0:
            raise ValueError('`min_block_size` cannot be negative')

        conf.set(
            'spark.hadoop.mapreduce.input.fileinputformat.split.minsize',
            str(min_block_size * 1024 * 1024),
        )

    return SparkSession.Builder().config(conf=conf).getOrCreate()


class SparkBackend(Py4JBackend):
    def __init__(
        self: 'SparkBackend',
        sc: pyspark.SparkContext | None,
        spark_conf: dict[str, str] | None,
        app_name: str | None,
        master: str | None,
        local: str | None,
        log: str,
        quiet: bool,
        append: bool,
        show_progress: bool | None,
        min_block_size: int | None,
        branching_factor: int,
        tmpdir: str,
        local_tmpdir: str,
        skip_logging_configuration: bool,
        *,
        requester_pays_config: GCSRequesterPaysConfiguration | None = None,
        copy_log_on_error: bool = False,
    ):
        sc = sc or pyspark.SparkContext._active_spark_context
        self._hail_managed_spark = sc is None

        if show_progress is None:
            str_value = configuration_of(ConfigVariable.QUERY_DISABLE_PROGRESS_BAR, None, None)
            show_progress = maybe(lambda v: v == '0', str_value)

        self._spark = _get_or_create_pyspark_session(
            sc,
            spark_conf=spark_conf,
            app_name=app_name,
            master=master if master is not None else local,
            local_tmpdir=local_tmpdir,
            show_progress=show_progress,
            min_block_size=min_block_size,
        )

        jvm = self._spark._jvm
        _is = getattr(jvm, 'is')

        if not skip_logging_configuration:
            py4jutils = scala_object(_is.hail.utils, 'py4jutils')
            py4jutils.pyConfigureLogging(log, quiet, append)

        if not quiet:
            sys.stderr.write(f'Running on Apache Spark version {self._spark.version}\n')
            if (uiUrl := self._spark.sparkContext.uiWebUrl) is not None:
                sys.stderr.write(f'SparkUI available at {uiUrl}\n')

        flags: dict[str, str] = {}
        if branching_factor is not None:
            flags['branching_factor'] = str(branching_factor)

        JSparkBackend = _is.hail.backend.spark.SparkBackend
        jbackend = JSparkBackend.getOrCreate(self._spark._jsparkSession)
        super().__init__(jvm, jbackend, flags, copy_log_on_error)

        self._fs = None
        self._router_fs = None

        self.remote_tmpdir = tmpdir
        self.local_tmpdir = local_tmpdir
        self.requester_pays_config = requester_pays_config
        self.logger.info(f'Hail {__version__}')

    def validate_file(self, uri: str) -> None:
        async_to_blocking(validate_file(uri, self.router_fs.afs))

    @property
    def fs(self) -> FS:
        if self._fs is None:
            self._fs = HadoopFS(self._py4jutils, self._jbackend.pyFs())
        return self._fs

    @fs.setter
    def fs(self, fs: FS | None) -> None:
        assert isinstance(fs, HadoopFS | None)
        self._fs = fs

    @property
    def router_fs(self) -> RouterFS:
        if self._router_fs is None:
            self._router_fs = RouterFS(
                gcs_kwargs={"gcs_requester_pays_configuration": self.requester_pays_config},
            )
        return self._router_fs

    @router_fs.setter
    def router_fs(self, router_fs: RouterFS | None) -> None:
        if self._router_fs is not None:
            self._router_fs.close()

        self._router_fs = router_fs

    @Py4JBackend.requester_pays_config.setter
    def requester_pays_config(self, config: GCSRequesterPaysConfiguration | None):
        Py4JBackend.requester_pays_config.__set__(self, config)
        self.router_fs = None

    def stop(self):
        if self._hail_managed_spark:
            self._spark.stop()
            super().stop()
            SparkContext._gateway.shutdown()

            # clean up pyspark's global state to support
            # re-init with different spark conf
            with pyspark.SparkContext._lock:
                pyspark.SparkContext._gateway = None
                pyspark.SparkContext._jvm = None
        else:
            super().stop()

        self.router_fs = None

    def from_spark(self, df, key):
        result_tuple = self._jbackend.pyFromDF(df._jdf, key)
        tir_id, type_json = result_tuple._1(), result_tuple._2()
        return Table._from_java(ttable._from_json(orjson.loads(type_json)), tir_id)

    def to_spark(self, t, flatten):
        t = t.expand_types()
        if flatten:
            t = t.flatten()
        return pyspark.sql.DataFrame(self._jbackend.pyToDF(self._render_ir(t._tir)), self._spark)

    @property
    def requires_lowering(self):
        return any(self.get_flags('lower', 'lower_bm').values())
