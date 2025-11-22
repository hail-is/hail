import os
import sys
from collections.abc import Callable
from typing import Any, Dict, Optional

import orjson
import pyspark
import pyspark.sql
from py4j.java_gateway import JavaGateway

from hail.expr.table_type import ttable
from hail.ir import BaseIR
from hail.table import Table
from hail.utils import copy_log
from hail.utils.java import scala_package_object
from hail.version import __version__
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration
from hailtop.aiotools.validators import validate_file
from hailtop.fs.fs import FS
from hailtop.fs.router_fs import RouterFS
from hailtop.utils import async_to_blocking

from ..fs.hadoop_fs import HadoopFS
from .backend import local_jar_information
from .py4j_backend import Py4JBackend, raise_when_mismatched_hail_versions


def _modify_spark_conf(conf: pyspark.SparkConf, key: str, update: Callable[[str | None], str]):
    old = conf.get(key, None)
    conf.set(key, update(old))


def _append_csv(*xs: str) -> Callable[[str | None], str]:
    return lambda csv: ','.join(xs if csv is None else (csv, *xs))


def _get_or_create_pyspark_gateway(
    sc: pyspark.SparkContext | None,
    spark_conf: Dict[str, Any] | None = None,
    local_tmpdir: str | None = None,
    quiet: bool = False,
) -> JavaGateway:
    if sc is not None and not quiet:
        sys.stderr.write(
            'pip-installed Hail requires additional configuration options in Spark referring\n'
            '  to the path to the Hail Python module directory HAIL_DIR,\n'
            '  e.g. /path/to/python/site-packages/hail:\n'
            '    spark.jars=HAIL_DIR/backend/hail-all-spark.jar\n'
            '    spark.driver.extraClassPath=HAIL_DIR/backend/hail-all-spark.jar\n'
            '    spark.executor.extraClassPath=./hail-all-spark.jar'
        )

    conf = pyspark.SparkConf()
    for k, v in (spark_conf or {}).items():
        conf.set(k, v)

    _, hail_jar, extra_classpath = local_jar_information()
    jars = [hail_jar]

    if os.environ.get('HAIL_SPARK_MONITOR') or os.environ.get('AZURE_SPARK') == '1':
        import sparkmonitor

        jars = [*jars, os.path.join(os.path.dirname(sparkmonitor.__file__), 'listener.jar')]
        _modify_spark_conf(
            conf, 'spark.extraListeners', _append_csv('sparkmonitor.listener.JupyterSparkMonitorListener')
        )

    _modify_spark_conf(conf, 'spark.jars', _append_csv(*jars))
    if os.environ.get('AZURE_SPARK') == '1':
        print('AZURE_SPARK environment variable is set to "1", assuming you are in HDInsight.')
        # Setting extraClassPath in HDInsight overrides the classpath entirely so you can't
        # load the Scala standard library. Interestingly, setting extraClassPath is not
        # necessary in HDInsight.
    else:
        _modify_spark_conf(conf, 'spark.driver.extraClassPath', _append_csv(*jars, *extra_classpath))
        _modify_spark_conf(conf, 'spark.executor.extraClassPath', _append_csv(hail_jar, *extra_classpath))

    if local_tmpdir is not None:
        conf.setIfMissing('spark.local.dir', local_tmpdir.removeprefix('file://'))

    conf.set('spark.ui.showConsoleProgress', str(not quiet).lower())

    pyspark.SparkContext._ensure_initialized(instance=sc, conf=conf)

    return pyspark.SparkContext._gateway


class SparkBackend(Py4JBackend):
    def __init__(
        self: 'SparkBackend',
        sc: pyspark.SparkContext | None,
        spark_conf: Dict[str, Any] | None,
        app_name: str | None,
        master: str | None,
        local: str | None,
        log: str,
        quiet: bool,
        append: bool,
        min_block_size: int,
        branching_factor: int,
        tmpdir: str,
        local_tmpdir: str,
        skip_logging_configuration: bool,
        *,
        gcs_requester_pays_config: Optional[GCSRequesterPaysConfiguration] = None,
        copy_log_on_error: bool = False,
    ):
        sc = sc or pyspark.SparkContext._active_spark_context
        self._hail_managed_spark = sc is None

        self._gateway = _get_or_create_pyspark_gateway(sc, spark_conf, local_tmpdir, quiet)
        jvm = self._gateway.jvm

        raise_when_mismatched_hail_versions(jvm)

        _is = getattr(jvm, 'is')

        py4jutils = scala_package_object(_is.hail.utils)
        if not skip_logging_configuration:
            py4jutils.configureLogging(log, quiet, append)

        JSparkBackend = _is.hail.backend.spark.SparkBackend
        if sc is not None:
            self._spark = pyspark.sql.SparkSession(sc)
        else:
            jspark_session = JSparkBackend.pySparkSession(app_name, master, local, min_block_size)
            jsc = jvm.JavaSparkContext(jspark_session.sparkContext())
            sc = pyspark.SparkContext(gateway=self._gateway, jsc=jsc)
            self._spark = pyspark.sql.SparkSession(sc, jspark_session)

        if not quiet:
            sys.stderr.write(f'Running on Apache Spark version {self._spark.version}\n')
            if (uiUrl := self._spark.sparkContext.uiWebUrl) is not None:
                sys.stderr.write(f'SparkUI available at {uiUrl}\n')

        flags: Dict[str, str] = {}
        if branching_factor is not None:
            flags['branching_factor'] = str(branching_factor)

        jbackend = JSparkBackend.getOrCreate(self._spark._jsparkSession)
        super().__init__(jvm, jbackend, flags)

        self._fs = None
        self._router_fs = None

        self.remote_tmpdir = tmpdir
        self.local_tmpdir = local_tmpdir
        self.gcs_requester_pays_configuration = gcs_requester_pays_config
        self._copy_log_on_error = copy_log_on_error

        self.logger.info(f'Hail {__version__}')

    def validate_file(self, uri: str) -> None:
        async_to_blocking(validate_file(uri, self.router_fs.afs))

    @property
    def fs(self) -> FS:
        if self._fs is None:
            self._fs = HadoopFS(self._utils_package_object, self._jbackend.pyFs())
        return self._fs

    @fs.setter
    def fs(self, fs: FS | None) -> None:
        assert isinstance(fs, HadoopFS | None)
        self._fs = fs

    @property
    def router_fs(self) -> RouterFS:
        if self._router_fs is None:
            self._router_fs = RouterFS(
                gcs_kwargs={"gcs_requester_pays_configuration": self.gcs_requester_pays_configuration},
            )
        return self._router_fs

    @router_fs.setter
    def router_fs(self, router_fs: RouterFS | None) -> None:
        if self._router_fs is not None:
            self._router_fs.close()

        self._router_fs = router_fs

    @Py4JBackend.gcs_requester_pays_configuration.setter
    def gcs_requester_pays_configuration(self, config: GCSRequesterPaysConfiguration | None):
        Py4JBackend.gcs_requester_pays_configuration.__set__(self, config)
        self.router_fs = None

    def stop(self):
        if self._hail_managed_spark:
            self._spark.stop()
            super().stop()
            self._gateway.shutdown()

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

    def execute(self, ir: BaseIR, timed: bool = False) -> Any:
        try:
            return super().execute(ir, timed)
        except Exception as err:
            if self._copy_log_on_error:
                try:
                    copy_log(self.remote_tmpdir)
                except Exception as fatal:
                    raise err from fatal

            raise err

    @property
    def requires_lowering(self):
        return False
