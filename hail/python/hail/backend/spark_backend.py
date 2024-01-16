import os
import sys
from typing import Optional

import orjson
import pyspark
import pyspark.sql

from hail.expr.table_type import ttable
from hail.fs.hadoop_fs import HadoopFS
from hail.ir.renderer import CSERenderer
from hail.table import Table
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.aiotools.validators import validate_file

from .backend import local_jar_information
from .py4j_backend import Py4JBackend


def append_to_comma_separated_list(conf: pyspark.SparkConf, k: str, *new_values: str):
    old = conf.get(k, None)
    if old is None:
        conf.set(k, ','.join(new_values))
    else:
        conf.set(k, old + ',' + ','.join(new_values))


class SparkBackend(Py4JBackend):
    def __init__(
        self,
        idempotent,
        sc,
        spark_conf,
        app_name,
        master,
        local,
        log,
        quiet,
        append,
        min_block_size,
        branching_factor,
        tmpdir,
        local_tmpdir,
        skip_logging_configuration,
        optimizer_iterations,
        *,
        gcs_requester_pays_project: Optional[str] = None,
        gcs_requester_pays_buckets: Optional[str] = None,
    ):
        assert gcs_requester_pays_project is not None or gcs_requester_pays_buckets is None

        try:
            local_jar_info = local_jar_information()
        except ValueError:
            local_jar_info = None

        if local_jar_info is not None:
            conf = pyspark.SparkConf()

            base_conf = spark_conf or {}
            for k, v in base_conf.items():
                conf.set(k, v)

            jars = [local_jar_info.path]
            extra_classpath = local_jar_info.extra_classpath

            if os.environ.get('HAIL_SPARK_MONITOR') or os.environ.get('AZURE_SPARK') == '1':
                import sparkmonitor

                jars.append(os.path.join(os.path.dirname(sparkmonitor.__file__), 'listener.jar'))
                append_to_comma_separated_list(
                    conf, 'spark.extraListeners', 'sparkmonitor.listener.JupyterSparkMonitorListener'
                )

            append_to_comma_separated_list(conf, 'spark.jars', *jars)
            if os.environ.get('AZURE_SPARK') == '1':
                print('AZURE_SPARK environment variable is set to "1", assuming you are in HDInsight.')
                # Setting extraClassPath in HDInsight overrides the classpath entirely so you can't
                # load the Scala standard library. Interestingly, setting extraClassPath is not
                # necessary in HDInsight.
            else:
                append_to_comma_separated_list(conf, 'spark.driver.extraClassPath', *jars, *extra_classpath)
                append_to_comma_separated_list(
                    conf, 'spark.executor.extraClassPath', './hail-all-spark.jar', *extra_classpath
                )

            if sc is None:
                pyspark.SparkContext._ensure_initialized(conf=conf)
            elif not quiet:
                sys.stderr.write(
                    'pip-installed Hail requires additional configuration options in Spark referring\n'
                    '  to the path to the Hail Python module directory HAIL_DIR,\n'
                    '  e.g. /path/to/python/site-packages/hail:\n'
                    '    spark.jars=HAIL_DIR/backend/hail-all-spark.jar\n'
                    '    spark.driver.extraClassPath=HAIL_DIR/backend/hail-all-spark.jar\n'
                    '    spark.executor.extraClassPath=./hail-all-spark.jar'
                )
        else:
            pyspark.SparkContext._ensure_initialized()

        self._gateway = pyspark.SparkContext._gateway
        jvm = pyspark.SparkContext._jvm
        assert jvm

        hail_package = getattr(jvm, 'is').hail
        jsc = sc._jsc.sc() if sc else None

        if idempotent:
            jbackend = hail_package.backend.spark.SparkBackend.getOrCreate(
                jsc,
                app_name,
                master,
                local,
                log,
                True,
                append,
                skip_logging_configuration,
                min_block_size,
                tmpdir,
                local_tmpdir,
                gcs_requester_pays_project,
                gcs_requester_pays_buckets,
            )
            jhc = hail_package.HailContext.getOrCreate(jbackend, branching_factor, optimizer_iterations)
        else:
            jbackend = hail_package.backend.spark.SparkBackend.apply(
                jsc,
                app_name,
                master,
                local,
                log,
                True,
                append,
                skip_logging_configuration,
                min_block_size,
                tmpdir,
                local_tmpdir,
                gcs_requester_pays_project,
                gcs_requester_pays_buckets,
            )
            jhc = hail_package.HailContext.apply(jbackend, branching_factor, optimizer_iterations)

        self._jsc = jbackend.sc()
        if sc:
            self.sc = sc
        else:
            self.sc = pyspark.SparkContext(gateway=self._gateway, jsc=jvm.JavaSparkContext(self._jsc))
        self._jspark_session = jbackend.sparkSession()
        self._spark_session = pyspark.sql.SparkSession(self.sc, self._jspark_session)

        super(SparkBackend, self).__init__(jvm, jbackend, jhc)

        self._fs = None
        self._logger = None

        if not quiet:
            sys.stderr.write('Running on Apache Spark version {}\n'.format(self.sc.version))
            if self._jsc.uiWebUrl().isDefined():
                sys.stderr.write('SparkUI available at {}\n'.format(self._jsc.uiWebUrl().get()))

            jbackend.startProgressBar()

        self._initialize_flags({})

        self._router_async_fs = RouterAsyncFS(
            gcs_kwargs={"gcs_requester_pays_configuration": gcs_requester_pays_project}
        )

    def validate_file(self, uri: str) -> None:
        validate_file(uri, self._router_async_fs)

    def stop(self):
        super().stop()
        self.sc.stop()
        self.sc = None

    @property
    def fs(self):
        if self._fs is None:
            self._fs = HadoopFS(self._utils_package_object, self._jbackend.fs())
        return self._fs

    def from_spark(self, df, key):
        result_tuple = self._jbackend.pyFromDF(df._jdf, key)
        tir_id, type_json = result_tuple._1(), result_tuple._2()
        return Table._from_java(ttable._from_json(orjson.loads(type_json)), tir_id)

    def to_spark(self, t, flatten):
        t = t.expand_types()
        if flatten:
            t = t.flatten()
        return pyspark.sql.DataFrame(self._jbackend.pyToDF(self._render_ir(t._tir)), self._spark_session)

    def register_ir_function(self, name, type_parameters, argument_names, argument_types, return_type, body):
        r = CSERenderer()
        assert not body._ir.uses_randomness
        code = r(body._ir)
        jbody = self._parse_value_ir(code, ref_map=dict(zip(argument_names, argument_types)))
        self._registered_ir_function_names.add(name)

        self.hail_package().expr.ir.functions.IRFunctionRegistry.pyRegisterIR(
            name,
            [ta._parsable_string() for ta in type_parameters],
            argument_names,
            [pt._parsable_string() for pt in argument_types],
            return_type._parsable_string(),
            jbody,
        )

    @property
    def requires_lowering(self):
        return False
