import os

import pyspark
import pytest

import hail as hl
from hail.backend.spark_backend import _get_or_create_pyspark_gateway
from hail.utils.java import Env
from test.hail.helpers import hl_init_for_test


def fatal(typ: hl.HailType, msg: str = "") -> hl.Expression:
    return hl.construct_expr(hl.ir.Die(hl.to_expr(msg, hl.tstr)._ir, typ), typ)


@pytest.mark.uninitialized
@pytest.mark.backend('spark')
@pytest.mark.parametrize('copy', [True, False])
def test_copy_spark_log(tmpdir, copy):
    hl.init(copy_spark_log_on_error=copy, tmp_dir=str(tmpdir))

    expr = fatal(hl.tint32)
    with pytest.raises(Exception):
        hl.eval(expr)

    hc = Env.hc()
    _, filename = os.path.split(hc._log)
    log = os.path.join(hc._tmpdir, filename)

    assert Env.fs().exists(log) == copy


@pytest.mark.backend('spark')
@pytest.mark.uninitialized
def test_init_without_existing_spark_context(request):
    hl_init_for_test(app_name=request.node.name, quiet=True)
    sc = Env.spark_session().sparkContext
    hl.stop()

    assert getattr(sc, '_jsc', None) is None, 'SparkBackend.close() did not stop the SparkContext.'


@pytest.mark.backend('spark')
@pytest.mark.uninitialized
def test_init_with_existing_spark_context(request):
    """
    Simulate sharing a spark context across notebook sessions.
    The first notebook successfully initialised hail.
    The second re-uses the same context.
    """

    gateway = _get_or_create_pyspark_gateway(None, None, quiet=True)
    JCons = getattr(gateway.jvm, 'is').hail.backend.spark.SparkBackend
    jspark = JCons.pySparkSession(request.node.name, 'local[1]', None, 0)
    sc = pyspark.SparkContext(gateway=gateway, jsc=gateway.jvm.JavaSparkContext(jspark.sparkContext()))

    try:
        hl_init_for_test(sc=sc, quiet=True)
        assert sc == Env.spark_session().sparkContext

        hl.stop()

        assert (jsc := getattr(sc, '_jsc', None)) is not None
        assert not jsc.sc().isStopped(), 'SparkBackend.close() should not stop a SparkContext it does not own.'
    finally:
        sc.stop()
        gateway.close()
