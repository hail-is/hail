import os
from typing import Any

import pytest
from pyspark import SparkConf, SparkContext

import hail as hl
from hail.backend.spark_backend import _configure_spark_classpath, _get_or_create_pyspark_session
from hail.utils.java import Env
from test.hail.helpers import hl_init_for_test

pytestmark = [pytest.mark.backend('spark'), pytest.mark.uninitialized]


def fatal(typ: hl.HailType, msg: str = "") -> hl.Expression:
    return hl.construct_expr(hl.ir.Die(hl.to_expr(msg, hl.tstr)._ir, typ), typ)


def prune(kvs: dict[str, Any | None]) -> dict[str, Any]:
    return {k: v for k, v in kvs.items() if v is not None}


@pytest.mark.parametrize('copy', [True, False])
def test_copy_spark_log(tmpdir, copy):
    hl.init(copy_log_on_error=copy, tmp_dir=str(tmpdir))

    expr = fatal(hl.tint32)
    with pytest.raises(Exception):
        hl.eval(expr)

    hc = Env.hc()
    _, filename = os.path.split(hc._log)
    log = os.path.join(hc._tmpdir, filename)

    assert Env.fs().exists(log) == copy


def test_init_without_existing_spark_context(request):
    hl_init_for_test(app_name=request.node.name, quiet=True)
    sc = Env.spark_session().sparkContext
    hl.stop()

    assert getattr(sc, '_jsc', None) is None, 'SparkBackend.close() did not stop the SparkContext.'


def test_init_with_existing_spark_context(request):
    """
    Simulate sharing a spark context across notebook sessions.
    The first notebook successfully initialised hail.
    The second re-uses the same context.
    """

    session = _get_or_create_pyspark_session(None, app_name=request.node.name, master='local[1]', show_progress=False)
    try:
        sc = session.sparkContext
        hl_init_for_test(sc=sc, quiet=True)
        assert sc == Env.spark_session().sparkContext
        hl.utils.range_table(10)._force_count()
        hl.stop()

        assert (jsc := getattr(sc, '_jsc', None)) is not None
        assert not jsc.sc().isStopped(), 'SparkBackend.close() should not stop a SparkContext it does not own.'
    finally:
        session.stop()
        with SparkContext._lock:
            SparkContext._gateway.shutdown()
            SparkContext._gateway = None
            SparkContext._jvm = None


@pytest.fixture(scope='class')
def jvm_gateway():
    conf = SparkConf(loadDefaults=False)
    _configure_spark_classpath(conf)
    SparkContext._ensure_initialized(conf=conf)
    try:
        yield
    finally:
        with SparkContext._lock:
            SparkContext._gateway.shutdown()
            SparkContext._gateway = None
            SparkContext._jvm = None


class TestSparkConf:
    cases = (
        ['spark.app.name', 'app_name', None, None, 'Hail'],  # default
        ['spark.app.name', 'app_name', 'Any', None, 'Hail'],
        ['spark.app.name', 'app_name', 'Any', 'Foo', 'Foo'],
        ['spark.master', 'master', None, None, 'local[*]'],  # default
        ['spark.master', 'master', None, 'local', 'local'],
        ['spark.master', 'master', 'local', None, 'local'],
        ['spark.master', 'master', 'local[*]', 'local', 'local'],
        ['spark.local.dir', 'local_tmpdir', None, None, None],  # default
        ['spark.local.dir', 'local_tmpdir', '/tmp', None, '/tmp'],
        ['spark.local.dir', 'local_tmpdir', None, '/tmp', '/tmp'],
        ['spark.local.dir', 'local_tmpdir', '/dev/null', '/tmp', '/tmp'],
        ['spark.hadoop.mapreduce.input.fileinputformat.split.minsize', 'min_block_size', None, None, None],  # default
        ['spark.hadoop.mapreduce.input.fileinputformat.split.minsize', 'min_block_size', 1, None, '1'],
        ['spark.hadoop.mapreduce.input.fileinputformat.split.minsize', 'min_block_size', None, 1, str(1 * 1024 * 1024)],
        ['spark.hadoop.mapreduce.input.fileinputformat.split.minsize', 'min_block_size', 1, 2, str(2 * 1024 * 1024)],
        ['spark.ui.showConsoleProgress', 'show_progress', None, None, 'true'],  # pyspark default
        ['spark.ui.showConsoleProgress', 'show_progress', 'false', None, 'false'],
        ['spark.ui.showConsoleProgress', 'show_progress', None, 'false', 'false'],
        ['spark.ui.showConsoleProgress', 'show_progress', 'true', 'false', 'false'],
    )

    @pytest.mark.usefixtures('jvm_gateway')
    @pytest.mark.parametrize('key,param,cvalue,arg,expected', cases)
    def test_(self, key: str, param: str, cvalue: str | None, arg: Any | None, expected: str):
        session = _get_or_create_pyspark_session(sc=None, spark_conf=prune({key: cvalue}), **prune({param: arg}))
        try:
            conf = session.sparkContext.getConf()
            assert conf.get(key) == expected
        finally:
            session.stop()


def test_min_block_size_pos_int():
    with pytest.raises(ValueError):
        hl.init(min_block_size=-1)
