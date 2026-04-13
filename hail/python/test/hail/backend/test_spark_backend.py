import os
from typing import Any

import pytest
from pyspark import SparkContext

import hail as hl
from hail.backend.spark_backend import _get_or_create_pyspark_session
from hail.utils.java import Env
from hailtop.utils import am_i_interactive
from test.hail.helpers import hl_init_for_test

pytestmark = [pytest.mark.backend('spark'), pytest.mark.uninitialized]


def fatal(typ: hl.HailType, msg: str = "") -> hl.Expression:
    return hl.construct_expr(hl.ir.Die(hl.to_expr(msg, hl.tstr)._ir, typ), typ)


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


@pytest.mark.parametrize('show_progress', [True, False])
def test_show_console_progress_bar(show_progress):
    hl.init(show_progress=show_progress)
    conf = hl.spark_context().getConf()
    assert conf.get('spark.ui.showConsoleProgress') == str(show_progress).lower()


@pytest.mark.parametrize(
    'cname,name,default',
    [
        ['spark.app.name', 'app_name', 'Hail'],
        ['spark.master', 'master', 'local[*]'],
        ['spark.master', 'local', 'local[*]'],
        ['spark.local.dir', 'local_tmpdir', '/tmp'],
        ['spark.hadoop.mapreduce.input.fileinputformat.split.minsize', 'min_block_size', '0'],
        ['spark.ui.showConsoleProgress', 'show_progress', str(am_i_interactive()).lower()],
    ],
)
def test_default_spark_conf(cname: str, name: str, default: str):
    hl.init(**{name: None})
    conf = hl.spark_context().getConf()
    assert conf.get(cname) == default


@pytest.mark.parametrize(
    'cname,cvalue,name,value',
    [
        ['spark.app.name', 'geoff', 'app_name', 'steve'],
        ['spark.master', 'local[1]', 'master', 'local[*]'],
        ['spark.master', 'local[1]', 'local', 'local[*]'],
        ['spark.hadoop.mapreduce.input.fileinputformat.split.minsize', '1', 'min_block_size', 0],
        ['spark.ui.showConsoleProgress', 'false', 'show_progress', True],
    ],
)
def test_spark_conf_takes_precedence(cname: str, cvalue: str, name: str, value: Any):
    hl.init(spark_conf={cname: cvalue}, **{name: value})
    conf = hl.spark_context().getConf()
    assert conf.get(cname) == cvalue


# you can set the spark local dir once per spark context, but you can
# set hail's temporary directory at any time
def test_spark_conf_takes_precedence_local_dir_special_case(tmp_path):
    hl.init(spark_conf={'spark.local.dir': str(tmp_path)}, local_tmpdir='/does/not/exist')
    conf = hl.spark_context().getConf()
    assert conf.get('spark.local.dir') == str(tmp_path)

    backend = hl.current_backend()
    assert backend.local_tmpdir == 'file:///does/not/exist'

    backend.local_tmpdir = '/dev/null'
    assert backend.local_tmpdir == '/dev/null'


def test_min_block_size_pos_int():
    with pytest.raises(ValueError):
        hl.init(min_block_size=-1)
