import os
from test.hail.helpers import skip_unless_spark_backend

import pytest

import hail as hl


def fatal(typ: hl.HailType, msg: str = "") -> hl.Expression:
    return hl.construct_expr(hl.ir.Die(hl.to_expr(msg, hl.tstr)._ir, typ), typ)


@skip_unless_spark_backend()
@pytest.mark.parametrize('copy', [True, False])
def test_copy_spark_log(copy):
    hl.stop()
    hl.init(copy_spark_log_on_error=copy)

    expr = fatal(hl.tint32)
    with pytest.raises(Exception):
        hl.eval(expr)

    from hail.utils.java import Env

    hc = Env.hc()
    _, filename = os.path.split(hc._log)
    log = os.path.join(hc._tmpdir, filename)

    assert Env.fs().exists(log) == copy


@skip_unless_spark_backend()
def test_idempotent_init():
    """
    Simulate sharing a spark context across notebook sessions.
    The first notebook successfully initialised hail.
    The second re-uses the same context and calls init with idempotent=True
    """

    from hail.backend.py4j_backend import uninstall_exception_handler
    from hail.utils.java import Env

    sc = Env.backend().sc

    # Setup globals to simulate new notebook session
    # please don't do this!
    Env._hc = None
    uninstall_exception_handler()

    hl.init(sc, idempotent=True)
