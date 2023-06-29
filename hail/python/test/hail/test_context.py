from typing import Tuple, Dict, Optional
import unittest

import hail as hl
from hail.utils.java import Env
from hail.backend.backend import Backend
from hail.backend.spark_backend import SparkBackend
from test.hail.helpers import skip_unless_spark_backend, hl_init_for_test, hl_stop_for_test


def _scala_map_str_to_tuple_str_str_to_dict(scala) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    it = scala.iterator()
    s: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    while it.hasNext():
        kv = it.next()
        k = kv._1()
        assert isinstance(k, str)
        v = kv._2()
        l = v._1()
        r = v._2()
        assert l is None or isinstance(l, str)
        assert r is None or isinstance(r, str)
        assert k not in s
        s[k] = (l, r)
    return s


def test_init_hail_context_twice():
    hl_init_for_test(idempotent=True)  # Should be no error
    hl_stop_for_test()

    hl_init_for_test(idempotent=True)
    hl.experimental.define_function(lambda x: x + 2, hl.tint32)
    # ensure functions are cleaned up without error
    hl_stop_for_test()

    hl_init_for_test(idempotent=True)  # Should be no error

    if isinstance(Env.backend(), SparkBackend):
        hl_init_for_test(hl.spark_context(), idempotent=True)  # Should be no error


def test_top_level_functions_are_do_not_error():
    hl.current_backend()
    hl.debug_info()


def test_tmpdir_runs():
    isinstance(hl.tmp_dir(), str)


def test_get_flags():
     assert hl._get_flags() == {}
     assert list(hl._get_flags('use_new_shuffle')) == ['use_new_shuffle']


@skip_unless_spark_backend(reason='requires JVM')
def test_flags_same_in_scala_and_python():
    b = hl.current_backend()
    assert isinstance(b, SparkBackend)

    scala_flag_map = _scala_map_str_to_tuple_str_str_to_dict(b._hail_package.HailFeatureFlags.defaults())
    assert scala_flag_map == Backend._flags_env_vars_and_defaults

def test_fast_restarts_feature():
    assert hl._get_flags('use_fast_restarts', 'cachedir') == {
        'use_fast_restarts': None,
        'cachedir': None
    }

    hl._set_flags(use_fast_restarts='1')
    assert hl._get_flags('use_fast_restarts', 'cachedir') == {
        'use_fast_restarts': '1',
        'cachedir': None
    }

    hl._set_flags(cachedir='gs://my-bucket/object-prefix')
    assert hl._get_flags('use_fast_restarts', 'cachedir') == {
        'use_fast_restarts': '1',
        'cachedir': 'gs://my-bucket/object-prefix'
    }
