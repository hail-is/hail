from typing import Dict, Optional, Tuple

import pytest

import hail as hl
from hail.backend.backend import Backend
from hail.backend.spark_backend import SparkBackend
from hail.utils.java import Env
from test.hail.helpers import hl_init_for_test, hl_stop_for_test, qobtest, skip_unless_spark_backend, with_flags


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


@qobtest
@pytest.mark.uninitialized
def test_init_hail_context_twice():
    hl_init_for_test()
    hl_init_for_test(idempotent=True)  # Should be no error
    hl_stop_for_test()

    hl_init_for_test(idempotent=True)
    hl.experimental.define_function(lambda x: x + 2, hl.tint32)
    # ensure functions are cleaned up without error
    hl_stop_for_test()

    hl_init_for_test(idempotent=True)  # Should be no error

    if isinstance(Env.backend(), SparkBackend):
        hl_init_for_test(sc=hl.spark_context(), idempotent=True)  # Should be no error


@qobtest
def test_top_level_functions_are_do_not_error():
    hl.current_backend()
    hl.debug_info()


@qobtest
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
    def is_featured_off():
        return hl._get_flags('use_fast_restarts', 'cachedir') == {'use_fast_restarts': None, 'cachedir': None}

    @with_flags(use_fast_restarts='1')
    def uses_fast_restarts():
        return hl._get_flags('use_fast_restarts', 'cachedir') == {'use_fast_restarts': '1', 'cachedir': None}

    @with_flags(use_fast_restarts='1', cachedir='gs://my-bucket/object-prefix')
    def uses_cachedir():
        return hl._get_flags('use_fast_restarts', 'cachedir') == {
            'use_fast_restarts': '1',
            'cachedir': 'gs://my-bucket/object-prefix',
        }

    assert is_featured_off()
    assert uses_fast_restarts()
    assert is_featured_off()
    assert uses_cachedir()
    assert is_featured_off()
