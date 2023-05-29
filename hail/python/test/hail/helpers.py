import os
from timeit import default_timer as timer
import unittest
import pytest
from decorator import decorator

from hail.utils.java import Env, choose_backend
import hail as hl


GCS_REQUESTER_PAYS_PROJECT = os.environ.get('GCS_REQUESTER_PAYS_PROJECT')


def startTestHailContext():
    backend_name = choose_backend()
    if backend_name == 'spark':
        hl.init(master='local[2]', min_block_size=0, quiet=True, global_seed=0)
    else:
        hl.init(global_seed=0)


def stopTestHailContext():
    hl.stop()


_test_dir = os.environ.get('HAIL_TEST_RESOURCES_DIR',
                           os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(hl.__file__))),
                                        'src/test/resources'))
_doctest_dir = os.environ.get('HAIL_DOCTEST_DATA_DIR', 'hail/docs/data')


def resource(filename):
    return os.path.join(_test_dir, filename)


def doctest_resource(filename):
    return os.path.join(_doctest_dir, filename)


def schema_eq(x, y):
    x_fds = dict(x)
    y_fds = dict(y)
    return x_fds == y_fds


def convert_struct_to_dict(x):
    if isinstance(x, hl.Struct):
        return {k: convert_struct_to_dict(v) for k, v in x._fields.items()}
    elif isinstance(x, list):
        return [convert_struct_to_dict(elt) for elt in x]
    elif isinstance(x, tuple):
        return tuple([convert_struct_to_dict(elt) for elt in x])
    elif isinstance(x, dict):
        return {k: convert_struct_to_dict(v) for k, v in x.items()}
    else:
        return x


_dataset = None


def get_dataset():
    global _dataset
    if _dataset is None:
        _dataset = hl.split_multi_hts(hl.import_vcf(resource('sample.vcf'))).cache()
    return _dataset

def assert_time(f, max_duration):
    start = timer()
    x = f()
    end = timer()
    assert (start - end) < max_duration
    print(f'took {end - start:.3f}')
    return x

def create_all_values():
    return hl.struct(
        f32=hl.float32(3.14),
        i64=hl.int64(-9),
        m=hl.missing(hl.tfloat64),
        astruct=hl.struct(a=hl.missing(hl.tint32), b=5.5),
        mstruct=hl.missing(hl.tstruct(x=hl.tint32, y=hl.tstr)),
        aset=hl.set(['foo', 'bar', 'baz']),
        mset=hl.missing(hl.tset(hl.tfloat64)),
        d=hl.dict({hl.array(['a', 'b']): 0.5, hl.array(['x', hl.missing(hl.tstr), 'z']): 0.3}),
        md=hl.missing(hl.tdict(hl.tint32, hl.tstr)),
        h38=hl.locus('chr22', 33878978, 'GRCh38'),
        ml=hl.missing(hl.tlocus('GRCh37')),
        i=hl.interval(
            hl.locus('1', 999),
            hl.locus('1', 1001)),
        c=hl.call(0, 1),
        mc=hl.missing(hl.tcall),
        t=hl.tuple([hl.call(1, 2, phased=True), 'foo', hl.missing(hl.tstr)]),
        mt=hl.missing(hl.ttuple(hl.tlocus('GRCh37'), hl.tbool)),
        nd=hl.nd.arange(0, 10).reshape((2, 5)),
    )

def prefix_struct(s, prefix):
    return hl.struct(**{prefix + k: s[k] for k in s})

def create_all_values_table():
    all_values = create_all_values()
    return (hl.utils.range_table(5, n_partitions=3)
            .annotate_globals(**prefix_struct(all_values, 'global_'))
            .annotate(**all_values)
            .cache())

def create_all_values_matrix_table():
    all_values = create_all_values()
    return (hl.utils.range_matrix_table(3, 2, n_partitions=2)
            .annotate_globals(**prefix_struct(all_values, 'global_'))
            .annotate_rows(**prefix_struct(all_values, 'row_'))
            .annotate_cols(**prefix_struct(all_values, 'col_'))
            .annotate_entries(**prefix_struct(all_values, 'entry_'))
            .cache())


def create_all_values_datasets():
    return (create_all_values_table(), create_all_values_matrix_table())


def skip_unless_spark_backend(reason='requires Spark'):
    from hail.backend.spark_backend import SparkBackend
    @decorator
    def wrapper(func, *args, **kwargs):
        if isinstance(hl.utils.java.Env.backend(), SparkBackend):
            return func(*args, **kwargs)
        else:
            raise unittest.SkipTest(reason)

    return wrapper


def skip_when_service_backend(reason='skipping for Service Backend'):
    from hail.backend.service_backend import ServiceBackend
    @decorator
    def wrapper(func, *args, **kwargs):
        if isinstance(hl.utils.java.Env.backend(), ServiceBackend):
            raise unittest.SkipTest(reason)
        else:
            return func(*args, **kwargs)

    return wrapper


def skip_unless_service_backend(reason='only relevant to service backend', clouds=None):
    from hail.backend.service_backend import ServiceBackend
    @decorator
    def wrapper(func, *args, **kwargs):
        if not isinstance(hl.utils.java.Env.backend(), ServiceBackend):
            raise unittest.SkipTest(reason)
        else:
            if clouds is None or os.environ['HAIL_CLOUD'] in clouds:
                return func(*args, **kwargs)
            raise unittest.SkipTest(f'{reason} for clouds {clouds}')

    return wrapper


fails_local_backend = pytest.mark.xfail(
    choose_backend() == 'local',
    reason="doesn't yet work on local backend",
    strict=True)


fails_service_backend = pytest.mark.xfail(
    choose_backend() == 'batch',
    reason="doesn't yet work on service backend",
    strict=True)


fails_spark_backend = pytest.mark.xfail(
    choose_backend() == 'spark',
    reason="doesn't yet work on spark backend",
    strict=True)


def assert_evals_to(e, v):
    res = hl.eval(e)
    if res != v:
        raise ValueError(f'  actual: {res}\n  expected: {v}')


def assert_all_eval_to(*expr_and_expected):
    exprs, expecteds = zip(*expr_and_expected)
    assert_evals_to(hl.tuple(exprs), expecteds)


def with_flags(**flags):
    @decorator
    def wrapper(func, *args, **kwargs):
        prev_flags = hl._get_flags(*flags)

        hl._set_flags(**{k: v for k, v in flags.items()})

        try:
            return func(*args, **kwargs)
        finally:
            hl._set_flags(**prev_flags)
    return wrapper


def set_gcs_requester_pays_configuration(gcs_project):
    if gcs_project is not None:
        return with_flags(gcs_requester_pays_project=gcs_project)
    return with_flags()


def lower_only():
    return with_flags(lower='1', lower_bm='1', lower_only='1')
