import os
from timeit import default_timer as timer
import unittest
from decorator import decorator

import hail as hl

_initialized = False


def startTestHailContext():
    global _initialized
    if not _initialized:
        hl.init(master='local[2]', min_block_size=0, quiet=True)
        _initialized = True


def stopTestHailContext():
    pass


_test_dir = None
_doctest_dir = None


def resource(filename):
    global _test_dir
    if _test_dir is None:
        path = '.'
        i = 0
        while not os.path.exists(os.path.join(path, 'build.gradle')):
            path = os.path.join(path, '..')
            i += 1
            if i > 100:
                raise EnvironmentError("Hail tests must be run from inside the Hail git repository")
        _test_dir = os.path.join(path, 'src', 'test', 'resources')

    return os.path.join(_test_dir, filename)


def doctest_resource(filename):
    global _doctest_dir
    if _doctest_dir is None:
        path = '.'
        i = 0
        while not os.path.exists(os.path.join(path, 'build.gradle')):
            path = os.path.join(path, '..')
            i += 1
            if i > 100:
                raise EnvironmentError("Hail tests must be run from inside the Hail git repository")
        _doctest_dir = os.path.join(path, 'python', 'hail', 'docs', 'data')

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
        m=hl.null(hl.tfloat64),
        astruct=hl.struct(a=hl.null(hl.tint32), b=5.5),
        mstruct=hl.null(hl.tstruct(x=hl.tint32, y=hl.tstr)),
        aset=hl.set(['foo', 'bar', 'baz']),
        mset=hl.null(hl.tset(hl.tfloat64)),
        d=hl.dict({hl.array(['a', 'b']): 0.5, hl.array(['x', hl.null(hl.tstr), 'z']): 0.3}),
        md=hl.null(hl.tdict(hl.tint32, hl.tstr)),
        h38=hl.locus('chr22', 33878978, 'GRCh38'),
        ml=hl.null(hl.tlocus('GRCh37')),
        i=hl.interval(
            hl.locus('1', 999),
            hl.locus('1', 1001)),
        c=hl.call(0, 1),
        mc=hl.null(hl.tcall),
        t=hl.tuple([hl.call(1, 2, phased=True), 'foo', hl.null(hl.tstr)]),
        mt=hl.null(hl.ttuple(hl.tlocus('GRCh37'), hl.tbool))
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

def skip_unless_spark_backend():
    @decorator
    def wrapper(func, *args, **kwargs):
        if isinstance(hl.utils.java.Env.backend(), hl.backend.SparkBackend):
            return func(*args, **kwargs)
        else:
            raise unittest.SkipTest('requires Spark')

    return wrapper


def run_with_cxx_compile():
    @decorator
    def wrapper(func, *args, **kwargs):
        old_flags = hl._get_flags('cpp')
        hl._set_flags(cpp='t')
        func(*args, **kwargs)
        hl._set_flags(**old_flags)

    return wrapper
