import hail as hl
from ..helpers import startTestHailContext, stopTestHailContext
from hailtop.utils import secret_alnum_string

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


def test_requester_pays_no_settings():
    try:
        hl.import_table('gs://hail-services-requester-pays/hello')
    except Exception as exc:
        assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
    else:
        assert False


def test_requester_pays_write_no_settings():
    try:
        random_filename = 'gs://hail-services-requester-pays/test_requester_pays_on_worker_driver_' + secret_alnum_string(10)
        try:
            hl.utils.range_table(4, n_partitions=4).write(random_filename, overwrite=True)
        except Exception as exc:
            assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
        else:
            assert False
    finally:
        hl.current_backend().fs.rmtree(random_filename)


def test_requester_pays_with_project():
    flags = hl._get_flags('requester_pays_project', 'requester_pays_buckets')
    try:
        hl._set_flags(requester_pays_project='broad-ctsa')
        assert hl.import_table('gs://hail-services-requester-pays/hello', no_header=True).collect() == [hl.Struct(f0='hello')]

        hl._set_flags(requester_pays_buckets='hail-services-requester-pays')
        assert hl.import_table('gs://hail-services-requester-pays/hello', no_header=True).collect() == [hl.Struct(f0='hello')]

        hl._set_flags(requester_pays_buckets='hail-services-requester-pays,other-bucket')
        assert hl.import_table('gs://hail-services-requester-pays/hello', no_header=True).collect() == [hl.Struct(f0='hello')]

        hl._set_flags(requester_pays_buckets='other-bucket')
        try:
            hl.import_table('gs://hail-services-requester-pays/hello')
        except Exception as exc:
            assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
        else:
            assert False

        hl._set_flags(requester_pays_buckets=None)
        assert hl.import_table('gs://hail-services-requester-pays/hello', no_header=True).collect() == [hl.Struct(f0='hello')]
    finally:
        hl._set_flags(requester_pays_project=flags.get('requester_pays_project'), requester_pays_buckets=flags.get('requester_pays_bucket'))


def test_requester_pays_write():
    flags = hl._get_flags('requester_pays_project', 'requester_pays_buckets')
    random_filename = 'gs://hail-services-requester-pays/test_requester_pays_on_worker_driver_' + secret_alnum_string(10)
    try:
        hl._set_flags(requester_pays_project='broad-ctsa')
        hl.utils.range_table(4, n_partitions=4).write(random_filename)
        assert hl.read_table(random_filename).collect() == [hl.Struct(idx=0), hl.Struct(idx=1), hl.Struct(idx=2), hl.Struct(idx=3)]

        hl._set_flags(requester_pays_buckets='hail-services-requester-pays')
        hl.utils.range_table(4, n_partitions=4).write(random_filename, overwrite=True)
        assert hl.read_table(random_filename).collect() == [hl.Struct(idx=0), hl.Struct(idx=1), hl.Struct(idx=2), hl.Struct(idx=3)]

        hl._set_flags(requester_pays_buckets='hail-services-requester-pays,other-bucket')
        hl.utils.range_table(4, n_partitions=4).write(random_filename, overwrite=True)
        assert hl.read_table(random_filename).collect() == [hl.Struct(idx=0), hl.Struct(idx=1), hl.Struct(idx=2), hl.Struct(idx=3)]

        hl._set_flags(requester_pays_buckets='other-bucket')
        try:
            hl.utils.range_table(4, n_partitions=4).write(random_filename, overwrite=True)
        except Exception as exc:
            assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
        else:
            assert False

        hl._set_flags(requester_pays_buckets=None)
        hl.utils.range_table(4, n_partitions=4).write(random_filename, overwrite=True)
        assert hl.read_table(random_filename).collect() == [hl.Struct(idx=0), hl.Struct(idx=1), hl.Struct(idx=2), hl.Struct(idx=3)]
    finally:
        hl.current_backend().fs.rmtree(random_filename)
        hl._set_flags(requester_pays_project=flags.get('requester_pays_project'), requester_pays_buckets=flags.get('requester_pays_bucket'))
