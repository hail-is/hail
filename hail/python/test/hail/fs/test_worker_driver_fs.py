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
    random_filename = 'gs://hail-services-requester-pays/test_requester_pays_on_worker_driver_' + secret_alnum_string(10)
    try:
        hl.utils.range_table(4, n_partitions=4).write(random_filename, overwrite=True)
    except Exception as exc:
        assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
    else:
        hl.current_backend().fs.rmtree(random_filename)
        assert False


def test_requester_pays_with_project():
    hl.stop()
    hl.init(gcs_requester_pays_configuration='hail-vdc')
    assert hl.import_table('gs://hail-services-requester-pays/hello', no_header=True).collect() == [hl.Struct(f0='hello')]

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['hail-services-requester-pays']))
    assert hl.import_table('gs://hail-services-requester-pays/hello', no_header=True).collect() == [hl.Struct(f0='hello')]

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['hail-services-requester-pays', 'other-bucket']))
    assert hl.import_table('gs://hail-services-requester-pays/hello', no_header=True).collect() == [hl.Struct(f0='hello')]

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['other-bucket']))
    try:
        hl.import_table('gs://hail-services-requester-pays/hello')
    except Exception as exc:
        assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
    else:
        assert False

    hl.stop()
    hl.init(gcs_requester_pays_configuration='hail-vdc')
    assert hl.import_table('gs://hail-services-requester-pays/hello', no_header=True).collect() == [hl.Struct(f0='hello')]


def test_requester_pays_with_project_more_than_one_partition():
    # NB: this test uses a file with more rows than partitions because Hadoop's Seekable input
    # streams do not permit seeking past the end of the input (ref:
    # https://hadoop.apache.org/docs/stable/api/org/apache/hadoop/fs/Seekable.html#seek-long-).
    #
    # Hail assumes that seeking past the end of the input does not raise an EOFException (see, for
    # example `skip` in java.io.FileInputStream:
    # https://docs.oracle.com/javase/7/docs/api/java/io/FileInputStream.html)
    hl.stop()
    hl.init(gcs_requester_pays_configuration='hail-vdc')
    assert hl.import_table('gs://hail-services-requester-pays/zero-to-nine', no_header=True, min_partitions=8).collect() == [hl.Struct(f0='hello')]

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['hail-services-requester-pays']))
    assert hl.import_table('gs://hail-services-requester-pays/zero-to-nine', no_header=True, min_partitions=8).collect() == [hl.Struct(f0='hello')]

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['hail-services-requester-pays', 'other-bucket']))
    assert hl.import_table('gs://hail-services-requester-pays/zero-to-nine', no_header=True, min_partitions=8).collect() == [hl.Struct(f0='hello')]

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['other-bucket']))
    try:
        hl.import_table('gs://hail-services-requester-pays/zero-to-nine', min_partitions=8)
    except Exception as exc:
        assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
    else:
        assert False

    hl.stop()
    hl.init(gcs_requester_pays_configuration='hail-vdc')
    assert hl.import_table('gs://hail-services-requester-pays/zero-to-nine', no_header=True, min_partitions=8).collect() == [hl.Struct(f0='hello')]
