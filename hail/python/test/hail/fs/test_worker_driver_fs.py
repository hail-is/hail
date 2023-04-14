import hail as hl
from hailtop.utils import secret_alnum_string
from hailtop.test_utils import skip_in_azure

from ..helpers import fails_local_backend


@skip_in_azure
def test_requester_pays_no_settings():
    try:
        hl.import_table('gs://hail-services-requester-pays/hello')
    except Exception as exc:
        assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
    else:
        assert False


@skip_in_azure
def test_requester_pays_write_no_settings():
    random_filename = 'gs://hail-services-requester-pays/test_requester_pays_on_worker_driver_' + secret_alnum_string(10)
    try:
        hl.utils.range_table(4, n_partitions=4).write(random_filename, overwrite=True)
    except Exception as exc:
        assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
    else:
        hl.current_backend().fs.rmtree(random_filename)
        assert False


@skip_in_azure
@fails_local_backend()
def test_requester_pays_write_with_project():
    hl.stop()
    hl.init(gcs_requester_pays_configuration='hail-vdc')
    random_filename = 'gs://hail-services-requester-pays/test_requester_pays_on_worker_driver_' + secret_alnum_string(10)
    try:
        hl.utils.range_table(4, n_partitions=4).write(random_filename, overwrite=True)
    finally:
        hl.current_backend().fs.rmtree(random_filename)


@skip_in_azure
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


@skip_in_azure
def test_requester_pays_with_project_more_than_one_partition():
    # NB: this test uses a file with more rows than partitions because Hadoop's Seekable input
    # streams do not permit seeking past the end of the input (ref:
    # https://hadoop.apache.org/docs/stable/api/org/apache/hadoop/fs/Seekable.html#seek-long-).
    #
    # Hail assumes that seeking past the end of the input does not raise an EOFException (see, for
    # example `skip` in java.io.FileInputStream:
    # https://docs.oracle.com/javase/7/docs/api/java/io/FileInputStream.html)

    expected_file_contents = [
        hl.Struct(f0='idx'),
        hl.Struct(f0='0'),
        hl.Struct(f0='1'),
        hl.Struct(f0='2'),
        hl.Struct(f0='3'),
        hl.Struct(f0='4'),
        hl.Struct(f0='5'),
        hl.Struct(f0='6'),
        hl.Struct(f0='7'),
        hl.Struct(f0='8'),
        hl.Struct(f0='9'),
    ]

    hl.stop()
    hl.init(gcs_requester_pays_configuration='hail-vdc')
    assert hl.import_table('gs://hail-services-requester-pays/zero-to-nine', no_header=True, min_partitions=8).collect() == expected_file_contents

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['hail-services-requester-pays']))
    assert hl.import_table('gs://hail-services-requester-pays/zero-to-nine', no_header=True, min_partitions=8).collect() == expected_file_contents

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['hail-services-requester-pays', 'other-bucket']))
    assert hl.import_table('gs://hail-services-requester-pays/zero-to-nine', no_header=True, min_partitions=8).collect() == expected_file_contents

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
    assert hl.import_table('gs://hail-services-requester-pays/zero-to-nine', no_header=True, min_partitions=8).collect() == expected_file_contents
