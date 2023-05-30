import hail as hl
import pytest
from hailtop.utils import secret_alnum_string
from hailtop.test_utils import skip_in_azure, run_if_azure

from ..helpers import fails_local_backend, backend_specific_timeout


@skip_in_azure
def test_requester_pays_no_settings():
    try:
        hl.import_table('gs://hail-test-requester-pays-fds32/hello')
    except Exception as exc:
        assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
    else:
        assert False


@skip_in_azure
def test_requester_pays_write_no_settings():
    random_filename = 'gs://hail-test-requester-pays-fds32/test_requester_pays_on_worker_driver_' + secret_alnum_string(10)
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
    random_filename = 'gs://hail-test-requester-pays-fds32/test_requester_pays_on_worker_driver_' + secret_alnum_string(10)
    try:
        hl.utils.range_table(4, n_partitions=4).write(random_filename, overwrite=True)
    finally:
        hl.current_backend().fs.rmtree(random_filename)


@skip_in_azure
def test_requester_pays_with_project():
    hl.stop()
    hl.init(gcs_requester_pays_configuration='hail-vdc')
    assert hl.import_table('gs://hail-test-requester-pays-fds32/hello', no_header=True).collect() == [hl.Struct(f0='hello')]

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['hail-test-requester-pays-fds32']))
    assert hl.import_table('gs://hail-test-requester-pays-fds32/hello', no_header=True).collect() == [hl.Struct(f0='hello')]

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['hail-test-requester-pays-fds32', 'other-bucket']))
    assert hl.import_table('gs://hail-test-requester-pays-fds32/hello', no_header=True).collect() == [hl.Struct(f0='hello')]

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['other-bucket']))
    try:
        hl.import_table('gs://hail-test-requester-pays-fds32/hello')
    except Exception as exc:
        assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
    else:
        assert False

    hl.stop()
    hl.init(gcs_requester_pays_configuration='hail-vdc')
    assert hl.import_table('gs://hail-test-requester-pays-fds32/hello', no_header=True).collect() == [hl.Struct(f0='hello')]


@skip_in_azure
@backend_specific_timeout(local=5 * 60)
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
    assert hl.import_table('gs://hail-test-requester-pays-fds32/zero-to-nine', no_header=True, min_partitions=8).collect() == expected_file_contents

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['hail-test-requester-pays-fds32']))
    assert hl.import_table('gs://hail-test-requester-pays-fds32/zero-to-nine', no_header=True, min_partitions=8).collect() == expected_file_contents

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['hail-test-requester-pays-fds32', 'other-bucket']))
    assert hl.import_table('gs://hail-test-requester-pays-fds32/zero-to-nine', no_header=True, min_partitions=8).collect() == expected_file_contents

    hl.stop()
    hl.init(gcs_requester_pays_configuration=('hail-vdc', ['other-bucket']))
    try:
        hl.import_table('gs://hail-test-requester-pays-fds32/zero-to-nine', min_partitions=8)
    except Exception as exc:
        assert "Bucket is a requester pays bucket but no user project provided" in exc.args[0]
    else:
        assert False

    hl.stop()
    hl.init(gcs_requester_pays_configuration='hail-vdc')
    assert hl.import_table('gs://hail-test-requester-pays-fds32/zero-to-nine', no_header=True, min_partitions=8).collect() == expected_file_contents


@run_if_azure
@fails_local_backend
def test_can_access_public_blobs():
    public_mt = 'hail-az://azureopendatastorage/gnomad/release/3.1/mt/genomes/gnomad.genomes.v3.1.hgdp_1kg_subset.mt'
    assert hl.hadoop_exists(public_mt)
    with hl.hadoop_open(f'{public_mt}/README.txt') as readme:
        assert len(readme.read()) > 0
    mt = hl.read_matrix_table(public_mt)
    mt.describe()
