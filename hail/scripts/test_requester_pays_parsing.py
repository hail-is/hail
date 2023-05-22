
from hailtop.config.user_config import get_gcs_requester_pays_configuration, spark_conf_path
from hailtop.utils.process import check_exec_output

# WARNING:
#
# This script will overwrite your spark-defaults.conf. It is intended to be executed inside a
# container.


SPARK_CONF_PATH = spark_conf_path()


def test_no_configuration():
    actual = get_gcs_requester_pays_configuration('test_no_configuration')
    assert actual is None



def test_no_project_is_error():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.mode AUTO\n')

    try:
        get_gcs_requester_pays_configuration('test_no_project_is_error')
    except ValueError as err:
        assert 'a project must be set if a mode other than DISABLED is set' in err.args[0]
    else:
        assert False


def test_auto_with_project():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode AUTO\n')
    actual = get_gcs_requester_pays_configuration('test_auto_with_project')
    assert actual == 'my_project'



def test_custom_no_buckets():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode CUSTOM\n')
    try:
        get_gcs_requester_pays_configuration('test_custom_no_buckets')
    except ValueError as err:
        assert 'with mode CUSTOM buckets must be set' in err.args[0]
    else:
        assert False



def test_custom_with_buckets():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode CUSTOM\n')
        f.write('spark.hadoop.fs.gs.requester.pays.buckets abc,def\n')

    actual = get_gcs_requester_pays_configuration('test_custom_with_buckets')
    assert actual == ('my_project', ['abc', 'def'])



def test_disabled():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode DISABLED\n')
        f.write('spark.hadoop.fs.gs.requester.pays.buckets abc,def\n')

    actual = get_gcs_requester_pays_configuration('test_disabled')
    assert actual is None



def test_enabled():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode ENABLED\n')
        f.write('spark.hadoop.fs.gs.requester.pays.buckets abc,def\n')

    actual = get_gcs_requester_pays_configuration('test_disabled')
    assert actual == 'my_project'



def test_hailctl_takes_precedence_1():
    check_exec_output(
        'hailctl',
        'config',
        'set',
        'gcs_requester_pays/project',
        'hailctl_project',
        echo=True
    )

    actual = get_gcs_requester_pays_configuration('test_hailctl_takes_precedence')
    assert actual == 'hailctl_project'



def test_hailctl_takes_precedence_2():
    check_exec_output(
        'hailctl',
        'config',
        'set',
        'gcs_requester_pays/project',
        'hailctl_project2',
        echo=True
    )

    check_exec_output(
        'hailctl',
        'config',
        'set',
        'gcs_requester_pays/buckets',
        'bucket1,bucket2',
        echo=True
    )

    actual = get_gcs_requester_pays_configuration('test_hailctl_takes_precedence')
    assert actual == ('hailctl_project2', ['bucket1', 'bucket2'])
