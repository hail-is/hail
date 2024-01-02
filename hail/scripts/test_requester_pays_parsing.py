import os
import sys

import pytest

from hailtop.aiocloud.aiogoogle import get_gcs_requester_pays_configuration
from hailtop.aiocloud.aiogoogle.user_config import get_spark_conf_gcs_requester_pays_configuration, spark_conf_path
from hailtop.config import ConfigVariable, configuration_of
from hailtop.utils.process import check_exec_output

if 'YOU_MAY_OVERWRITE_MY_SPARK_DEFAULTS_CONF_AND_HAILCTL_SETTINGS' not in os.environ:
    print(
        'This script will overwrite your spark-defaults.conf and hailctl settings. It is intended to be executed inside a container.'
    )
    sys.exit(1)


SPARK_CONF_PATH = spark_conf_path()
assert SPARK_CONF_PATH


async def unset_hailctl():
    await check_exec_output(
        'hailctl',
        'config',
        'unset',
        'gcs_requester_pays/project',
    )

    await check_exec_output(
        'hailctl',
        'config',
        'unset',
        'gcs_requester_pays/buckets',
    )


@pytest.mark.asyncio
async def test_no_configuration():
    with open(SPARK_CONF_PATH, 'w'):
        pass

    await unset_hailctl()

    actual = get_gcs_requester_pays_configuration()
    assert actual is None


@pytest.mark.asyncio
async def test_no_project_is_error():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.mode AUTO\n')

    await unset_hailctl()

    with pytest.raises(ValueError, match='.*a project must be set if a mode other than DISABLED is set.*'):
        get_gcs_requester_pays_configuration()


@pytest.mark.asyncio
async def test_auto_with_project():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode AUTO\n')

    await unset_hailctl()

    actual = get_gcs_requester_pays_configuration()
    assert actual == 'my_project'


@pytest.mark.asyncio
async def test_custom_no_buckets():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode CUSTOM\n')

    await unset_hailctl()

    with pytest.raises(ValueError, match='.*with mode CUSTOM, buckets must be set.*'):
        get_gcs_requester_pays_configuration()


@pytest.mark.asyncio
async def test_custom_with_buckets():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode CUSTOM\n')
        f.write('spark.hadoop.fs.gs.requester.pays.buckets abc,def\n')

    await unset_hailctl()

    actual = get_gcs_requester_pays_configuration()
    assert actual == ('my_project', ['abc', 'def'])


@pytest.mark.asyncio
async def test_disabled():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode DISABLED\n')
        f.write('spark.hadoop.fs.gs.requester.pays.buckets abc,def\n')

    await unset_hailctl()

    actual = get_gcs_requester_pays_configuration()
    assert actual is None


@pytest.mark.asyncio
async def test_enabled():
    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode ENABLED\n')
        f.write('spark.hadoop.fs.gs.requester.pays.buckets abc,def\n')

    await unset_hailctl()

    actual = get_gcs_requester_pays_configuration()
    assert actual == 'my_project'


@pytest.mark.asyncio
async def test_hailctl_takes_precedence_1():
    await unset_hailctl()

    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode ENABLED\n')
        f.write('spark.hadoop.fs.gs.requester.pays.buckets abc,def\n')

    await check_exec_output('hailctl', 'config', 'set', 'gcs_requester_pays/project', 'hailctl_project', echo=True)

    actual = get_gcs_requester_pays_configuration()
    assert actual == 'hailctl_project', str(
        (
            configuration_of(ConfigVariable.GCS_REQUESTER_PAYS_PROJECT, None, None),
            configuration_of(ConfigVariable.GCS_REQUESTER_PAYS_BUCKETS, None, None),
            get_spark_conf_gcs_requester_pays_configuration(),
            open('/Users/dking/.config/hail/config.ini', 'r').readlines(),
        )
    )


@pytest.mark.asyncio
async def test_hailctl_takes_precedence_2():
    await unset_hailctl()

    with open(SPARK_CONF_PATH, 'w') as f:
        f.write('spark.hadoop.fs.gs.requester.pays.project.id my_project\n')
        f.write('spark.hadoop.fs.gs.requester.pays.mode ENABLED\n')
        f.write('spark.hadoop.fs.gs.requester.pays.buckets abc,def\n')

    await check_exec_output('hailctl', 'config', 'set', 'gcs_requester_pays/project', 'hailctl_project2', echo=True)

    await check_exec_output('hailctl', 'config', 'set', 'gcs_requester_pays/buckets', 'bucket1,bucket2', echo=True)

    actual = get_gcs_requester_pays_configuration()
    assert actual == ('hailctl_project2', ['bucket1', 'bucket2'])
