import pytest

from hailtop.hailctl.emr import start
from hailtop.hailctl.emr.artifact import HailArtifactManifest


def artifact_manifest(**overrides):
    values = {
        'schema_version': 1,
        'hail_git_revision': 'a' * 40,
        'hail_pip_version': '0.2.140',
        'spark_version': '4.1.2',
        'scala_version': '2.13.18',
        'python_version': '3.12',
        'wheel_uri': 's3://artifact-bucket/hail.whl',
        'wheel_sha256': 'b' * 64,
        'wheelhouse_uri': 's3://artifact-bucket/wheelhouse.tar.gz',
        'wheelhouse_sha256': 'c' * 64,
    }
    values.update(overrides)
    return HailArtifactManifest(**values)


def test_default_release_matches_hail_spark_minor():
    config = start.release_config(start.DEFAULT_EMR_RELEASE)
    assert config.release_label == 'emr-spark-8.1.0'
    assert config.spark_version == '4.1.1'
    assert config.scala_version == '2.13.17'
    assert start.check_release_spark_compatibility(config.release_label, '4.1.2') == config


def test_unsupported_release_errors():
    with pytest.raises(ValueError, match='unsupported EMR release'):
        start.release_config('emr-spark-8.0.0')


def test_spark_minor_mismatch_raises():
    with pytest.raises(ValueError, match='major and minor versions must agree'):
        start.check_release_spark_compatibility(start.DEFAULT_EMR_RELEASE, '4.0.2')


def test_validate_artifact_compatibility():
    release = start.release_config(start.DEFAULT_EMR_RELEASE)
    start.validate_artifact_compatibility(artifact_manifest(), release, '4.1.2')
    with pytest.raises(ValueError, match='artifact was built for Spark'):
        start.validate_artifact_compatibility(artifact_manifest(spark_version='4.0.2'), release, '4.1.2')
    with pytest.raises(ValueError, match='artifact was built for Scala'):
        start.validate_artifact_compatibility(artifact_manifest(scala_version='2.12.18'), release, '4.1.2')
    with pytest.raises(ValueError, match='requires Python'):
        start.validate_artifact_compatibility(artifact_manifest(python_version='3.13'), release, '4.1.2')


def test_hail_configurations_sets_hail_cloud_and_jar():
    confs = start.hail_configurations(off_heap_memory_per_core_mb=None)
    spark_defaults = next(c for c in confs if c['Classification'] == 'spark-defaults')
    props = spark_defaults['Properties']
    assert props['spark.jars'] == f'local://{start.HAIL_JAR_PATH}'
    assert props['spark.serializer'] == 'org.apache.spark.serializer.KryoSerializer'
    assert props['spark.kryo.registrator'] == 'is.hail.kryo.HailKryoRegistrator'
    assert props['spark.executorEnv.HAIL_CLOUD'] == 'aws'
    assert props['spark.yarn.appMasterEnv.HAIL_CLOUD'] == 'aws'
    assert props['spark.executorEnv.PYTHONHASHSEED'] == '0'
    assert props['spark.pyspark.python'] == start.EMR_PYSPARK_PYTHON
    spark = next(c for c in confs if c['Classification'] == 'spark')
    assert spark['Properties']['maximizeResourceAllocation'] == 'true'
    spark_env = next(c for c in confs if c['Classification'] == 'spark-env')
    export = next(c for c in spark_env['Configurations'] if c['Classification'] == 'export')
    assert export['Properties']['HAIL_CLOUD'] == 'aws'
    assert export['Properties']['PYSPARK_PYTHON'] == start.EMR_PYSPARK_PYTHON
    assert export['Properties']['SPARK_DIST_CLASSPATH'] == '$(hadoop classpath)'


def test_hail_configurations_off_heap_overlay():
    confs = start.hail_configurations(off_heap_memory_per_core_mb=1024)
    spark_defaults = next(c for c in confs if c['Classification'] == 'spark-defaults')
    assert spark_defaults['Properties']['spark.executorEnv.HAIL_WORKER_OFF_HEAP_MEMORY_PER_CORE_MB'] == '1024'


def test_deep_merge_overlay_wins_and_recurses():
    base = {'Instances': {'KeepJobFlowAliveWhenNoSteps': True}, 'Name': 'a'}
    overlay = {'Instances': {'Ec2SubnetId': 'subnet-1'}, 'Name': 'b'}
    merged = start.deep_merge(base, overlay)
    assert merged['Name'] == 'b'
    assert merged['Instances']['KeepJobFlowAliveWhenNoSteps'] is True
    assert merged['Instances']['Ec2SubnetId'] == 'subnet-1'


def test_merge_run_job_flow_overlay_switches_to_instance_fleets():
    base = {'Instances': {'InstanceGroups': [{'Name': 'Core'}], 'KeepJobFlowAliveWhenNoSteps': True}}
    overlay = {'Instances': {'InstanceFleets': [{'Name': 'Core fleet'}]}}
    merged = start.merge_run_job_flow_overlay(base, overlay)
    assert 'InstanceGroups' not in merged['Instances']
    assert merged['Instances']['InstanceFleets'] == [{'Name': 'Core fleet'}]
    assert merged['Instances']['KeepJobFlowAliveWhenNoSteps'] is True


def test_build_run_job_flow_kwargs_shape():
    artifact = artifact_manifest()
    kwargs = start.build_run_job_flow_kwargs(
        cluster_name='c1',
        release_label='emr-spark-8.1.0',
        master_instance_type='m5.xlarge',
        core_instance_type='m5.xlarge',
        core_instance_count=2,
        ec2_key_name=None,
        subnet_id='subnet-1',
        service_access_security_group='sg-service',
        primary_security_group='sg-primary',
        core_security_group='sg-core',
        log_uri='s3://bkt/logs/',
        bootstrap_s3_uri='s3://bkt/bootstrap/install-hail-emr.sh',
        artifact=artifact,
        off_heap_memory_per_core_mb=None,
        use_default_roles=True,
        service_role=None,
        instance_profile=None,
        idle_timeout=3600,
    )
    assert kwargs['Name'] == 'c1'
    assert kwargs['ReleaseLabel'] == 'emr-spark-8.1.0'
    assert kwargs['Applications'] == [{'Name': 'Spark'}]
    assert kwargs['LogUri'] == 's3://bkt/logs/'
    assert kwargs['ServiceRole'] == 'EMR_DefaultRole'
    assert kwargs['JobFlowRole'] == 'EMR_EC2_DefaultRole'
    assert kwargs['AutoTerminationPolicy'] == {'IdleTimeout': 3600}
    assert kwargs['Instances']['Ec2SubnetId'] == 'subnet-1'
    assert kwargs['Instances']['ServiceAccessSecurityGroup'] == 'sg-service'
    assert kwargs['Instances']['EmrManagedMasterSecurityGroup'] == 'sg-primary'
    assert kwargs['Instances']['EmrManagedSlaveSecurityGroup'] == 'sg-core'
    ba = kwargs['BootstrapActions'][0]
    assert ba['ScriptBootstrapAction']['Path'] == 's3://bkt/bootstrap/install-hail-emr.sh'
    assert ba['ScriptBootstrapAction']['Args'] == artifact.bootstrap_args()
    tags = {tag['Key']: tag['Value'] for tag in kwargs['Tags']}
    assert tags['for-use-with-amazon-emr-managed-policies'] == 'true'
    assert tags['hail-revision'] == 'a' * 40
    assert tags['hail-spark-version'] == '4.1.2'
    igs = {g['InstanceRole']: g for g in kwargs['Instances']['InstanceGroups']}
    assert igs['MASTER']['Name'] == 'Primary'
    assert igs['CORE']['InstanceCount'] == 2
