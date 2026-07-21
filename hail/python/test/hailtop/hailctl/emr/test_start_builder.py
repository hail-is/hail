import pytest

from hailtop.hailctl.emr import start


def test_default_release_is_known_and_matches_hail_spark():
    assert start.DEFAULT_EMR_RELEASE in start.EMR_RELEASE_SPARK_VERSION
    spark = start.EMR_RELEASE_SPARK_VERSION[start.DEFAULT_EMR_RELEASE]
    assert spark.rsplit('.', 1)[0] == start.HAIL_REQUIRED_SPARK_MINOR


def test_check_release_unknown_warns(capsys):
    start.check_release_spark_compatibility('emr-9.9.9')
    assert 'unknown' in capsys.readouterr().err.lower()


def test_check_release_mismatch_raises():
    with pytest.raises(ValueError, match='Spark'):
        start.check_release_spark_compatibility('emr-6.15.0')


def test_check_release_match_ok():
    start.check_release_spark_compatibility(start.DEFAULT_EMR_RELEASE)  # no raise


def test_hail_configurations_sets_hail_cloud_and_jar():
    confs = start.hail_configurations(off_heap_memory_per_core_mb=None)
    spark_defaults = next(c for c in confs if c['Classification'] == 'spark-defaults')
    props = spark_defaults['Properties']
    assert props['spark.jars'] == f'local://{start.HAIL_JAR_PATH}'
    assert props['spark.executorEnv.HAIL_CLOUD'] == 'aws'
    assert props['spark.yarn.appMasterEnv.HAIL_CLOUD'] == 'aws'
    assert props['spark.executorEnv.PYTHONHASHSEED'] == '0'
    spark = next(c for c in confs if c['Classification'] == 'spark')
    assert spark['Properties']['maximizeResourceAllocation'] == 'true'
    spark_env = next(c for c in confs if c['Classification'] == 'spark-env')
    export = next(c for c in spark_env['Configurations'] if c['Classification'] == 'export')
    assert export['Properties']['HAIL_CLOUD'] == 'aws'


def test_hail_configurations_off_heap_overlay():
    confs = start.hail_configurations(off_heap_memory_per_core_mb=1024)
    spark_defaults = next(c for c in confs if c['Classification'] == 'spark-defaults')
    props = spark_defaults['Properties']
    assert props['spark.executorEnv.HAIL_WORKER_OFF_HEAP_MEMORY_PER_CORE_MB'] == '1024'


def test_deep_merge_overlay_wins_and_recurses():
    base = {'Instances': {'KeepJobFlowAliveWhenNoSteps': True}, 'Name': 'a'}
    overlay = {'Instances': {'Ec2SubnetId': 'subnet-1'}, 'Name': 'b'}
    merged = start.deep_merge(base, overlay)
    assert merged['Name'] == 'b'
    assert merged['Instances']['KeepJobFlowAliveWhenNoSteps'] is True
    assert merged['Instances']['Ec2SubnetId'] == 'subnet-1'


def test_build_run_job_flow_kwargs_shape():
    kwargs = start.build_run_job_flow_kwargs(
        cluster_name='c1',
        release_label='emr-7.3.0',
        master_instance_type='m5.xlarge',
        core_instance_type='m5.xlarge',
        core_instance_count=2,
        ec2_key_name=None,
        subnet_id=None,
        log_uri='s3://bkt/logs/',
        bootstrap_s3_uri='s3://bkt/bootstrap/install-hail-emr.sh',
        pip_version='0.2.140',
        off_heap_memory_per_core_mb=None,
        use_default_roles=True,
        service_role=None,
        instance_profile=None,
    )
    assert kwargs['Name'] == 'c1'
    assert kwargs['ReleaseLabel'] == 'emr-7.3.0'
    assert kwargs['Applications'] == [{'Name': 'Spark'}]
    assert kwargs['LogUri'] == 's3://bkt/logs/'
    assert kwargs['ServiceRole'] == 'EMR_DefaultRole'
    assert kwargs['JobFlowRole'] == 'EMR_EC2_DefaultRole'
    ba = kwargs['BootstrapActions'][0]
    assert ba['ScriptBootstrapAction']['Path'] == 's3://bkt/bootstrap/install-hail-emr.sh'
    assert ba['ScriptBootstrapAction']['Args'] == ['0.2.140']
    igs = {g['InstanceRole']: g for g in kwargs['Instances']['InstanceGroups']}
    assert igs['MASTER']['InstanceType'] == 'm5.xlarge'
    assert igs['CORE']['InstanceCount'] == 2
