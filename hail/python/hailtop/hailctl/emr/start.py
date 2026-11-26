import copy
from dataclasses import dataclass
from typing import Optional

from .artifact import HailArtifactManifest, major_minor


@dataclass(frozen=True)
class EMRReleaseConfig:
    release_label: str
    spark_version: str
    scala_version: str
    python_executable: str
    storage_connector: str


EMR_RELEASE_CONFIGS = {
    'emr-spark-8.1.0': EMRReleaseConfig(
        release_label='emr-spark-8.1.0',
        spark_version='4.1.1',
        scala_version='2.13.17',
        python_executable='/usr/bin/python3.12',
        storage_connector='s3a',
    )
}
DEFAULT_EMR_RELEASE = 'emr-spark-8.1.0'
HAIL_JAR_PATH = '/usr/lib/hail/hail-all-spark.jar'
EMR_PYSPARK_PYTHON = '/usr/bin/python3.12'


def release_config(release_label: str) -> EMRReleaseConfig:
    config = EMR_RELEASE_CONFIGS.get(release_label)
    if config is None:
        supported = ', '.join(sorted(EMR_RELEASE_CONFIGS))
        raise ValueError(f'unsupported EMR release {release_label!r}; supported release(s): {supported}')
    return config


def check_release_spark_compatibility(release_label: str, hail_spark_version: str) -> EMRReleaseConfig:
    config = release_config(release_label)
    if major_minor(config.spark_version) != major_minor(hail_spark_version):
        raise ValueError(
            f'EMR release {release_label!r} ships Spark {config.spark_version}, but this Hail artifact '
            f'was built for Spark {hail_spark_version}; the Spark major and minor versions must agree.'
        )
    return config


def validate_artifact_compatibility(
    artifact: HailArtifactManifest,
    release: EMRReleaseConfig,
    hail_spark_version: str,
) -> None:
    if major_minor(artifact.spark_version) != major_minor(hail_spark_version):
        raise ValueError(
            f'artifact was built for Spark {artifact.spark_version}, but hailctl was built for Spark '
            f'{hail_spark_version}'
        )
    if major_minor(artifact.spark_version) != major_minor(release.spark_version):
        raise ValueError(
            f'artifact was built for Spark {artifact.spark_version}, but EMR release {release.release_label!r} '
            f'ships Spark {release.spark_version}'
        )
    if major_minor(artifact.scala_version) != major_minor(release.scala_version):
        raise ValueError(
            f'artifact was built for Scala {artifact.scala_version}, but EMR release {release.release_label!r} '
            f'ships Scala {release.scala_version}'
        )
    if major_minor(artifact.python_version) != major_minor(EMR_PYSPARK_PYTHON.rsplit('python', 1)[1]):
        raise ValueError(
            f'artifact requires Python {artifact.python_version}, but EMR bootstrap is configured for '
            f'{EMR_PYSPARK_PYTHON}'
        )


def hail_configurations(off_heap_memory_per_core_mb: Optional[int]) -> list[dict]:
    spark_defaults = {
        'spark.jars': f'local://{HAIL_JAR_PATH}',
        'spark.driver.extraClassPath': HAIL_JAR_PATH,
        'spark.executor.extraClassPath': HAIL_JAR_PATH,
        'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
        'spark.kryo.registrator': 'is.hail.kryo.HailKryoRegistrator',
        'spark.task.maxFailures': '20',
        'spark.driver.extraJavaOptions': '-Xss4M',
        'spark.executor.extraJavaOptions': '-Xss4M',
        'spark.executorEnv.PYTHONHASHSEED': '0',
        'spark.yarn.appMasterEnv.PYTHONHASHSEED': '0',
        'spark.executorEnv.HAIL_CLOUD': 'aws',
        'spark.yarn.appMasterEnv.HAIL_CLOUD': 'aws',
        'spark.pyspark.python': EMR_PYSPARK_PYTHON,
    }
    if off_heap_memory_per_core_mb is not None:
        spark_defaults['spark.executorEnv.HAIL_WORKER_OFF_HEAP_MEMORY_PER_CORE_MB'] = str(off_heap_memory_per_core_mb)
    return [
        {'Classification': 'spark-defaults', 'Properties': spark_defaults},
        {'Classification': 'spark', 'Properties': {'maximizeResourceAllocation': 'true'}},
        {
            'Classification': 'spark-env',
            'Configurations': [
                {
                    'Classification': 'export',
                    'Properties': {'HAIL_CLOUD': 'aws', 'PYSPARK_PYTHON': EMR_PYSPARK_PYTHON},
                }
            ],
            'Properties': {},
        },
    ]


def deep_merge(base: dict, overlay: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_run_job_flow_overlay(base: dict, overlay: dict) -> dict:
    result = deep_merge(base, overlay)
    instances_overlay = overlay.get('Instances')
    if (
        isinstance(instances_overlay, dict)
        and 'InstanceFleets' in instances_overlay
        and 'InstanceGroups' not in instances_overlay
    ):
        instances = result.get('Instances')
        if isinstance(instances, dict):
            instances.pop('InstanceGroups', None)
    return result


def build_run_job_flow_kwargs(
    *,
    cluster_name: str,
    release_label: str,
    master_instance_type: str,
    core_instance_type: str,
    core_instance_count: int,
    ec2_key_name: Optional[str],
    subnet_id: Optional[str],
    service_access_security_group: Optional[str],
    primary_security_group: Optional[str],
    core_security_group: Optional[str],
    log_uri: Optional[str],
    bootstrap_s3_uri: str,
    artifact: HailArtifactManifest,
    off_heap_memory_per_core_mb: Optional[int],
    use_default_roles: bool,
    service_role: Optional[str],
    instance_profile: Optional[str],
    idle_timeout: int,
) -> dict:
    kwargs: dict = {
        'Name': cluster_name,
        'ReleaseLabel': release_label,
        'Applications': [{'Name': 'Spark'}],
        'Configurations': hail_configurations(off_heap_memory_per_core_mb),
        'BootstrapActions': [
            {
                'Name': 'install-hail',
                'ScriptBootstrapAction': {'Path': bootstrap_s3_uri, 'Args': artifact.bootstrap_args()},
            }
        ],
        'Instances': {
            'InstanceGroups': [
                {
                    'Name': 'Primary',
                    'InstanceRole': 'MASTER',
                    'InstanceType': master_instance_type,
                    'InstanceCount': 1,
                },
                {
                    'Name': 'Core',
                    'InstanceRole': 'CORE',
                    'InstanceType': core_instance_type,
                    'InstanceCount': core_instance_count,
                },
            ],
            'KeepJobFlowAliveWhenNoSteps': True,
            'TerminationProtected': False,
        },
        'AutoTerminationPolicy': {'IdleTimeout': idle_timeout},
        'VisibleToAllUsers': True,
        'Tags': [
            {'Key': 'for-use-with-amazon-emr-managed-policies', 'Value': 'true'},
            {'Key': 'hailctl', 'Value': 'emr'},
            {'Key': 'hail-version', 'Value': artifact.hail_pip_version},
            {'Key': 'hail-revision', 'Value': artifact.hail_git_revision},
            {'Key': 'hail-spark-version', 'Value': artifact.spark_version},
        ],
    }
    if log_uri is not None:
        kwargs['LogUri'] = log_uri
    if ec2_key_name is not None:
        kwargs['Instances']['Ec2KeyName'] = ec2_key_name
    if subnet_id is not None:
        kwargs['Instances']['Ec2SubnetId'] = subnet_id
    if service_access_security_group is not None:
        kwargs['Instances']['ServiceAccessSecurityGroup'] = service_access_security_group
    if primary_security_group is not None:
        kwargs['Instances']['EmrManagedMasterSecurityGroup'] = primary_security_group
    if core_security_group is not None:
        kwargs['Instances']['EmrManagedSlaveSecurityGroup'] = core_security_group

    if use_default_roles:
        kwargs['ServiceRole'] = 'EMR_DefaultRole'
        kwargs['JobFlowRole'] = 'EMR_EC2_DefaultRole'
    else:
        if service_role is None or instance_profile is None:
            raise ValueError('Either pass --use-default-roles, or provide both --service-role and --instance-profile.')
        kwargs['ServiceRole'] = service_role
        kwargs['JobFlowRole'] = instance_profile

    return kwargs
