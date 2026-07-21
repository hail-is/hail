import copy
import sys
from typing import Dict, List, Optional

# EMR release label -> the Spark version that release ships.
# To support a new EMR release, add one entry here.
EMR_RELEASE_SPARK_VERSION: Dict[str, str] = {
    'emr-7.3.0': '3.5.3',  # matches Hail's SPARK_VERSION
    'emr-7.2.0': '3.5.2',
    'emr-7.1.0': '3.5.1',
    'emr-7.0.0': '3.5.0',
    # EMR 6.x ships Spark 3.4.x — incompatible with Hail's Spark 3.5.x requirement
    'emr-6.15.0': '3.4.1',
    'emr-6.14.0': '3.4.1',
    'emr-6.13.0': '3.4.1',
}
DEFAULT_EMR_RELEASE = 'emr-7.3.0'
HAIL_REQUIRED_SPARK_MINOR = '3.5'
HAIL_JAR_PATH = '/usr/lib/hail/hail-all-spark.jar'


def check_release_spark_compatibility(release_label: str) -> None:
    spark_version = EMR_RELEASE_SPARK_VERSION.get(release_label)
    if spark_version is None:
        print(
            f'Warning: unknown EMR release {release_label!r}. Hail requires Spark '
            f'{HAIL_REQUIRED_SPARK_MINOR}.x; proceeding without a compatibility check.',
            file=sys.stderr,
        )
        return
    minor = spark_version.rsplit('.', 1)[0]
    if minor != HAIL_REQUIRED_SPARK_MINOR:
        raise ValueError(
            f'EMR release {release_label!r} ships Spark {spark_version}, but Hail '
            f'requires Spark {HAIL_REQUIRED_SPARK_MINOR}.x.'
        )


def hail_configurations(off_heap_memory_per_core_mb: Optional[int]) -> List[dict]:
    spark_defaults = {
        'spark.jars': f'local://{HAIL_JAR_PATH}',
        'spark.driver.extraClassPath': HAIL_JAR_PATH,
        'spark.executor.extraClassPath': HAIL_JAR_PATH,
        'spark.task.maxFailures': '20',
        'spark.driver.extraJavaOptions': '-Xss4M',
        'spark.executor.extraJavaOptions': '-Xss4M',
        'spark.executorEnv.PYTHONHASHSEED': '0',
        'spark.yarn.appMasterEnv.PYTHONHASHSEED': '0',
        'spark.executorEnv.HAIL_CLOUD': 'aws',
        'spark.yarn.appMasterEnv.HAIL_CLOUD': 'aws',
    }
    if off_heap_memory_per_core_mb is not None:
        spark_defaults['spark.executorEnv.HAIL_WORKER_OFF_HEAP_MEMORY_PER_CORE_MB'] = str(
            off_heap_memory_per_core_mb
        )
    return [
        {'Classification': 'spark-defaults', 'Properties': spark_defaults},
        {'Classification': 'spark', 'Properties': {'maximizeResourceAllocation': 'true'}},
        {
            # spark-env.sh is sourced by every spark-submit invocation, so this is
            # what actually delivers HAIL_CLOUD to a client-deploy-mode driver
            # (executorEnv/appMasterEnv do not reach it).
            'Classification': 'spark-env',
            'Configurations': [{'Classification': 'export', 'Properties': {'HAIL_CLOUD': 'aws'}}],
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


def build_run_job_flow_kwargs(
    *,
    cluster_name: str,
    release_label: str,
    master_instance_type: str,
    core_instance_type: str,
    core_instance_count: int,
    ec2_key_name: Optional[str],
    subnet_id: Optional[str],
    log_uri: Optional[str],
    bootstrap_s3_uri: str,
    pip_version: str,
    off_heap_memory_per_core_mb: Optional[int],
    use_default_roles: bool,
    service_role: Optional[str],
    instance_profile: Optional[str],
) -> dict:
    kwargs: dict = {
        'Name': cluster_name,
        'ReleaseLabel': release_label,
        'Applications': [{'Name': 'Spark'}],
        'Configurations': hail_configurations(off_heap_memory_per_core_mb),
        'BootstrapActions': [
            {
                'Name': 'install-hail',
                'ScriptBootstrapAction': {'Path': bootstrap_s3_uri, 'Args': [pip_version]},
            }
        ],
        'Instances': {
            'InstanceGroups': [
                {
                    'Name': 'Master',
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
        'VisibleToAllUsers': True,
    }
    if log_uri is not None:
        kwargs['LogUri'] = log_uri
    if ec2_key_name is not None:
        kwargs['Instances']['Ec2KeyName'] = ec2_key_name
    if subnet_id is not None:
        kwargs['Instances']['Ec2SubnetId'] = subnet_id

    if use_default_roles:
        kwargs['ServiceRole'] = 'EMR_DefaultRole'
        kwargs['JobFlowRole'] = 'EMR_EC2_DefaultRole'
    else:
        if service_role is None or instance_profile is None:
            raise ValueError(
                'Either pass --use-default-roles, or provide both --service-role and --instance-profile.'
            )
        kwargs['ServiceRole'] = service_role
        kwargs['JobFlowRole'] = instance_profile

    return kwargs
