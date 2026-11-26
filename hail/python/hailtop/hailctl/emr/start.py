import copy
import hashlib
import importlib.resources as ir
import json
import os
from typing import Optional
from urllib.parse import urlparse

import typer

from . import emr

# v1 supports exactly one EMR release.
EMR_RELEASE_LABEL = 'emr-spark-8.1.0'
EMR_PYTHON = '/usr/bin/python3.12'
HAIL_JAR_PATH = '/usr/lib/hail/hail-all-spark.jar'
OFF_HEAP_MEMORY_PER_CORE_MB = 1024

S3A_FILESYSTEM_CLASS = 'org.apache.hadoop.fs.s3a.S3AFileSystem'
S3A_CORE_SITE_PROPERTIES = {
    'fs.s3.impl': S3A_FILESYSTEM_CLASS,
    'fs.s3n.impl': S3A_FILESYSTEM_CLASS,
    'fs.s3a.impl': S3A_FILESYSTEM_CLASS,
    'mapreduce.outputcommitter.factory.scheme.s3a': 'org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory',
    'fs.s3a.committer.magic.enabled': 'true',
    'fs.s3a.committer.name': 'magicv2',
    'fs.s3a.committer.magic.track.commits.in.memory.enabled': 'true',
    'fs.s3a.committer.magic.overwrite.and.commit': 'true',
}


def hail_configurations() -> list[dict]:
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
        'spark.executorEnv.HAIL_WORKER_OFF_HEAP_MEMORY_PER_CORE_MB': str(OFF_HEAP_MEMORY_PER_CORE_MB),
        'spark.pyspark.python': EMR_PYTHON,
    }
    return [
        {'Classification': 'core-site', 'Properties': copy.deepcopy(S3A_CORE_SITE_PROPERTIES)},
        {'Classification': 'spark-defaults', 'Properties': spark_defaults},
        {'Classification': 'spark', 'Properties': {'maximizeResourceAllocation': 'true'}},
        {
            'Classification': 'spark-env',
            'Configurations': [
                {
                    'Classification': 'export',
                    'Properties': {
                        'HAIL_CLOUD': 'aws',
                        'PYSPARK_PYTHON': EMR_PYTHON,
                        'SPARK_DIST_CLASSPATH': '$(hadoop classpath)',
                    },
                }
            ],
            'Properties': {},
        },
    ]


def check_wheel_uri(wheel: str) -> None:
    parsed = urlparse(wheel)
    filename = os.path.basename(parsed.path)
    if parsed.scheme != 's3' or not parsed.netloc or not (filename.startswith('hail-') and filename.endswith('.whl')):
        raise typer.BadParameter(
            '--wheel must be an S3 URI of a Hail wheel, e.g. s3://my-bucket/hail-0.2.140-py3-none-any.whl'
        )


def bootstrap_script() -> bytes:
    return ir.files('hailtop.hailctl.emr').joinpath('resources/install-hail-emr.sh').read_bytes()


def bootstrap_uri(wheel: str, script: bytes) -> str:
    """Content-addressed location next to the wheel, so the bootstrap needs no extra bucket."""
    return f"{wheel.rsplit('/', 1)[0]}/bootstrap/install-hail-emr-{hashlib.sha256(script).hexdigest()}.sh"


def build_run_job_flow_kwargs(
    *,
    cluster_name: str,
    bootstrap_s3_uri: str,
    wheel: str,
    num_workers: int,
    master_instance_type: str,
    worker_instance_type: str,
    subnet_id: Optional[str],
    service_access_security_group: Optional[str],
    log_uri: Optional[str],
    idle_timeout: int,
) -> dict:
    instances: dict = {
        'InstanceGroups': [
            {'Name': 'Primary', 'InstanceRole': 'MASTER', 'InstanceType': master_instance_type, 'InstanceCount': 1},
            {
                'Name': 'Core',
                'InstanceRole': 'CORE',
                'InstanceType': worker_instance_type,
                'InstanceCount': num_workers,
            },
        ],
        'KeepJobFlowAliveWhenNoSteps': True,
        'TerminationProtected': False,
    }
    if subnet_id is not None:
        instances['Ec2SubnetId'] = subnet_id
    if service_access_security_group is not None:
        instances['ServiceAccessSecurityGroup'] = service_access_security_group
    kwargs: dict = {
        'Name': cluster_name,
        'ReleaseLabel': EMR_RELEASE_LABEL,
        'Applications': [{'Name': 'Spark'}],
        'Configurations': hail_configurations(),
        'BootstrapActions': [
            {'Name': 'install-hail', 'ScriptBootstrapAction': {'Path': bootstrap_s3_uri, 'Args': [wheel]}}
        ],
        'Instances': instances,
        'AutoTerminationPolicy': {'IdleTimeout': idle_timeout},
        'ServiceRole': 'EMR_DefaultRole',
        'JobFlowRole': 'EMR_EC2_DefaultRole',
        'VisibleToAllUsers': True,
        'Tags': [
            {'Key': 'for-use-with-amazon-emr-managed-policies', 'Value': 'true'},
            {'Key': 'hailctl', 'Value': 'emr'},
        ],
    }
    if log_uri is not None:
        kwargs['LogUri'] = log_uri
    return kwargs


def start(
    cluster_name: str,
    wheel: str,
    region: Optional[str],
    subnet_id: Optional[str],
    service_access_security_group: Optional[str],
    num_workers: int,
    master_instance_type: str,
    worker_instance_type: str,
    log_uri: Optional[str],
    idle_timeout: int,
    dry_run: bool,
) -> None:
    check_wheel_uri(wheel)
    if idle_timeout < 60 or idle_timeout > 604800:
        raise typer.BadParameter('--idle-timeout must be between 60 and 604800 seconds.')
    script = bootstrap_script()
    bootstrap_s3_uri = bootstrap_uri(wheel, script)
    kwargs = build_run_job_flow_kwargs(
        cluster_name=cluster_name,
        bootstrap_s3_uri=bootstrap_s3_uri,
        wheel=wheel,
        num_workers=num_workers,
        master_instance_type=master_instance_type,
        worker_instance_type=worker_instance_type,
        subnet_id=subnet_id,
        service_access_security_group=service_access_security_group,
        log_uri=log_uri,
        idle_timeout=idle_timeout,
    )
    if dry_run:
        print(json.dumps(kwargs, indent=2))
        return
    client = emr.emr_client(region)
    emr.upload_to_s3(bootstrap_s3_uri, script)
    resp = client.run_job_flow(**kwargs)
    print(f"Started cluster {resp['JobFlowId']}.")
