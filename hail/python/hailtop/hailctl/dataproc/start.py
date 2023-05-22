import re
from enum import Enum

import yaml

from typing import Optional, List

from . import gcloud
from .cluster_config import ClusterConfig


class VepVersion(str, Enum):
    GRCH37 = 'GRCh37'
    GRCH38 = 'GRCh38'


DEFAULT_PROPERTIES = {
    "spark:spark.task.maxFailures": "20",
    "spark:spark.driver.extraJavaOptions": "-Xss4M",
    "spark:spark.executor.extraJavaOptions": "-Xss4M",
    'spark:spark.speculation': 'true',
    "hdfs:dfs.replication": "1",
    'dataproc:dataproc.logging.stackdriver.enable': 'false',
    'dataproc:dataproc.monitoring.stackdriver.enable': 'false',
}

# leadre (master) machine type to memory map, used for setting
# spark.driver.memory property
MACHINE_MEM = {
    'n1-standard-1': 3.75,
    'n1-standard-2': 7.5,
    'n1-standard-4': 15,
    'n1-standard-8': 30,
    'n1-standard-16': 60,
    'n1-standard-32': 120,
    'n1-standard-64': 240,
    'n1-highmem-2': 13,
    'n1-highmem-4': 26,
    'n1-highmem-8': 52,
    'n1-highmem-16': 104,
    'n1-highmem-32': 208,
    'n1-highmem-64': 416,
    'n1-highcpu-2': 1.8,
    'n1-highcpu-4': 3.6,
    'n1-highcpu-8': 7.2,
    'n1-highcpu-16': 14.4,
    'n1-highcpu-32': 28.8,
    'n1-highcpu-64': 57.6,
    'n2-standard-2': 8,
    'n2-standard-4': 16,
    'n2-standard-8': 32,
    'n2-standard-16': 64,
    'n2-standard-32': 128,
    'n2-standard-48': 192,
    'n2-standard-64': 256,
    'n2-standard-80': 320,
    'n2-highmem-2': 16,
    'n2-highmem-4': 32,
    'n2-highmem-8': 64,
    'n2-highmem-16': 128,
    'n2-highmem-32': 256,
    'n2-highmem-48': 384,
    'n2-highmem-64': 512,
    'n2-highmem-80': 640,
    'n2-highcpu-2': 2,
    'n2-highcpu-4': 4,
    'n2-highcpu-8': 8,
    'n2-highcpu-16': 16,
    'n2-highcpu-32': 32,
    'n2-highcpu-48': 48,
    'n2-highcpu-64': 64,
    'n2-highcpu-80': 80,
    'n2d-standard-2': 8,
    'n2d-standard-4': 16,
    'n2d-standard-8': 32,
    'n2d-standard-16': 64,
    'n2d-standard-32': 128,
    'n2d-standard-48': 192,
    'n2d-standard-64': 256,
    'n2d-standard-80': 320,
    'n2d-standard-96': 384,
    'n2d-standard-128': 512,
    'n2d-standard-224': 896,
    'n2d-highmem-2': 16,
    'n2d-highmem-4': 32,
    'n2d-highmem-8': 64,
    'n2d-highmem-16': 128,
    'n2d-highmem-32': 256,
    'n2d-highmem-48': 384,
    'n2d-highmem-64': 512,
    'n2d-highmem-80': 640,
    'n2d-highmem-96': 786,
    'n2d-highcpu-2': 2,
    'n2d-highcpu-4': 4,
    'n2d-highcpu-8': 8,
    'n2d-highcpu-16': 16,
    'n2d-highcpu-32': 32,
    'n2d-highcpu-48': 48,
    'n2d-highcpu-64': 64,
    'n2d-highcpu-80': 80,
    'n2d-highcpu-96': 96,
    'n2d-highcpu-128': 128,
    'n2d-highcpu-224': 224,
    'e2-standard-2': 8,
    'e2-standard-4': 16,
    'e2-standard-8': 32,
    'e2-standard-16': 64,
    'e2-highmem-2': 16,
    'e2-highmem-4': 32,
    'e2-highmem-8': 64,
    'e2-highmem-16': 128,
    'e2-highcpu-2': 2,
    'e2-highcpu-4': 4,
    'e2-highcpu-8': 8,
    'e2-highcpu-16': 16,
    'm1-ultramem-40': 961,
    'm1-ultramem-80': 1922,
    'm1-ultramem-160': 3844,
    'm1-megamem-96': 1433,
    'm2-ultramem-2084': 5888,
    'm2-ultramem-4164': 11776,
    'c2-standard-4': 16,
    'c2-standard-8': 32,
    'c2-standard-16': 64,
    'c2-standard-30': 120,
    'c2-standard-60': 240,
}

REGION_TO_REPLICATE_MAPPING = {
    'us-central1': 'us',
    'us-east1': 'us',
    'us-east4': 'us',
    'us-west1': 'us',
    'us-west2': 'us',
    'us-west3': 'us',
    # Europe != EU
    'europe-north1': 'eu',
    'europe-west1': 'eu',
    'europe-west2': 'uk',
    'europe-west3': 'eu',
    'europe-west4': 'eu',
    'australia-southeast1': 'aus-sydney',
}

ANNOTATION_DB_BUCKETS = ["hail-datasets-us", "hail-datasets-eu"]

IMAGE_VERSION = '2.1.2-debian11'


def start(
    name: str,
    pass_through_args: List[str],
    master_machine_type: str,
    master_memory_fraction: float,
    master_boot_disk_size: int,
    num_master_local_ssds: int,
    num_secondary_workers: int,
    num_worker_local_ssds: int,
    num_workers: int,
    secondary_worker_boot_disk_size: int,
    worker_boot_disk_size: int,
    worker_machine_type: Optional[str],
    region: Optional[str],
    zone: Optional[str],
    properties: Optional[str],
    metadata: Optional[str],
    packages: Optional[str],
    project: Optional[str],
    configuration: Optional[str],
    max_idle: Optional[str],
    expiration_time: Optional[str],
    max_age: Optional[str],
    bucket: Optional[str],
    temp_bucket: Optional[str],
    network: Optional[str],
    subnet: Optional[str],
    service_account: Optional[str],
    master_tags: Optional[str],
    scopes: Optional[str],
    wheel: Optional[str],
    init: str,
    init_timeout: str,
    vep: Optional[VepVersion],
    dry_run: bool,
    no_off_heap_memory: bool,
    big_executors: bool,  # pylint: disable=unused-argument
    off_heap_memory_fraction: float,
    off_heap_memory_hard_limit: bool,
    yarn_memory_fraction: float,
    requester_pays_allow_all: bool,
    requester_pays_allow_buckets: Optional[str],
    requester_pays_allow_annotation_db: bool,
    debug_mode: bool,
    beta: bool,
):
    import pkg_resources  # pylint: disable=import-outside-toplevel

    conf = ClusterConfig()
    conf.extend_flag('image-version', IMAGE_VERSION)

    if not pkg_resources.resource_exists('hailtop.hailctl', "deploy.yaml"):
        raise RuntimeError("package has no 'deploy.yaml' file")
    deploy_metadata = yaml.safe_load(pkg_resources.resource_stream('hailtop.hailctl', "deploy.yaml"))['dataproc']

    conf.extend_flag('properties', DEFAULT_PROPERTIES)
    if properties:
        conf.parse_and_extend('properties', properties)

    if debug_mode:
        conf.extend_flag(
            'properties',
            {
                "spark:spark.driver.extraJavaOptions": "-Xss4M -XX:+HeapDumpOnOutOfMemoryError -XX:-OmitStackTraceInFastThrow",
                "spark:spark.executor.extraJavaOptions": "-Xss4M -XX:+HeapDumpOnOutOfMemoryError -XX:-OmitStackTraceInFastThrow",
            },
        )

    # default to highmem machines if using VEP
    if not worker_machine_type:
        worker_machine_type = 'n1-highmem-8' if vep else 'n1-standard-8'

    # default initialization script to start up cluster with
    conf.extend_flag('initialization-actions', [deploy_metadata['init_notebook.py']])

    # requester pays support
    if requester_pays_allow_all or requester_pays_allow_buckets or requester_pays_allow_annotation_db:
        if requester_pays_allow_all and requester_pays_allow_buckets:
            raise RuntimeError("Cannot specify both 'requester_pays_allow_all' and 'requester_pays_allow_buckets")

        if requester_pays_allow_all:
            requester_pays_mode = "AUTO"
        else:
            requester_pays_mode = "CUSTOM"
            requester_pays_bucket_sources = []
            if requester_pays_allow_buckets:
                requester_pays_bucket_sources.append(requester_pays_allow_buckets)
            if requester_pays_allow_annotation_db:
                requester_pays_bucket_sources.extend(ANNOTATION_DB_BUCKETS)

            conf.extend_flag(
                "properties",
                {"spark:spark.hadoop.fs.gs.requester.pays.buckets": ",".join(requester_pays_bucket_sources)},
            )

        # Need to pick requester pays project.
        requester_pays_project = project if project else gcloud.get_config("project")

        conf.extend_flag(
            "properties",
            {
                "spark:spark.hadoop.fs.gs.requester.pays.mode": requester_pays_mode,
                "spark:spark.hadoop.fs.gs.requester.pays.project.id": requester_pays_project,
            },
        )

    # gcloud version 277 and onwards requires you to specify a region. Let's just require it for all hailctl users for consistency.
    if region:
        project_region = region
    else:
        maybe_project_region = gcloud.get_config("dataproc/region")
        if not maybe_project_region:
            raise RuntimeError(
                "Could not determine dataproc region. Use --region argument to hailctl, or use `gcloud config set dataproc/region <my-region>` to set a default."
            )
        project_region = maybe_project_region

    # add VEP init script
    if vep:
        # VEP is too expensive if you have to pay egress charges. We must choose the right replicate.
        replicate = REGION_TO_REPLICATE_MAPPING.get(project_region)
        if replicate is None:
            raise RuntimeError(
                f"The --vep argument is not currently provided in your region.\n"
                f"  Please contact the Hail team on https://discuss.hail.is for support.\n"
                f"  Your region: {project_region}\n"
                f"  Supported regions: {', '.join(REGION_TO_REPLICATE_MAPPING.keys())}"
            )
        print(f"Pulling VEP data from bucket in {replicate}.")
        conf.extend_flag('metadata', {"VEP_REPLICATE": replicate})
        vep_config_path = "/vep_data/vep-gcloud.json"
        conf.extend_flag(
            'metadata', {"VEP_CONFIG_PATH": vep_config_path, "VEP_CONFIG_URI": f"file://{vep_config_path}"}
        )
        conf.extend_flag('initialization-actions', [deploy_metadata[f'vep-{vep.value}.sh']])
    # add custom init scripts
    if init:
        conf.extend_flag('initialization-actions', init.split(','))

    if metadata:
        conf.parse_and_extend('metadata', metadata)

    wheel = wheel or deploy_metadata['wheel']
    conf.extend_flag('metadata', {'WHEEL': wheel})

    # if Python packages requested, add metadata variable
    hail_packages = deploy_metadata['pip_dependencies'].strip('|').split('|||')
    metadata_pkgs = conf.flags['metadata'].get('PKGS')
    split_regex = r'[|,]'
    if metadata_pkgs:
        hail_packages.extend(re.split(split_regex, metadata_pkgs))
    if packages:
        hail_packages.extend(re.split(split_regex, packages))
    conf.extend_flag('metadata', {'PKGS': '|'.join(set(hail_packages))})

    def disk_size(size):
        if vep:
            size = max(size, 200)
        return str(size) + 'GB'

    conf.extend_flag(
        'properties',
        {
            "spark:spark.driver.memory": "{driver_memory}g".format(
                driver_memory=str(int(MACHINE_MEM[master_machine_type] * master_memory_fraction))
            )
        },
    )
    conf.flags['master-machine-type'] = master_machine_type
    conf.flags['master-boot-disk-size'] = '{}GB'.format(master_boot_disk_size)
    conf.flags['num-master-local-ssds'] = num_master_local_ssds
    conf.flags['num-secondary-workers'] = num_secondary_workers
    conf.flags['num-worker-local-ssds'] = num_worker_local_ssds
    conf.flags['num-workers'] = num_workers
    conf.flags['secondary-worker-boot-disk-size'] = disk_size(secondary_worker_boot_disk_size)
    conf.flags['worker-boot-disk-size'] = disk_size(worker_boot_disk_size)
    conf.flags['worker-machine-type'] = worker_machine_type

    if not no_off_heap_memory:
        worker_memory = MACHINE_MEM[worker_machine_type]

        # A Google support engineer recommended the strategy of passing the YARN
        # config params, and the default value of 95% of machine memory to give to YARN.
        # yarn.nodemanager.resource.memory-mb - total memory per machine
        # yarn.scheduler.maximum-allocation-mb - max memory to allocate to each container
        available_memory_fraction = yarn_memory_fraction
        available_memory_mb = int(worker_memory * available_memory_fraction * 1024)
        cores_per_machine = int(worker_machine_type.split('-')[-1])
        executor_cores = min(cores_per_machine, 4)
        available_memory_per_core_mb = available_memory_mb // cores_per_machine

        memory_per_executor_mb = int(available_memory_per_core_mb * executor_cores)

        off_heap_mb = int(memory_per_executor_mb * off_heap_memory_fraction)
        on_heap_mb = memory_per_executor_mb - off_heap_mb

        if off_heap_memory_hard_limit:
            off_heap_memory_per_core = off_heap_mb // executor_cores
        else:
            off_heap_memory_per_core = available_memory_per_core_mb

        print(
            f"hailctl dataproc: Creating a cluster with workers of machine type {worker_machine_type}.\n"
            f"  Allocating {memory_per_executor_mb} MB of memory per executor ({executor_cores} cores),\n"
            f"  with at least {off_heap_mb} MB for Hail off-heap values and {on_heap_mb} MB for the JVM."
            f"  Using a maximum Hail memory reservation of {off_heap_memory_per_core} MB per core."
        )

        conf.extend_flag(
            'properties',
            {
                'yarn:yarn.nodemanager.resource.memory-mb': f'{available_memory_mb}',
                'yarn:yarn.scheduler.maximum-allocation-mb': f'{executor_cores * available_memory_per_core_mb}',
                'spark:spark.executor.cores': f'{executor_cores}',
                'spark:spark.executor.memory': f'{on_heap_mb}m',
                'spark:spark.executor.memoryOverhead': f'{off_heap_mb}m',
                'spark:spark.memory.storageFraction': '0.2',
                'spark:spark.executorEnv.HAIL_WORKER_OFF_HEAP_MEMORY_PER_CORE_MB': str(off_heap_memory_per_core),
            },
        )

    if region:
        conf.flags['region'] = region
    if zone:
        conf.flags['zone'] = zone
    conf.flags['initialization-action-timeout'] = init_timeout
    if network and subnet:
        raise RuntimeError("Cannot define both 'network' and 'subnet' at the same time.")
    if network:
        conf.flags['network'] = network
    if subnet:
        conf.flags['subnet'] = subnet
    if configuration:
        conf.flags['configuration'] = configuration
    if project:
        conf.flags['project'] = project
    if bucket:
        conf.flags['bucket'] = bucket
    if temp_bucket:
        conf.flags['temp-bucket'] = temp_bucket
    if scopes:
        conf.flags['scopes'] = scopes

    account = gcloud.get_config("account")
    if account:
        conf.flags['labels'] = 'creator=' + re.sub(r'[^0-9a-z_\-]', '_', account.lower())[:63]

    # rewrite metadata and properties to escape them
    conf.flags['metadata'] = '^|||^' + '|||'.join(f'{k}={v}' for k, v in conf.flags['metadata'].items())
    conf.flags['properties'] = '^|||^' + '|||'.join(f'{k}={v}' for k, v in conf.flags['properties'].items())

    # command to start cluster
    cmd = conf.get_command(name)

    if beta:
        cmd.insert(1, 'beta')
    if max_idle:
        cmd.append('--max-idle={}'.format(max_idle))
    if max_age:
        cmd.append('--max-age={}'.format(max_age))
    if expiration_time:
        cmd.append('--expiration_time={}'.format(expiration_time))
    if service_account:
        cmd.append('--service-account={}'.format(service_account))

    cmd.extend(pass_through_args)

    # print underlying gcloud command
    print(' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[5:]))

    # spin up cluster
    if not dry_run:
        print("Starting cluster '{}'...".format(name))
        gcloud.run(cmd[1:])

    if master_tags:
        add_tags_command = ['compute', 'instances', 'add-tags', name + '-m', '--tags', master_tags]

        if project:
            add_tags_command.append(f"--project={project}")
        if zone:
            add_tags_command.append(f"--zone={zone}")

        print('gcloud ' + ' '.join(add_tags_command))
        if not dry_run:
            gcloud.run(add_tags_command)
