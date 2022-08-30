import re

import yaml

from . import gcloud
from .cluster_config import ClusterConfig

DEFAULT_PROPERTIES = {
    "spark:spark.task.maxFailures": "20",
    "spark:spark.driver.extraJavaOptions": "-Xss4M",
    "spark:spark.executor.extraJavaOptions": "-Xss4M",
    'spark:spark.speculation': 'true',
    "hdfs:dfs.replication": "1",
    'dataproc:dataproc.logging.stackdriver.enable': 'false',
    'dataproc:dataproc.monitoring.stackdriver.enable': 'false'
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
    'australia-southeast1': 'aus-sydney'
}

ANNOTATION_DB_BUCKETS = ["hail-datasets-us", "hail-datasets-eu"]

IMAGE_VERSION = '2.0.44-debian10'


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')

    # arguments with default parameters
    parser.add_argument('--master-machine-type', '--master', '-m', default='n1-highmem-8', type=str,
                        help='Master machine type (default: %(default)s).')
    parser.add_argument('--master-memory-fraction', default=0.8, type=float,
                        help='Fraction of master memory allocated to the JVM. '
                             'Use a smaller value to reserve more memory '
                             'for Python. (default: %(default)s)')
    parser.add_argument('--master-boot-disk-size', default=100, type=int,
                        help='Disk size of master machine, in GB (default: %(default)s).')
    parser.add_argument('--num-master-local-ssds', default=0, type=int,
                        help='Number of local SSDs to attach to the master machine (default: %(default)s).')
    parser.add_argument('--num-secondary-workers', '--num-preemptible-workers', '--n-pre-workers', '-p', default=0, type=int,
                        help='Number of secondary (preemptible) worker machines (default: %(default)s).')
    parser.add_argument('--num-worker-local-ssds', default=0, type=int,
                        help='Number of local SSDs to attach to each worker machine (default: %(default)s).')
    parser.add_argument('--num-workers', '--n-workers', '-w', default=2, type=int,
                        help='Number of worker machines (default: %(default)s).')
    parser.add_argument('--secondary-worker-boot-disk-size', '--preemptible-worker-boot-disk-size', default=40, type=int,
                        help='Disk size of secondary (preemptible) worker machines, in GB (default: %(default)s).')
    parser.add_argument('--worker-boot-disk-size', default=40, type=int,
                        help='Disk size of worker machines, in GB (default: %(default)s).')
    parser.add_argument('--worker-machine-type', '--worker',
                        help='Worker machine type (default: n1-standard-8, or n1-highmem-8 with --vep).')
    parser.add_argument('--region',
                        help='Compute region for the cluster.')
    parser.add_argument('--zone',
                        help='Compute zone for the cluster.')
    parser.add_argument('--properties',
                        help='Additional configuration properties for the cluster')
    parser.add_argument('--metadata',
                        help='Comma-separated list of metadata to add: KEY1=VALUE1,KEY2=VALUE2...')
    parser.add_argument('--packages', '--pkgs',
                        help='Comma-separated list of Python packages to be installed on the master node.')
    parser.add_argument('--project', help='Google Cloud project to start cluster (defaults to currently set project).')
    parser.add_argument('--configuration',
                        help='Google Cloud configuration to start cluster (defaults to currently set configuration).')
    parser.add_argument('--max-idle', type=str, help='If specified, maximum idle time before shutdown (e.g. 60m).')
    max_age_group = parser.add_mutually_exclusive_group()
    max_age_group.add_argument('--expiration-time', type=str, help='If specified, time at which cluster is shutdown (e.g. 2020-01-01T00:00:00Z).')
    max_age_group.add_argument('--max-age', type=str, help='If specified, maximum age before shutdown (e.g. 60m).')
    parser.add_argument('--bucket', type=str,
                        help='The Google Cloud Storage bucket to use for cluster staging (just the bucket name, no gs:// prefix).')
    parser.add_argument('--temp-bucket', type=str,
                        help='The Google Cloud Storage bucket to use for cluster temporary storage (just the bucket name, no gs:// prefix).')
    parser.add_argument('--network', type=str, help='the network for all nodes in this cluster')
    parser.add_argument('--service-account', type=str, help='The Google Service Account to use for cluster creation (default to the Compute Engine service account).')
    parser.add_argument('--master-tags', type=str, help='comma-separated list of instance tags to apply to the mastern node')
    parser.add_argument('--scopes', help='Specifies access scopes for the node instances')

    parser.add_argument('--wheel', help='Non-default Hail installation. Warning: experimental.')

    # initialization action flags
    parser.add_argument('--init', default='', help='Comma-separated list of init scripts to run.')
    parser.add_argument('--init_timeout', default='20m',
                        help='Flag to specify a timeout period for the initialization action')
    parser.add_argument('--vep',
                        help='Install VEP for the specified reference genome.',
                        required=False,
                        choices=['GRCh37', 'GRCh38'])
    parser.add_argument('--dry-run', action='store_true', help="Print gcloud dataproc command, but don't run it.")
    parser.add_argument('--no-off-heap-memory', action='store_true',
                        help="If true, don't partition JVM memory between hail heap and JVM heap")
    parser.add_argument('--big-executors', action='store_true',
                        help="If true, double memory allocated per executor, using half the cores of the cluster with an extra large memory allotment per core.")
    parser.add_argument('--off-heap-memory-fraction', type=float, default=0.6,
                        help="Minimum fraction of worker memory dedicated to off-heap Hail values.")
    parser.add_argument('--off-heap-memory-hard-limit', action='store_true',
                        help="If true, limit off-heap allocations to the dedicated fraction")
    parser.add_argument('--yarn-memory-fraction', type=float,
                        help="Fraction of machine memory to allocate to the yarn container scheduler.",
                        default=0.95)

    # requester pays
    parser.add_argument('--requester-pays-allow-all',
                        help="Allow reading from all requester-pays buckets.",
                        action='store_true',
                        required=False)
    parser.add_argument('--requester-pays-allow-buckets',
                        help="Comma-separated list of requester-pays buckets to allow reading from.")
    parser.add_argument('--requester-pays-allow-annotation-db',
                        action='store_true',
                        help="Allows reading from any of the requester-pays buckets that hold data for the annotation database.")
    parser.add_argument('--debug-mode',
                        action='store_true',
                        help="Enable debug features on created cluster (heap dump on out-of-memory error)")


async def main(args, pass_through_args):
    import pkg_resources  # pylint: disable=import-outside-toplevel

    conf = ClusterConfig()
    conf.extend_flag('image-version', IMAGE_VERSION)

    if not pkg_resources.resource_exists('hailtop.hailctl', "deploy.yaml"):
        raise RuntimeError("package has no 'deploy.yaml' file")
    deploy_metadata = yaml.safe_load(
        pkg_resources.resource_stream('hailtop.hailctl', "deploy.yaml"))['dataproc']

    conf.extend_flag('properties', DEFAULT_PROPERTIES)
    if args.properties:
        conf.parse_and_extend('properties', args.properties)

    if args.debug_mode:
        conf.extend_flag('properties', {
            "spark:spark.driver.extraJavaOptions": "-Xss4M -XX:+HeapDumpOnOutOfMemoryError -XX:-OmitStackTraceInFastThrow",
            "spark:spark.executor.extraJavaOptions": "-Xss4M -XX:+HeapDumpOnOutOfMemoryError -XX:-OmitStackTraceInFastThrow",
        })

    # default to highmem machines if using VEP
    if not args.worker_machine_type:
        args.worker_machine_type = 'n1-highmem-8' if args.vep else 'n1-standard-8'

    # default initialization script to start up cluster with
    conf.extend_flag('initialization-actions',
                     [deploy_metadata['init_notebook.py']])

    # requester pays support
    if args.requester_pays_allow_all or args.requester_pays_allow_buckets or args.requester_pays_allow_annotation_db:
        if args.requester_pays_allow_all and args.requester_pays_allow_buckets:
            raise RuntimeError("Cannot specify both 'requester_pays_allow_all' and 'requester_pays_allow_buckets")

        if args.requester_pays_allow_all:
            requester_pays_mode = "AUTO"
        else:
            requester_pays_mode = "CUSTOM"
            requester_pays_bucket_sources = []
            if args.requester_pays_allow_buckets:
                requester_pays_bucket_sources.append(args.requester_pays_allow_buckets)
            if args.requester_pays_allow_annotation_db:
                requester_pays_bucket_sources.extend(ANNOTATION_DB_BUCKETS)

            conf.extend_flag("properties", {"spark:spark.hadoop.fs.gs.requester.pays.buckets": ",".join(requester_pays_bucket_sources)})

        # Need to pick requester pays project.
        requester_pays_project = args.project if args.project else gcloud.get_config("project")

        conf.extend_flag("properties", {"spark:spark.hadoop.fs.gs.requester.pays.mode": requester_pays_mode,
                                        "spark:spark.hadoop.fs.gs.requester.pays.project.id": requester_pays_project})

    # gcloud version 277 and onwards requires you to specify a region. Let's just require it for all hailctl users for consistency.
    if args.region:
        project_region = args.region
    else:
        project_region = gcloud.get_config("dataproc/region")

    if not project_region:
        raise RuntimeError("Could not determine dataproc region. Use --region argument to hailctl, or use `gcloud config set dataproc/region <my-region>` to set a default.")

    # add VEP init script
    if args.vep:
        # VEP is too expensive if you have to pay egress charges. We must choose the right replicate.
        replicate = REGION_TO_REPLICATE_MAPPING.get(project_region)
        if replicate is None:
            raise RuntimeError(f"The --vep argument is not currently provided in your region.\n"
                               f"  Please contact the Hail team on https://discuss.hail.is for support.\n"
                               f"  Your region: {project_region}\n"
                               f"  Supported regions: {', '.join(REGION_TO_REPLICATE_MAPPING.keys())}")
        print(f"Pulling VEP data from bucket in {replicate}.")
        conf.extend_flag('metadata', {"VEP_REPLICATE": replicate})
        vep_config_path = "/vep_data/vep-gcloud.json"
        conf.extend_flag('metadata', {"VEP_CONFIG_PATH": vep_config_path, "VEP_CONFIG_URI": f"file://{vep_config_path}"})
        conf.extend_flag('initialization-actions', [deploy_metadata[f'vep-{args.vep}.sh']])
    # add custom init scripts
    if args.init:
        conf.extend_flag('initialization-actions', args.init.split(','))

    if args.metadata:
        conf.parse_and_extend('metadata', args.metadata)

    wheel = args.wheel or deploy_metadata['wheel']
    conf.extend_flag('metadata', {'WHEEL': wheel})

    # if Python packages requested, add metadata variable
    packages = deploy_metadata['pip_dependencies'].strip('|').split('|||')
    metadata_pkgs = conf.flags['metadata'].get('PKGS')
    split_regex = r'[|,]'
    if metadata_pkgs:
        packages.extend(re.split(split_regex, metadata_pkgs))
    if args.packages:
        packages.extend(re.split(split_regex, args.packages))
    conf.extend_flag('metadata', {'PKGS': '|'.join(set(packages))})

    def disk_size(size):
        if args.vep:
            size = max(size, 200)
        return str(size) + 'GB'

    conf.extend_flag('properties',
                     {"spark:spark.driver.memory": "{driver_memory}g".format(
                         driver_memory=str(int(MACHINE_MEM[args.master_machine_type] * args.master_memory_fraction)))})
    conf.flags['master-machine-type'] = args.master_machine_type
    conf.flags['master-boot-disk-size'] = '{}GB'.format(args.master_boot_disk_size)
    conf.flags['num-master-local-ssds'] = args.num_master_local_ssds
    conf.flags['num-secondary-workers'] = args.num_secondary_workers
    conf.flags['num-worker-local-ssds'] = args.num_worker_local_ssds
    conf.flags['num-workers'] = args.num_workers
    conf.flags['secondary-worker-boot-disk-size'] = disk_size(args.secondary_worker_boot_disk_size)
    conf.flags['worker-boot-disk-size'] = disk_size(args.worker_boot_disk_size)
    conf.flags['worker-machine-type'] = args.worker_machine_type

    if not args.no_off_heap_memory:
        worker_memory = MACHINE_MEM[args.worker_machine_type]

        # A Google support engineer recommended the strategy of passing the YARN
        # config params, and the default value of 95% of machine memory to give to YARN.
        # yarn.nodemanager.resource.memory-mb - total memory per machine
        # yarn.scheduler.maximum-allocation-mb - max memory to allocate to each container
        available_memory_fraction = args.yarn_memory_fraction
        available_memory_mb = int(worker_memory * available_memory_fraction * 1024)
        cores_per_machine = int(args.worker_machine_type.split('-')[-1])
        executor_cores = min(cores_per_machine, 4)
        available_memory_per_core_mb = available_memory_mb // cores_per_machine

        memory_per_executor_mb = int(available_memory_per_core_mb * executor_cores)

        off_heap_mb = int(memory_per_executor_mb * args.off_heap_memory_fraction)
        on_heap_mb = memory_per_executor_mb - off_heap_mb

        if args.off_heap_memory_hard_limit:
            off_heap_memory_per_core = off_heap_mb // executor_cores
        else:
            off_heap_memory_per_core = available_memory_per_core_mb

        print(f"hailctl dataproc: Creating a cluster with workers of machine type {args.worker_machine_type}.\n"
              f"  Allocating {memory_per_executor_mb} MB of memory per executor ({executor_cores} cores),\n"
              f"  with at least {off_heap_mb} MB for Hail off-heap values and {on_heap_mb} MB for the JVM."
              f"  Using a maximum Hail memory reservation of {off_heap_memory_per_core} MB per core.")

        conf.extend_flag('properties',
                         {
                             'yarn:yarn.nodemanager.resource.memory-mb': f'{available_memory_mb}',
                             'yarn:yarn.scheduler.maximum-allocation-mb': f'{executor_cores * available_memory_per_core_mb}',
                             'spark:spark.executor.cores': f'{executor_cores}',
                             'spark:spark.executor.memory': f'{on_heap_mb}m',
                             'spark:spark.executor.memoryOverhead': f'{off_heap_mb}m',
                             'spark:spark.memory.storageFraction': '0.2',
                             'spark:spark.executorEnv.HAIL_WORKER_OFF_HEAP_MEMORY_PER_CORE_MB': str(
                                 off_heap_memory_per_core),
                         }
                         )

    if args.region:
        conf.flags['region'] = args.region
    if args.zone:
        conf.flags['zone'] = args.zone
    conf.flags['initialization-action-timeout'] = args.init_timeout
    if args.network:
        conf.flags['network'] = args.network
    if args.configuration:
        conf.flags['configuration'] = args.configuration
    if args.project:
        conf.flags['project'] = args.project
    if args.bucket:
        conf.flags['bucket'] = args.bucket
    if args.temp_bucket:
        conf.flags['temp-bucket'] = args.bucket
    if args.scopes:
        conf.flags['scopes'] = args.scopes

    account = gcloud.get_config("account")
    if account:
        conf.flags['labels'] = 'creator=' + re.sub(r'[^0-9a-z_\-]', '_', account.lower())[:63]

    # rewrite metadata and properties to escape them
    conf.flags['metadata'] = '^|||^' + '|||'.join(f'{k}={v}' for k, v in conf.flags['metadata'].items())
    conf.flags['properties'] = '^|||^' + '|||'.join(f'{k}={v}' for k, v in conf.flags['properties'].items())

    # command to start cluster
    cmd = conf.get_command(args.name)

    if args.beta:
        cmd.insert(1, 'beta')
    if args.max_idle:
        cmd.append('--max-idle={}'.format(args.max_idle))
    if args.max_age:
        cmd.append('--max-age={}'.format(args.max_age))
    if args.expiration_time:
        cmd.append('--expiration_time={}'.format(args.expiration_time))
    if args.service_account:
        cmd.append('--service-account={}'.format(args.service_account))

    cmd.extend(pass_through_args)

    # print underlying gcloud command
    print(' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[5:]))

    # spin up cluster
    if not args.dry_run:
        print("Starting cluster '{}'...".format(args.name))
        gcloud.run(cmd[1:])

    if args.master_tags:
        add_tags_command = ['compute', 'instances', 'add-tags', args.name + '-m', '--tags', args.master_tags]

        if args.project:
            add_tags_command.append(f"--project={args.project}")
        if args.zone:
            add_tags_command.append(f"--zone={args.zone}")

        print('gcloud ' + ' '.join(add_tags_command))
        if not args.dry_run:
            gcloud.run(add_tags_command)
