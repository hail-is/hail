import re
import pkg_resources
import yaml
import click

from . import gcloud
from .dataproc import dataproc
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

ANNOTATION_DB_BUCKETS = ["hail-datasets-us", "hail-datasets-eu", "gnomad-public-requester-pays"]

IMAGE_VERSION = '1.4-debian9'


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'start',
        help='Start a Dataproc cluster configured for Hail.',
        description='Start a Dataproc cluster configured for Hail.')
    parser.set_defaults(module='hailctl dataproc start')


@dataproc.command(
    help="Start a Dataproc cluster configured for Hail.")
@click.argument('cluster_name')
@click.option('--master-machine-type', '--master', '-m',
              default='n1-highmem-8', show_default=True,
              help="Master machine type.")
@click.option('--master-memory-fraction',
              type=float, default=0.8, show_default=True,
              help=("Fraction of master memory allocated to the JVM. "
                    "Use a smaller value to reserve more memory "
                    "for Python."))
@click.option('--master-boot-disk-size',
              type=int, default=100, show_default=True,
              help="Disk size of master machine, in GB.")
@click.option('--num-master-local-ssds',
              type=int, default=0, show_default=True,
              help="Number of local SSDs to attach to the master machine.")
@click.option('--num-secondary-workers', '--num-preemptible-workers', '--n-pre-workers', '-p',
              type=int, default=0, show_default=True,
              help="Number of secondary (preemptible) worker machines.")
@click.option('--num-worker-local-ssds',
              type=int, default=0, show_default=True,
              help="Number of local SSDs to attach to each worker machine.")
@click.option('--num-workers', '--n-workers', '-w',
              type=int, default=2, show_default=True,
              help="Number of worker machines.")
@click.option('--secondary-worker-boot-disk-size', '--preemptible-worker-boot-disk-size',
              type=int, default=40, show_default=True,
              help="Disk size of secondary (preemptible) worker machines, in GB.")
@click.option('--worker-boot-disk-size',
              type=int, default=40, show_default=True,
              help="Disk size of worker machines, in GB.")
@click.option('--worker-machine-type', '--worker',
              help="Worker machine type. [default: (n1-standard-8, or n1-highmem-8 with --vep)]")
@click.option('--region',
              help="Compute region for the cluster.")
@click.option('--zone',
              help="Compute zone for the cluster.")
@click.option('--properties',
              help="Additional configuration properties for the cluster")
@click.option('--metadata',
              help="Comma-separated list of metadata to add: KEY1=VALUE1,KEY2=VALUE2...")
@click.option('--packages', '--pkgs',
              help="Comma-separated list of Python packages to be installed on the master node.")
@click.option('--project',
              help='Google Cloud project to start cluster. [default: (currently set project)]')
@click.option('--configuration',
              help='Google Cloud configuration to start cluster. [default: (currently set configuration)]')
@click.option('--max-idle',
              help="If specified, maximum idle time before shutdown (e.g. 60m).")
@click.option('--expiration-time',
              help="If specified, time at which cluster is shutdown (e.g. 2020-01-01T00:00:00Z).")
@click.option('--max-age',
              help='If specified, maximum age before shutdown (e.g. 60m).')
@click.option('--bucket', type=str,
              help="The Google Cloud Storage bucket to use for cluster staging (just the bucket name, no gs:// prefix).")
@click.option('--network',
              help="The network for all nodes in this cluster.")
@click.option('--master-tags',
              help="Comma-separated list of instance tags to apply to the mastern node.")
@click.option('--wheel',
              help='Non-default Hail installation. Warning: experimental.')
@click.option('--init',
              default='', show_default=True,
              help="Comma-separated list of init scripts to run.")
@click.option('--init_timeout',
              default='20m', show_default=True,
              help="Flag to specify a timeout period for the initialization action.")
@click.option('--vep',
              type=click.Choice(['GRCh37', 'GRCh38']),
              help='Install VEP for the specified reference genome.')
@click.option('--dry-run', is_flag=True,
              help="Print gcloud dataproc command, but don't run it.")
@click.option('--requester-pays-allow-all', is_flag=True,
              help="Allow reading from all requester-pays buckets.")
@click.option('--requester-pays-allow-buckets',
              help="Comma-separated list of requester-pays buckets to allow reading from.")
@click.option('--requester-pays-allow-annotation-db', is_flag=True,
              help="Allows reading from any of the requester-pays buckets that hold data for the annotation database.")
@click.option('--debug-mode', is_flag=True,
              help="Enable debug features on created cluster (heap dump on out-of-memory error).")
@click.argument('gcloud_args', nargs=-1)
def start(
        cluster_name,
        master_machine_type, master_memory_fraction, master_boot_disk_size,
        num_master_local_ssds, num_secondary_workers, num_worker_local_ssds,
        num_workers, secondary_worker_boot_disk_size,
        worker_boot_disk_size, worker_machine_type,
        region, zone, properties, metadata, packages, project, configuration,
        max_idle, expiration_time, no_max_idle,
        max_age, no_max_age,
        bucket, network, master_tags, wheel,
        init, init_timeout, vep, dry_run,
        requester_pays_allow_all, requester_pays_allow_buckets,
        requester_pays_allow_annotation_db,
        debug_mode, gcloud_args):
    conf = ClusterConfig()
    conf.extend_flag('image-version', IMAGE_VERSION)

    if not pkg_resources.resource_exists('hailtop.hailctl', "deploy.yaml"):
        raise RuntimeError("package has no 'deploy.yaml' file")
    deploy_metadata = yaml.safe_load(
        pkg_resources.resource_stream('hailtop.hailctl', "deploy.yaml"))['dataproc']

    conf.extend_flag('properties', DEFAULT_PROPERTIES)
    if properties:
        conf.parse_and_extend('properties', properties)

    if debug_mode:
        conf.extend_flag('properties', {
            "spark:spark.driver.extraJavaOptions": "-Xss4M -XX:+HeapDumpOnOutOfMemoryError",
            "spark:spark.executor.extraJavaOptions": "-Xss4M -XX:+HeapDumpOnOutOfMemoryError",
        })

    # default to highmem machines if using VEP
    if not worker_machine_type:
        worker_machine_type = 'n1-highmem-8' if vep else 'n1-standard-8'

    # default initialization script to start up cluster with
    conf.extend_flag('initialization-actions',
                     [deploy_metadata['init_notebook.py']])

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

            conf.extend_flag("properties", {"spark:spark.hadoop.fs.gs.requester.pays.buckets": ",".join(requester_pays_bucket_sources)})

        # Need to pick requester pays project.
        requester_pays_project = project if project else gcloud.get_config("project")

        conf.extend_flag("properties", {"spark:spark.hadoop.fs.gs.requester.pays.mode": requester_pays_mode,
                                        "spark:spark.hadoop.fs.gs.requester.pays.project.id": requester_pays_project})

    # gcloud version 277 and onwards requires you to specify a region. Let's just require it for all hailctl users for consistency.
    if region:
        project_region = region
    else:
        project_region = gcloud.get_config("dataproc/region")

    if not project_region:
        raise RuntimeError("Could not determine dataproc region. Use --region argument to hailctl, or use `gcloud config set dataproc/region <my-region>` to set a default.")

    # add VEP init script
    if vep:
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
        conf.extend_flag('initialization-actions', [deploy_metadata[f'vep-{vep}.sh']])
    # add custom init scripts
    if init:
        conf.extend_flag('initialization-actions', init.split(','))

    if metadata:
        conf.parse_and_extend('metadata', metadata)

    wheel = wheel or deploy_metadata['wheel']
    conf.extend_flag('metadata', {'WHEEL': wheel})

    # if Python packages requested, add metadata variable
    packages = deploy_metadata['pip_dependencies'].strip('|').split('|||')
    metadata_pkgs = conf.flags['metadata'].get('PKGS')
    split_regex = r'[|,]'
    if metadata_pkgs:
        packages.extend(re.split(split_regex, metadata_pkgs))
    if packages:
        packages.extend(re.split(split_regex, packages))
    conf.extend_flag('metadata', {'PKGS': '|'.join(set(packages))})

    def disk_size(size):
        if vep:
            size = max(size, 200)
        return str(size) + 'GB'

    conf.extend_flag('properties',
                     {"spark:spark.driver.memory": "{driver_memory}g".format(
                         driver_memory=str(int(MACHINE_MEM[master_machine_type] * master_memory_fraction)))})
    conf.flags['master-machine-type'] = master_machine_type
    conf.flags['master-boot-disk-size'] = '{}GB'.format(master_boot_disk_size)
    conf.flags['num-master-local-ssds'] = num_master_local_ssds
    conf.flags['num-secondary-workers'] = num_secondary_workers
    conf.flags['num-worker-local-ssds'] = num_worker_local_ssds
    conf.flags['num-workers'] = num_workers
    conf.flags['secondary-worker-boot-disk-size'] = disk_size(secondary_worker_boot_disk_size)
    conf.flags['worker-boot-disk-size'] = disk_size(worker_boot_disk_size)
    conf.flags['worker-machine-type'] = worker_machine_type
    if region:
        conf.flags['region'] = region
    if zone:
        conf.flags['zone'] = zone
    conf.flags['initialization-action-timeout'] = init_timeout
    if network:
        conf.flags['network'] = network
    if configuration:
        conf.flags['configuration'] = configuration
    if project:
        conf.flags['project'] = project
    if bucket:
        conf.flags['bucket'] = bucket

    account = gcloud.get_config("account")
    if account:
        conf.flags['labels'] = 'creator=' + re.sub(r'[^0-9a-z_\-]', '_', account.lower())[:63]

    # rewrite metadata and properties to escape them
    conf.flags['metadata'] = '^|||^' + '|||'.join(f'{k}={v}' for k, v in conf.flags['metadata'].items())
    conf.flags['properties'] = '^|||^' + '|||'.join(f'{k}={v}' for k, v in conf.flags['properties'].items())

    # command to start cluster
    cmd = conf.get_command(cluster_name)

    if beta or max_idle or max_age:
        cmd.insert(1, 'beta')
    if max_idle:
        cmd.append('--max-idle={}'.format(max_idle))
    if max_age:
        cmd.append('--max-age={}'.format(max_age))
    if expiration_time:
        cmd.append('--expiration_time={}'.format(expiration_time))

    if gcloud_args:
        cmd.extend(gcloud_args)

    # print underlying gcloud command
    print(' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[5:]))

    # spin up cluster
    if not dry_run:
        print("Starting cluster '{}'...".format(cluster_name))
        gcloud.run(cmd[1:])

    if master_tags:
        add_tags_command = ['compute', 'instances', 'add-tags', cluster_name + '-m', '--tags', master_tags]

        if project:
            add_tags_command.append(f"--project={project}")
        if zone:
            add_tags_command.append(f"--zone={zone}")

        print('gcloud ' + ' '.join(add_tags_command))
        if not dry_run:
            gcloud.run(add_tags_command)
