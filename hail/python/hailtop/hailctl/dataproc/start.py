import re
import subprocess as sp
import sys

import pkg_resources
import yaml

from .cluster_config import ClusterConfig

DEFAULT_PROPERTIES = {
    "spark:spark.driver.maxResultSize": "0",
    "spark:spark.task.maxFailures": "20",
    "spark:spark.kryoserializer.buffer.max": "1g",
    "spark:spark.driver.extraJavaOptions": "-Xss4M",
    "spark:spark.executor.extraJavaOptions": "-Xss4M",
    "hdfs:dfs.replication": "1",
    'dataproc:dataproc.logging.stackdriver.enable': 'false',
    'dataproc:dataproc.monitoring.stackdriver.enable': 'false'
}

# master machine type to memory map, used for setting spark.driver.memory property
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
    'n1-highcpu-64': 57.6
}

SPARK_VERSION = '2.4.0'
IMAGE_VERSION = '1.4-debian9'


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
    parser.add_argument('--num-preemptible-workers', '--n-pre-workers', '-p', default=0, type=int,
                        help='Number of preemptible worker machines (default: %(default)s).')
    parser.add_argument('--num-worker-local-ssds', default=0, type=int,
                        help='Number of local SSDs to attach to each worker machine (default: %(default)s).')
    parser.add_argument('--num-workers', '--n-workers', '-w', default=2, type=int,
                        help='Number of worker machines (default: %(default)s).')
    parser.add_argument('--preemptible-worker-boot-disk-size', default=40, type=int,
                        help='Disk size of preemptible machines, in GB (default: %(default)s).')
    parser.add_argument('--worker-boot-disk-size', default=40, type=int,
                        help='Disk size of worker machines, in GB (default: %(default)s).')
    parser.add_argument('--worker-machine-type', '--worker',
                        help='Worker machine type (default: n1-standard-8, or n1-highmem-8 with --vep).')
    parser.add_argument('--zone', default='us-central1-b',
                        help='Compute zone for the cluster (default: %(default)s).')
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
    parser.add_argument('--max-age', type=str, help='If specified, maximum age before shutdown (e.g. 60m).')
    parser.add_argument('--bucket', type=str,
                        help='The Google Cloud Storage bucket to use for cluster staging (just the bucket name, no gs:// prefix).')

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


def main(args, pass_through_args):
    conf = ClusterConfig()
    conf.extend_flag('image-version', IMAGE_VERSION)

    if not pkg_resources.resource_exists('hailtop.hailctl', "deploy.yaml"):
        raise RuntimeError(f"package has no 'deploy.yaml' file")
    deploy_metadata = yaml.safe_load(
        pkg_resources.resource_stream('hailtop.hailctl', "deploy.yaml"))['dataproc']

    conf.extend_flag('properties', DEFAULT_PROPERTIES)
    if args.properties:
        conf.parse_and_extend('properties', args.properties)

    # default to highmem machines if using VEP
    if not args.worker_machine_type:
        args.worker_machine_type = 'n1-highmem-8' if args.vep else 'n1-standard-8'

    # default initialization script to start up cluster with
    conf.extend_flag('initialization-actions',
                     [deploy_metadata['init_notebook.py']])

    # add VEP init script
    if args.vep:
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
    conf.extend_flag('metadata', {'PKGS': '|'.join(packages)})

    def disk_size(size):
        if args.vep:
            size = max(size, 200)
        return str(size) + 'GB'

    # rewrite metadata to escape it
    conf.flags['metadata'] = '^|||^' + '|||'.join(f'{k}={v}' for k, v in conf.flags['metadata'].items())

    conf.extend_flag('properties',
                     {"spark:spark.driver.memory": "{driver_memory}g".format(
                         driver_memory=str(int(MACHINE_MEM[args.master_machine_type] * args.master_memory_fraction)))})
    conf.flags['master-machine-type'] = args.master_machine_type
    conf.flags['master-boot-disk-size'] = '{}GB'.format(args.master_boot_disk_size)
    conf.flags['num-master-local-ssds'] = args.num_master_local_ssds
    conf.flags['num-preemptible-workers'] = args.num_preemptible_workers
    conf.flags['num-worker-local-ssds'] = args.num_worker_local_ssds
    conf.flags['num-workers'] = args.num_workers
    conf.flags['preemptible-worker-boot-disk-size'] = disk_size(args.preemptible_worker_boot_disk_size)
    conf.flags['worker-boot-disk-size'] = disk_size(args.worker_boot_disk_size)
    conf.flags['worker-machine-type'] = args.worker_machine_type
    conf.flags['zone'] = args.zone
    conf.flags['initialization-action-timeout'] = args.init_timeout
    if args.configuration:
        conf.flags['configuration'] = args.configuration
    if args.project:
        conf.flags['project'] = args.project
    if args.bucket:
        conf.flags['bucket'] = args.bucket

    try:
        label = sp.check_output(['gcloud', 'config', 'get-value', 'account'])
        conf.flags['labels'] = 'creator=' + re.sub(r'[^0-9a-z_\-]', '_', label.decode().strip().lower())[:63]
    except sp.CalledProcessError as e:
        sys.stderr.write("Warning: could not run 'gcloud config get-value account': " + e.output.decode() + "\n")

    # command to start cluster
    cmd = conf.get_command(args.name)

    if args.beta or args.max_idle or args.max_age:
        cmd.insert(1, 'beta')
    if args.max_idle:
        cmd.append('--max-idle={}'.format(args.max_idle))
    if args.max_age:
        cmd.append('--max-age={}'.format(args.max_age))

    cmd.extend(pass_through_args)

    # print underlying gcloud command
    print(' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[5:]))

    # spin up cluster
    if not args.dry_run:
        print("Starting cluster '{}'...".format(args.name))
        sp.check_call(cmd)
