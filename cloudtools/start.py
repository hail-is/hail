from subprocess import call, check_call, check_output
import sys
import json


COMPATIBILITY_VERSION = 1
init_script = 'gs://hail-common/cloudtools/init_notebook{}.py'.format(COMPATIBILITY_VERSION)

# master machine type to memory map, used for setting spark.driver.memory property
machine_mem = {
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


class ClusterConfig:
    def __init__(self, json_str):
        params = json.loads(json_str)
        self.vars = params['vars']
        self.flags = params['flags']

    def extend_flag(self, flag, values):
        if flag not in self.flags:
            self.flags[flag] = values
        elif isinstance(self.flags[flag], list):
            assert isinstance(values, list)
            self.flags[flag].extend(values)
        else:
            assert isinstance(self.flags[flag], dict)
            assert isinstance(values, dict)
            self.flags[flag].update(values)

    def parse_and_extend(self, flag, values):
        values = dict(tuple(pair.split('=')) for pair in values.split(',') if '=' in pair)
        self.extend_flag(flag, values)

    def format(self, obj):
        if isinstance(obj, dict):
            return self.format(['{}={}'.format(k, v) for k, v in obj.items()])
        if isinstance(obj, list):
            return self.format(','.join(obj))
        else:
            return str(obj).format(**self.vars)

    def get_command(self, name):
        flags = ['--{}={}'.format(f, self.format(v)) for f, v in self.flags.items()]
        return ['gcloud',
                'dataproc',
                'clusters',
                'create',
                name] + flags


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')

    # arguments with default parameters
    parser.add_argument('--hash', default='latest', type=str,
                        help='Hail build to use for notebook initialization (default: %(default)s).')
    parser.add_argument('--spark', type=str,
                        help='Spark version used to build Hail (default: 2.2.0 for 0.2 and 2.0.2 for 0.1)')
    parser.add_argument('--version', default='devel', type=str, choices=['0.1', 'devel'],
                        help='Hail version to use (default: %(default)s).')
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
    parser.add_argument('--max-idle', type=str, help='If specified, maximum idle time before shutdown (e.g. 60m).')
    parser.add_argument('--bucket', type=str, help='The Google Cloud Storage bucket to use for cluster staging (just the bucket name, no gs:// prefix).')

    # specify custom Hail jar and zip
    parser.add_argument('--jar', help='Hail jar to use for Jupyter notebook.')
    parser.add_argument('--zip', help='Hail zip to use for Jupyter notebook.')

    # initialization action flags
    parser.add_argument('--init', default='', help='Comma-separated list of init scripts to run.')
    parser.add_argument('--init_timeout', default='20m', help='Flag to specify a timeout period for the initialization action')
    parser.add_argument('--vep', action='store_true', help='Configure the cluster to run VEP.')
    parser.add_argument('--dry-run', action='store_true', help="Print gcloud dataproc command, but don't run it.")

    # custom config file
    parser.add_argument('--config-file', help='Pass in a custom json file to load configurations.')


def main(args):
    if not args.spark:
        args.spark = '2.2.0' if args.version == 'devel' else '2.0.2'

    if args.hash == 'latest':
        hash_file = 'gs://hail-common/builds/{}/latest-hash-spark-{}.txt'.format(args.version, args.spark)
        hash = check_output(['gsutil', 'cat', hash_file]).strip()
        # Python 3 check_output returns a byte string that needs decoding
        hash = hash.decode() if sys.version_info >= (3, 0) else hash
    else:
        hash = args.hash

    if not args.config_file:
        args.config_file = 'gs://hail-common/builds/{version}/config/hail-config-{version}-{hash}.json'.format(version=args.version, hash=hash)
        exists = call(['gsutil', '-q', 'stat', args.config_file])
        if exists != 0:
            args.config_file = 'gs://hail-common/builds/{version}/config/hail-config-{version}-default.json'.format(version=args.version)
    if args.config_file.startswith('gs://'):
        conf = ClusterConfig(check_output(['gsutil', 'cat', args.config_file]).strip())
    else:
        conf = ClusterConfig(check_output(['cat', args.config_file]).strip())

    if args.spark not in conf.vars['supported_spark'].keys():
        sys.stderr.write("ERROR: Hail version '{}' requires one of Spark {}."
                         .format(args.version, ','.join(conf.vars['supported_spark'].keys())))
        sys.exit(1)
    conf.vars['spark'] = args.spark
    conf.vars['image'] = conf.vars['supported_spark'][args.spark]
    conf.vars['hash'] = hash

    # parse Spark and HDFS configuration parameters, combine into properties argument
    if args.properties:
        conf.parse_and_extend('properties', args.properties)

    # default to highmem machines if using VEP
    if not args.worker_machine_type:
        args.worker_machine_type = 'n1-highmem-8' if args.vep else 'n1-standard-8'

    # default initialization script to start up cluster with
    conf.extend_flag('initialization-actions',
                     ['gs://dataproc-initialization-actions/conda/bootstrap-conda.sh',
                      init_script])
    # add VEP init script
    if args.vep:
        conf.extend_flag('initialization-actions', ['gs://hail-common/vep/vep/vep85-init.sh'])
    # add custom init scripts
    if args.init:
        conf.extend_flag('initialization-actions', args.init)

    if args.jar and args.zip:
        conf.extend_flag('metadata', {'JAR': args.jar, 'ZIP': args.zip})
    elif args.jar or args.zip:
        sys.stderr.write('ERROR: pass both --jar and --zip or neither')
        sys.exit(1)

    if args.metadata:
        conf.parse_and_extend('metadata', args.metadata)
    # if Python packages requested, add metadata variable
    if args.packages:
        packages = '|'.join([conf.flags['metadata']['PKGS']] + args.packages.split(','))
        conf.extend_flag('metadata', {'PKGS': packages})

    conf.vars['driver_memory'] = str(int(machine_mem[args.master_machine_type] * args.master_memory_fraction))
    conf.flags['master-machine-type'] = args.master_machine_type
    conf.flags['master-boot-disk-size'] = '{}GB'.format(args.master_boot_disk_size)
    conf.flags['num-master-local-ssds'] = args.num_master_local_ssds
    conf.flags['num-preemptible-workers'] = args.num_preemptible_workers
    conf.flags['num-worker-local-ssds'] = args.num_worker_local_ssds
    conf.flags['num-workers'] = args.num_workers
    conf.flags['preemptible-worker-boot-disk-size']='{}GB'.format(args.preemptible_worker_boot_disk_size)
    conf.flags['worker-boot-disk-size'] = args.worker_boot_disk_size
    conf.flags['worker-machine-type'] = args.worker_machine_type
    conf.flags['zone'] = args.zone
    conf.flags['initialization-action-timeout'] = args.init_timeout
    if args.bucket:
        conf.flags['bucket'] = args.bucket

    # command to start cluster
    cmd = conf.get_command(args.name)

    if args.max_idle:
        cmd.insert(1, 'beta')
        cmd.append('--max-idle={}'.format(args.max_idle))

    # print underlying gcloud command
    print('gcloud command:')
    print(' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[5:]))

    # spin up cluster
    if not args.dry_run:
        print("Starting cluster '{}'...".format(args.name))
        check_call(cmd)
