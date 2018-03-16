from subprocess import call, check_output
import sys


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


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')

    # arguments with default parameters
    parser.add_argument('--hash', default='latest', type=str,
                        help='Hail build to use for notebook initialization (default: %(default)s).')
    parser.add_argument('--spark', default='2.0.2', type=str, choices=['2.0.2', '2.2.0'],
                        help='Spark version used to build Hail (default: %(default)s)')
    parser.add_argument('--version', default='0.1', type=str, choices=['0.1', 'devel'],
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
    parser.add_argument('--metadata', default='',
                        help='Comma-separated list of metadata to add: KEY1=VALUE1,KEY2=VALUE2...')
    parser.add_argument('--packages', '--pkgs', default='',
                        help='Comma-separated list of Python packages to be installed on the master node.')

    # specify custom Hail jar and zip
    parser.add_argument('--jar', help='Hail jar to use for Jupyter notebook.')
    parser.add_argument('--zip', help='Hail zip to use for Jupyter notebook.')

    # initialization action flags
    parser.add_argument('--init', default='', help='Comma-separated list of init scripts to run.')
    parser.add_argument('--vep', action='store_true', help='Configure the cluster to run VEP.')
    parser.add_argument('--dry-run', action='store_true', help="Print gcloud dataproc command, but don't run it.")


def main(args):

    # Google dataproc image version to use
    if args.spark == '2.0.2':
        image_version = '1.1'
    else:
        assert args.spark == '2.2.0'
        image_version = '1.2'

    # default to highmem machines if using VEP
    if not args.worker_machine_type:
        if args.vep:
            args.worker_machine_type = 'n1-highmem-8'
        else:
            args.worker_machine_type = 'n1-standard-8'  # default
    
    # parse Spark and HDFS configuration parameters, combine into properties argument
    properties = [
        'spark:spark.driver.memory={}g'.format(str(int(machine_mem[args.master_machine_type] * args.master_memory_fraction))),
        'spark:spark.driver.maxResultSize=0',
        'spark:spark.task.maxFailures=20',
        'spark:spark.kryoserializer.buffer.max=1g',
        'spark:spark.driver.extraJavaOptions=-Xss4M',
        'spark:spark.executor.extraJavaOptions=-Xss4M',
        'hdfs:dfs.replication=1'
    ]
    if args.properties:
        properties.append(args.properties)

    # default initialization script to start up cluster with
    init_actions = ['gs://dataproc-initialization-actions/conda/bootstrap-conda.sh',
                    init_script]

    if args.version == 'devel' and args.spark != '2.2.0':
        sys.stderr.write("ERROR: Hail version 'devel' requires Spark 2.2.0.")
        sys.exit(1)

    # add VEP init script
    if args.vep:
        init_actions.append('gs://hail-common/vep/vep/vep85-init.sh')

    # add custom init scripts
    if args.init:
        init_actions.append(args.init)


    if (not (args.jar and args.zip)) and (args.jar or args.zip):
        sys.stderr.write('ERROR: pass both --jar and --zip or neither')
        sys.exit(1)

    jar = None
    py_zip = None

    if args.jar:
        jar = args.jar
        py_zip = args.zip
    else:
        if sys.version_info >= (3,0):
            # Python 3 check_output returns a byte string
            decode_f = lambda x: x.decode()
        else:
            # In Python 2, bytes and str are the same
            decode_f = lambda x: x

        if args.hash == 'latest':
            hail_hash = decode_f(check_output(['gsutil', 'cat', 'gs://hail-common/builds/{0}/latest-hash-spark-{1}.txt'.format(args.version, args.spark)]).strip())
        else:
            hail_hash = args.hash

        hail_jar = 'hail-{0}-{1}-Spark-{2}.jar'.format(args.version, hail_hash, args.spark)
        jar = 'gs://hail-common/builds/{0}/jars/{1}'.format(args.version, hail_jar)
        hail_zip = 'hail-{0}-{1}.zip'.format(args.version, hail_hash)
        py_zip = 'gs://hail-common/builds/{0}/python/{1}'.format(args.version, hail_zip)

    # get Hail build (default to latest)

    # prepare metadata values
    metadata = []
    metadata.append('JAR={}'.format(jar))
    metadata.append('ZIP={}'.format(py_zip))
    if args.metadata:
        metadata.append(args.metadata)

    # if Python packages requested, add metadata variable
    if args.packages:
        metadata.append('PKGS={}'.format(args.packages.replace(',', '|')))

    if args.version == 'devel':
        metadata.append('MINICONDA_VERSION=4.4.10')
    else:
        metadata.append('MINICONDA_VARIANT=2')

    # command to start cluster
    cmd = [
        'gcloud', 
        'dataproc', 
        'clusters', 
        'create',
        args.name,
        '--image-version={}'.format(image_version),
        '--master-machine-type={}'.format(args.master_machine_type),
        '--metadata={}'.format(','.join(metadata)),
        '--master-boot-disk-size={}GB'.format(args.master_boot_disk_size),
        '--num-master-local-ssds={}'.format(args.num_master_local_ssds),
        '--num-preemptible-workers={}'.format(args.num_preemptible_workers),
        '--num-worker-local-ssds={}'.format(args.num_worker_local_ssds),
        '--num-workers={}'.format(args.num_workers),
        '--preemptible-worker-boot-disk-size={}GB'.format(args.preemptible_worker_boot_disk_size),
        '--worker-boot-disk-size={}GB'.format(args.worker_boot_disk_size),
        '--worker-machine-type={}'.format(args.worker_machine_type),
        '--zone={}'.format(args.zone),
        '--properties={}'.format(",".join(properties)),
        '--initialization-actions={}'.format(','.join(init_actions))
    ]

    # print underlying gcloud command
    print('gcloud command:')
    print(' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[5:]))

    # spin up cluster
    if not args.dry_run:
        print("Starting cluster '{}'...".format(args.name))
        call(cmd)
