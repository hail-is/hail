import sys

import typer
from typer import Option as Opt, Argument as Arg

from typing import List, Optional
from typing_extensions import Annotated as Ann

from .connect import connect as dataproc_connect, DataprocConnectService
from .submit import submit as dataproc_submit
from .diagnose import diagnose as dataproc_diagnose
from .modify import modify as dataproc_modify
from .start import start as dataproc_start, VepVersion
from ..describe import describe
from . import gcloud


MINIMUM_REQUIRED_GCLOUD_VERSION = (285, 0, 0)


BetaOption = Ann[bool, Opt(help='Force use of `beta` in gcloud commands')]
use_gcloud_beta = False

ProjectOption = Ann[
    Optional[str], Opt(help='Google Cloud project for the cluster (defaults to currently set project).')
]

ZoneOption = Ann[
    Optional[str],
    Opt('--zone', '-z', help='Compute zone for Dataproc cluster.'),
]

DryRunOption = Ann[
    bool,
    Opt(help="Print gcloud dataproc command, but don't run it."),
]

NumWorkersOption = Ann[Optional[int], Opt('--num-workers', '--n-workers', '-w', help='Number of worker machines.')]

NumSecondaryWorkersOption = Ann[
    Optional[int],
    Opt(
        '--num-secondary-workers',
        '--num-preemptible-workers',
        '--n-pre-workers',
        '-p',
        help='Number of secondary (preemptible) worker machines.',
    ),
]


app = typer.Typer(
    name='dataproc',
    no_args_is_help=True,
    help='Manage Hail Dataproc clusters.',
    pretty_exceptions_show_locals=False,
)


@app.callback()
def check_gcloud_version(beta: BetaOption = False):
    global use_gcloud_beta
    use_gcloud_beta = beta

    try:
        gcloud_version = gcloud.get_version()
        if gcloud_version < MINIMUM_REQUIRED_GCLOUD_VERSION:
            sys.exit(
                f"hailctl dataproc requires Google Cloud SDK (gcloud) version {'.'.join(map(str, MINIMUM_REQUIRED_GCLOUD_VERSION))} or higher",
            )
    except Exception:
        # If gcloud's output format changes in the future and the version can't be parsed,
        # then continue and attempt to run gcloud.
        print("Warning: unable to determine Google Cloud SDK version", file=sys.stderr)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def start(
    ctx: typer.Context,
    name: str,
    # arguments with default parameters
    master_machine_type: Ann[str, Opt('--master-machine-type', '--master', '-m')] = 'n1-highmem-8',
    master_memory_fraction: Ann[
        float,
        Opt(
            help='Fraction of master memory allocated to the JVM. Use a smaller value to reserve more memory for Python.'
        ),
    ] = 0.8,
    master_boot_disk_size: Ann[int, Opt(help='Disk size of master machine, in GB')] = 100,
    num_master_local_ssds: Ann[int, Opt(help='Number of local SSDs to attach to the master machine.')] = 0,
    num_secondary_workers: NumSecondaryWorkersOption = 0,
    num_worker_local_ssds: Ann[int, Opt(help='Number of local SSDs to attach to each worker machine.')] = 0,
    num_workers: NumWorkersOption = 2,
    secondary_worker_boot_disk_size: Ann[
        int,
        Opt(
            '--secondary-worker-boot-disk-size',
            '--preemptible-worker-boot-disk-size',
            help='Disk size of secondary (preemptible) worker machines, in GB.',
        ),
    ] = 40,
    worker_boot_disk_size: Ann[int, Opt(help='Disk size of worker machines, in GB.')] = 40,
    worker_machine_type: Ann[
        Optional[str],
        Opt(
            '--worker-machine-type',
            '--worker',
            help='Worker machine type (default: n1-standard-8, or n1-highmem-8 with --vep).',
        ),
    ] = None,
    region: Ann[Optional[str], Opt(help='Compute region for the cluster.')] = None,
    zone: ZoneOption = None,
    properties: Ann[Optional[str], Opt(help='Additional configuration properties for the cluster.')] = None,
    metadata: Ann[Optional[str], Opt(help='Comma-separated list of metadata to add: KEY1=VALUE1,KEY2=VALUE2')] = None,
    packages: Ann[
        Optional[str], Opt(help='Comma-separated list of Python packages to be installed on the master node.')
    ] = None,
    project: Ann[Optional[str], Opt(help='GCP project to start cluster (defaults to currently set project).')] = None,
    configuration: Ann[
        Optional[str],
        Opt(help='Google Cloud configuration to start cluster (defaults to currently set configuration).'),
    ] = None,
    max_idle: Ann[Optional[str], Opt(help='If specified, maximum idle time before shutdown (e.g. 60m).')] = None,
    expiration_time: Ann[
        Optional[str], Opt(help='If specified, time at which cluster is shutdown (e.g. 2020-01-01T00:00:00Z).')
    ] = None,
    max_age: Ann[Optional[str], Opt(help='If specified, maximum age before shutdown (e.g. 60m).')] = None,
    bucket: Ann[
        Optional[str],
        Opt(
            help='The Google Cloud Storage bucket to use for cluster temporary storage (just the bucket name, no gs:// prefix).'
        ),
    ] = None,
    temp_bucket: Ann[
        Optional[str],
        Opt(
            help='The Google Cloud Storage bucket to use for cluster temporary storage (just the bucket name, no gs:// prefix).'
        ),
    ] = None,
    network: Ann[Optional[str], Opt(help='The network for all nodes in this cluster.')] = None,
    subnet: Ann[Optional[str], Opt(help='The subnet for all nodes in this cluster.')] = None,
    service_account: Ann[
        Optional[str],
        Opt(
            help='The Google Service Account to use for cluster creation (default to the Compute Engine service account).'
        ),
    ] = None,
    master_tags: Ann[
        Optional[str], Opt(help='Comma-separated list of instance tags to apply to the master node')
    ] = None,
    scopes: Ann[Optional[str], Opt(help='Specifies access scopes for the node instances')] = None,
    wheel: Ann[Optional[str], Opt(help='Non-default Hail installation. Warning: experimental.')] = None,
    # initialization action flags
    init: Ann[str, Opt(help='Comma-separated list of init scripts to run.')] = '',
    init_timeout: Ann[
        str, Opt('--init_timeout', help='Flag to specify a timeout period for the initialization action')
    ] = '20m',
    vep: Ann[Optional[VepVersion], Opt(help='Install VEP for the specified reference genome.')] = None,
    dry_run: DryRunOption = False,
    no_off_heap_memory: Ann[
        bool, Opt('--no-off-heap-memory', help="Don't partition JVM memory between hail heap and JVM heap")
    ] = False,
    big_executors: Ann[
        bool,
        Opt(
            help="Double memory allocated per executor, using half the cores of the cluster with an extra large memory allotment per core."
        ),
    ] = False,
    off_heap_memory_fraction: Ann[
        float, Opt(help='Minimum fraction of worker memory dedicated to off-heap Hail values.')
    ] = 0.6,
    off_heap_memory_hard_limit: Ann[bool, Opt(help='Limit off-heap allocations to the dedicated fraction')] = False,
    yarn_memory_fraction: Ann[
        float, Opt(help='Fraction of machine memory to allocate to the yarn container scheduler.')
    ] = 0.95,
    # requester pays
    requester_pays_allow_all: Ann[bool, Opt(help='Allow reading from all requester-pays buckets.')] = False,
    requester_pays_allow_buckets: Ann[
        Optional[str], Opt(help='Comma-separated list of requester-pays buckets to allow reading from.')
    ] = None,
    requester_pays_allow_annotation_db: Ann[
        bool,
        Opt(help='Allows reading from any of the requester-pays buckets that hold data for the annotation database.'),
    ] = False,
    debug_mode: Ann[
        bool, Opt(help='Enable debug features on created cluster (heap dump on out-of-memory error)')
    ] = False,
):
    '''
    Start a Dataproc cluster configured for Hail.
    '''
    assert num_secondary_workers is not None
    assert num_workers is not None

    dataproc_start(
        name,
        ctx.args,
        master_machine_type,
        master_memory_fraction,
        master_boot_disk_size,
        num_master_local_ssds,
        num_secondary_workers,
        num_worker_local_ssds,
        num_workers,
        secondary_worker_boot_disk_size,
        worker_boot_disk_size,
        worker_machine_type,
        region,
        zone,
        properties,
        metadata,
        packages,
        project,
        configuration,
        max_idle,
        expiration_time,
        max_age,
        bucket,
        temp_bucket,
        network,
        subnet,
        service_account,
        master_tags,
        scopes,
        wheel,
        init,
        init_timeout,
        vep,
        dry_run,
        no_off_heap_memory,
        big_executors,
        off_heap_memory_fraction,
        off_heap_memory_hard_limit,
        yarn_memory_fraction,
        requester_pays_allow_all,
        requester_pays_allow_buckets,
        requester_pays_allow_annotation_db,
        debug_mode,
        use_gcloud_beta,
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def stop(
    ctx: typer.Context,
    name: str,
    asink: Ann[bool, Opt('--async/--sync', help='Do not wait for cluster deletion')] = False,
    dry_run: DryRunOption = False,
):
    '''
    Shut down a Dataproc cluster.
    '''
    print("Stopping cluster '{}'...".format(name))

    cmd = ['dataproc', 'clusters', 'delete', '--quiet', name]
    if asink:
        cmd.append('--async')

    cmd.extend(ctx.args)

    # print underlying gcloud command
    print('gcloud ' + ' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[6:]))

    if not dry_run:
        gcloud.run(cmd)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def list(
    ctx: typer.Context,
):
    '''
    List active Dataproc clusters.
    '''
    gcloud.run(['dataproc', 'clusters', 'list', *ctx.args])


@app.command()
def connect(
    name: str,
    service: DataprocConnectService,
    pass_through_args: Ann[Optional[List[str]], Arg()] = None,
    project: ProjectOption = None,
    port: Ann[str, Opt(help='Local port to use for SSH tunnel to leader (master) node')] = '10000',
    zone: ZoneOption = None,
    dry_run: DryRunOption = False,
):
    '''
    Connect to a running Dataproc cluster with name NAME and start
    the web service SERVICE.
    '''
    dataproc_connect(name, service, project, port, zone, dry_run, pass_through_args or [])


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def submit(
    ctx: typer.Context,
    name: str,
    script: str,
    files: Ann[
        str, Opt(help='Comma-separated list of files to add to the working directory of the Hail application.')
    ] = '',
    pyfiles: Ann[
        str, Opt(help='Comma-separated list of files (or directories with python files) to add to the PYTHONPATH.')
    ] = '',
    properties: Ann[Optional[str], Opt('--properties', '-p', help='Extra Spark properties to set.')] = None,
    gcloud_configuration: Ann[
        Optional[str],
        Opt(
            '--gcloud_configuration',
            help='Google Cloud configuration to submit job (defaults to currently set configuration).',
        ),
    ] = None,
    dry_run: DryRunOption = False,
    region: Ann[Optional[str], Opt(help='Compute region for the cluster.')] = None,
):
    '''
    Submit the Python script at path SCRIPT to a running Dataproc cluster with
    name NAME. To pass arguments to the script being submitted, just list them
    after the name of the script.
    '''
    dataproc_submit(name, script, files, pyfiles, properties, gcloud_configuration, dry_run, region, ctx.args)


@app.command()
def diagnose(
    name: str,
    dest: Ann[str, Opt('--dest', '-d', help="Directory for diagnose output -- must be local.")],
    hail_log: Ann[str, Opt('--hail-log', '-l', help='Path for hail.log file')] = '/home/hail/hail.log',
    overwrite: Ann[bool, Opt(help='Delete dest directory before adding new files')] = False,
    no_diagnose: Ann[bool, Opt('--no-diagnose', help='Do not run gcloud dataproc clusters diagnose.')] = False,
    compress: Ann[bool, Opt('--compress', '-z', help='GZIP all files')] = False,
    workers: Ann[Optional[List[str]], Opt(help='Specific workers to get log files from.')] = None,
    take: Ann[Optional[int], Opt(help='Only download logs from the first N workers.')] = None,
):
    '''
    Diagnose problems in a Dataproc cluster with name NAME.
    '''
    dataproc_diagnose(name, dest, hail_log, overwrite, no_diagnose, compress, workers or [], take)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def modify(
    ctx: typer.Context,
    name: str,
    num_workers: NumWorkersOption = None,
    num_secondary_workers: NumSecondaryWorkersOption = None,
    graceful_decommission_timeout: Ann[
        Optional[str],
        Opt(
            '--graceful-decommision-timeout',
            '--graceful',
            help='If set, cluster size downgrade will use graceful decommissioning with the given timeout (e.g. "60m").',
        ),
    ] = None,
    max_idle: Ann[
        Optional[str],
        Opt(help='New maximum idle time before shutdown (e.g. "60m").'),
    ] = None,
    no_max_idle: Ann[bool, Opt('--no-max-idle', help='Disable auto deletion after idle time.')] = False,
    expiration_time: Ann[
        Optional[str],
        Opt(
            help=(
                'The time when cluster will be auto-deleted. (e.g. "2020-01-01T20:00:00Z"). '
                'Execute gcloud topic datatimes for more information.'
            )
        ),
    ] = None,
    max_age: Ann[
        Optional[str],
        Opt(
            help=(
                'If the cluster is older than this, it will be auto-deleted. (e.g. "2h")'
                'Execute gcloud topic datatimes for more information.'
            )
        ),
    ] = None,
    no_max_age: Ann[bool, Opt('--no-max-age', help='Disable auto-deletion due to max age or expiration time.')] = False,
    dry_run: DryRunOption = False,
    zone: ZoneOption = None,
    update_hail_version: Ann[
        bool,
        Opt(help="Update the version of hail running on cluster to match the currently installed version."),
    ] = False,
    wheel: Ann[Optional[str], Opt(help='New Hail installation.')] = None,
):
    '''
    Modify an active dataproc cluster with name NAME.
    '''
    dataproc_modify(
        name,
        num_workers,
        num_secondary_workers,
        graceful_decommission_timeout,
        max_idle,
        no_max_idle,
        expiration_time,
        max_age,
        no_max_age,
        dry_run,
        zone,
        update_hail_version,
        wheel,
        use_gcloud_beta,
        ctx.args,
    )


app.command(help='DEPRECATED. Describe Hail Matrix Table and Table files.')(describe)
