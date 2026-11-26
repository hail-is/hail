from typing import Annotated as Ann
from typing import Optional

import typer
from typer import Argument as Arg
from typer import Option as Opt

app = typer.Typer(
    name='emr',
    no_args_is_help=True,
    help='Manage and monitor Hail Amazon EMR clusters.',
    pretty_exceptions_show_locals=False,
)

# Clusters that still exist and may still cost money. TERMINATING is included so
# a cluster does not vanish from `list` before it has actually shut down.
ACTIVE_CLUSTER_STATES = ['STARTING', 'BOOTSTRAPPING', 'RUNNING', 'WAITING', 'TERMINATING']


@app.command()
def start(
    cluster_name: str,
    wheel: Ann[
        str, Opt(help='S3 URI of a Hail wheel built for Spark 4.1 (s3://bucket/hail-VERSION-py3-none-any.whl).')
    ],
    region: Ann[Optional[str], Opt(help='AWS region. Defaults to the AWS CLI region (`aws configure`).')] = None,
    num_workers: Ann[int, Opt('--num-workers', '--n-workers', '-w', help='Number of worker machines.')] = 2,
    master_instance_type: Ann[str, Opt(help='Instance type of the primary node.')] = 'm5.xlarge',
    worker_instance_type: Ann[str, Opt(help='Instance type of the worker nodes.')] = 'm5.xlarge',
    subnet_id: Ann[Optional[str], Opt(help='VPC subnet for the cluster. Defaults to the default VPC subnet.')] = None,
    service_access_security_group: Ann[
        Optional[str], Opt(help='Service access security group. Required for private subnets.')
    ] = None,
    log_uri: Ann[Optional[str], Opt(help='S3 URI for EMR logs.')] = None,
    idle_timeout: Ann[int, Opt(help='Terminate the cluster after this many idle seconds (60-604800).')] = 3600,
    dry_run: Ann[bool, Opt(help="Print the EMR request but don't call AWS.")] = False,
):
    """Start an EMR cluster configured for Hail."""
    from . import start as start_mod  # pylint: disable=import-outside-toplevel

    start_mod.start(
        cluster_name,
        wheel,
        region,
        subnet_id,
        service_access_security_group,
        num_workers,
        master_instance_type,
        worker_instance_type,
        log_uri,
        idle_timeout,
        dry_run,
    )


@app.command()
def stop(
    cluster_id: Ann[str, Arg(help='The EMR cluster id, e.g. j-XXXX.')],
    region: Ann[Optional[str], Opt(help='AWS region.')] = None,
):
    """Terminate an EMR cluster."""
    from . import emr  # pylint: disable=import-outside-toplevel

    print(f'Terminating cluster {cluster_id} ...')
    emr.emr_client(region).terminate_job_flows(JobFlowIds=[cluster_id])


@app.command()
def list(
    region: Ann[Optional[str], Opt(help='AWS region.')] = None,
    all_states: Ann[bool, Opt('--all', help='Include terminated clusters.')] = False,
):
    """List active EMR clusters."""
    from . import emr  # pylint: disable=import-outside-toplevel

    kwargs = {} if all_states else {'ClusterStates': ACTIVE_CLUSTER_STATES}
    paginator = emr.emr_client(region).get_paginator('list_clusters')
    for page in paginator.paginate(**kwargs):
        for cluster in page.get('Clusters', []):
            print(f"{cluster['Id']}\t{cluster['Status']['State']}\t{cluster['Name']}")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def submit(
    ctx: typer.Context,
    cluster_id: Ann[str, Arg(help='The EMR cluster id, e.g. j-XXXX.')],
    script: Ann[str, Arg(help='Python script: an s3:// URI, or a local file together with --upload-to.')],
    upload_to: Ann[Optional[str], Opt(help='S3 URI prefix to upload a local script to.')] = None,
    region: Ann[Optional[str], Opt(help='AWS region.')] = None,
    no_wait: Ann[bool, Opt('--no-wait', help='Return immediately after submitting.')] = False,
):
    """Submit a Python job to an EMR cluster configured for Hail."""
    from . import submit as submit_mod  # pylint: disable=import-outside-toplevel

    submit_mod.submit(cluster_id, script, upload_to, region, ctx.args, wait=not no_wait)
