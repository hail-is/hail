import builtins
import json
import re
from typing import Annotated as Ann
from typing import Optional
from urllib.parse import urlparse

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


def _require_s3_uri(uri: str, option_name: str) -> None:
    from hailtop.aiocloud.aioaws.fs import S3AsyncFS  # pylint: disable=import-outside-toplevel

    try:
        parsed = urlparse(uri)
    except ValueError:
        parsed = None
    bucket = parsed.netloc if parsed is not None else ''
    valid_bucket = (
        re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]', bucket) is not None
        and '..' not in bucket
        and '.-' not in bucket
        and '-.' not in bucket
    )
    valid = (
        parsed is not None
        and S3AsyncFS.valid_url(uri)
        and parsed.scheme == 's3'
        and valid_bucket
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
    )
    if not valid:
        raise typer.BadParameter(f'{option_name} must be a valid S3 URI such as s3://my-bucket/hail-tmp/.')


@app.command()
def start(
    cluster_name: str,
    s3_scratch: Ann[
        Optional[str],
        Opt(
            '--s3-scratch',
            help='S3 URI for scratch data (e.g. s3://bucket/hail-tmp/). Defaults to the emr/remote_tmpdir config.',
        ),
    ] = None,
    region: Ann[Optional[str], Opt(help='AWS region for the cluster.')] = None,
    release_label: Ann[str, Opt(help='EMR release label.')] = 'emr-7.3.0',
    master_instance_type: Ann[str, Opt(help='Instance type for the master node.')] = 'm5.xlarge',
    core_instance_type: Ann[str, Opt(help='Instance type for core (worker) nodes.')] = 'm5.xlarge',
    core_instance_count: Ann[int, Opt(help='Number of core (worker) nodes.')] = 2,
    ec2_key_name: Ann[Optional[str], Opt(help='EC2 key pair name for SSH access.')] = None,
    subnet_id: Ann[Optional[str], Opt(help='VPC subnet id to launch the cluster in.')] = None,
    log_uri: Ann[Optional[str], Opt(help='S3 URI for EMR logs. Defaults to <s3-scratch>/logs/.')] = None,
    use_default_roles: Ann[bool, Opt(help='Use EMR_DefaultRole and EMR_EC2_DefaultRole.')] = True,
    service_role: Ann[Optional[str], Opt(help='Custom EMR service role (requires --no-use-default-roles).')] = None,
    instance_profile: Ann[
        Optional[str], Opt(help='Custom EC2 instance profile (requires --no-use-default-roles).')
    ] = None,
    no_off_heap_memory: Ann[
        bool, Opt('--no-off-heap-memory', help="Don't set a per-core cap on Hail off-heap allocations.")
    ] = False,
    off_heap_memory_per_core_mb: Ann[int, Opt(help='Maximum Hail off-heap allocation per task core, in MB.')] = 1024,
    run_job_flow_json: Ann[
        Optional[str],
        Opt(
            '--run-job-flow-json',
            help='JSON object deep-merged into the boto3 run_job_flow request for advanced options.',
        ),
    ] = None,
    vep: Ann[Optional[str], Opt(help='(Phase 2) Install VEP for the given reference genome.')] = None,
    dry_run: Ann[bool, Opt(help="Build the request but don't call AWS.")] = False,
):
    """Start an EMR cluster configured for Hail."""
    from hailtop import __pip_version__  # pylint: disable=import-outside-toplevel
    from hailtop.config import ConfigVariable, configuration_of  # pylint: disable=import-outside-toplevel

    from . import emr  # pylint: disable=import-outside-toplevel
    from . import start as start_mod  # pylint: disable=import-outside-toplevel

    if vep is not None:
        raise NotImplementedError('VEP on EMR is planned for a future release (Phase 2).')

    scratch = configuration_of(ConfigVariable.EMR_REMOTE_TMPDIR, s3_scratch, None)
    if scratch is None:
        raise typer.BadParameter('Provide --s3-scratch or set `hailctl config set emr/remote_tmpdir`.')
    _require_s3_uri(scratch, '--s3-scratch')

    if use_default_roles and (service_role is not None or instance_profile is not None):
        raise typer.BadParameter(
            'Default roles are enabled; pass --no-use-default-roles with '
            '--service-role and --instance-profile to use custom roles.'
        )

    resolved_region = emr.resolve_region(region)
    bootstrap_s3_uri = f'{scratch.rstrip("/")}/bootstrap/{cluster_name}/install-hail-emr.sh'
    resolved_log_uri = log_uri or (scratch.rstrip('/') + '/logs/')
    _require_s3_uri(resolved_log_uri, '--log-uri')

    kwargs = start_mod.build_run_job_flow_kwargs(
        cluster_name=cluster_name,
        release_label=release_label,
        master_instance_type=master_instance_type,
        core_instance_type=core_instance_type,
        core_instance_count=core_instance_count,
        ec2_key_name=ec2_key_name,
        subnet_id=subnet_id,
        log_uri=resolved_log_uri,
        bootstrap_s3_uri=bootstrap_s3_uri,
        pip_version=__pip_version__,
        off_heap_memory_per_core_mb=None if no_off_heap_memory else off_heap_memory_per_core_mb,
        use_default_roles=use_default_roles,
        service_role=service_role,
        instance_profile=instance_profile,
    )

    if run_job_flow_json is not None:
        try:
            overlay = json.loads(run_job_flow_json)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f'--run-job-flow-json is not valid JSON: {exc}') from exc
        if not isinstance(overlay, dict):
            raise typer.BadParameter('--run-job-flow-json must contain a JSON object.')
        kwargs = start_mod.merge_run_job_flow_overlay(kwargs, overlay)

    final_release_label = kwargs.get('ReleaseLabel')
    if not isinstance(final_release_label, str):
        raise typer.BadParameter('The final RunJobFlow ReleaseLabel must be a string.')
    start_mod.check_release_spark_compatibility(final_release_label)

    applications = kwargs.get('Applications')
    if not isinstance(applications, builtins.list) or not any(
        isinstance(application, dict)
        and isinstance(application.get('Name'), str)
        and application['Name'].lower() == 'spark'
        for application in applications
    ):
        raise typer.BadParameter('The final RunJobFlow Applications list must include Spark.')

    instances = kwargs.get('Instances')
    if not isinstance(instances, dict):
        raise typer.BadParameter('The final RunJobFlow Instances value must be a JSON object.')
    if 'InstanceGroups' in instances and 'InstanceFleets' in instances:
        raise typer.BadParameter('RunJobFlow Instances cannot contain both InstanceGroups and InstanceFleets.')

    final_log_uri = kwargs.get('LogUri')
    if not isinstance(final_log_uri, str):
        raise typer.BadParameter('The final RunJobFlow LogUri must be a string.')
    _require_s3_uri(final_log_uri, 'The final RunJobFlow LogUri')

    if dry_run:
        print(json.dumps(kwargs, indent=2))
        return

    final_uses_default_service_role = kwargs.get('ServiceRole') == emr.DEFAULT_SERVICE_ROLE
    final_uses_default_job_flow_role = kwargs.get('JobFlowRole') == emr.DEFAULT_JOB_FLOW_ROLE
    if final_uses_default_service_role or final_uses_default_job_flow_role:
        emr.check_default_roles(
            check_service_role=final_uses_default_service_role,
            check_job_flow_role=final_uses_default_job_flow_role,
        )

    _upload_bootstrap(bootstrap_s3_uri)
    resp = emr.emr_client(resolved_region).run_job_flow(**kwargs)
    print(f"Started cluster {resp['JobFlowId']}.")


def _upload_bootstrap(bootstrap_s3_uri: str) -> None:
    import importlib.resources as ir  # pylint: disable=import-outside-toplevel

    from . import emr  # pylint: disable=import-outside-toplevel

    script_bytes = ir.files('hailtop.hailctl.emr').joinpath('resources/install-hail-emr.sh').read_bytes()
    emr.upload_to_s3(bootstrap_s3_uri, script_bytes)


@app.command()
def stop(
    cluster_id: Ann[str, Arg(help='The EMR cluster (job flow) id, e.g. j-XXXX.')],
    region: Ann[Optional[str], Opt(help='AWS region.')] = None,
):
    """Terminate an EMR cluster."""
    from . import emr  # pylint: disable=import-outside-toplevel

    resolved_region = emr.resolve_region(region)
    print(f'Terminating cluster {cluster_id} ...')
    emr.emr_client(resolved_region).terminate_job_flows(JobFlowIds=[cluster_id])


@app.command()
def list(
    region: Ann[Optional[str], Opt(help='AWS region.')] = None,
    all_states: Ann[bool, Opt('--all', help='Include terminated clusters.')] = False,
):
    """List active EMR clusters."""
    from . import emr  # pylint: disable=import-outside-toplevel

    resolved_region = emr.resolve_region(region)
    kwargs = {} if all_states else {'ClusterStates': ACTIVE_CLUSTER_STATES}
    paginator = emr.emr_client(resolved_region).get_paginator('list_clusters')
    for page in paginator.paginate(**kwargs):
        for cluster in page.get('Clusters', []):
            print(f"{cluster['Id']}\t{cluster['Status']['State']}\t{cluster['Name']}")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def submit(
    ctx: typer.Context,
    cluster_id: Ann[str, Arg(help='The EMR cluster (job flow) id, e.g. j-XXXX.')],
    script: Ann[str, Arg(help='Path to the local Python script.')],
    s3_scratch: Ann[
        Optional[str],
        Opt('--s3-scratch', help='S3 URI for scratch data. Defaults to the emr/remote_tmpdir config.'),
    ] = None,
    region: Ann[Optional[str], Opt(help='AWS region.')] = None,
    no_wait: Ann[
        bool, Opt('--no-wait', help='Return immediately after submitting, without waiting for completion.')
    ] = False,
):
    """Submit a Python job to an EMR cluster configured for Hail."""
    from hailtop.config import ConfigVariable, configuration_of  # pylint: disable=import-outside-toplevel

    from . import submit as submit_mod  # pylint: disable=import-outside-toplevel

    scratch = configuration_of(ConfigVariable.EMR_REMOTE_TMPDIR, s3_scratch, None)
    if scratch is None:
        raise typer.BadParameter('Provide --s3-scratch or set `hailctl config set emr/remote_tmpdir`.')
    _require_s3_uri(scratch, '--s3-scratch')

    submit_mod.submit(cluster_id, script, scratch, region, ctx.args, wait=not no_wait)
