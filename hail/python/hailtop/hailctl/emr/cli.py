import json
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


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def start(
    ctx: typer.Context,
    cluster_name: str,
    s3_scratch: Ann[
        Optional[str],
        Opt('--s3-scratch', help='S3 URI for scratch data (e.g. s3://bucket/hail-tmp/). Defaults to the emr/remote_tmpdir config.'),
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
    instance_profile: Ann[Optional[str], Opt(help='Custom EC2 instance profile (requires --no-use-default-roles).')] = None,
    no_off_heap_memory: Ann[bool, Opt('--no-off-heap-memory', help="Don't reserve off-heap memory for Hail values.")] = False,
    off_heap_memory_per_core_mb: Ann[int, Opt(help='Off-heap memory reserved per core, in MB.')] = 1024,
    run_job_flow_json: Ann[
        Optional[str],
        Opt('--run-job-flow-json', help='JSON object deep-merged into the boto3 run_job_flow request for advanced options.'),
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

    start_mod.check_release_spark_compatibility(release_label)

    resolved_region = emr.resolve_region(region)
    bootstrap_s3_uri = f'{scratch.rstrip("/")}/bootstrap/{cluster_name}/install-hail-emr.sh'

    kwargs = start_mod.build_run_job_flow_kwargs(
        cluster_name=cluster_name,
        release_label=release_label,
        master_instance_type=master_instance_type,
        core_instance_type=core_instance_type,
        core_instance_count=core_instance_count,
        ec2_key_name=ec2_key_name,
        subnet_id=subnet_id,
        log_uri=log_uri or (scratch.rstrip('/') + '/logs/'),
        bootstrap_s3_uri=bootstrap_s3_uri,
        pip_version=__pip_version__,
        off_heap_memory_per_core_mb=None if no_off_heap_memory else off_heap_memory_per_core_mb,
        use_default_roles=use_default_roles,
        service_role=service_role,
        instance_profile=instance_profile,
    )

    if run_job_flow_json is not None:
        kwargs = start_mod.deep_merge(kwargs, json.loads(run_job_flow_json))

    if dry_run:
        print(json.dumps(kwargs, indent=2))
        return

    _upload_bootstrap(bootstrap_s3_uri)
    resp = emr.emr_client(resolved_region).run_job_flow(**kwargs)
    print(f"Started cluster {resp['JobFlowId']}.")


def _upload_bootstrap(bootstrap_s3_uri: str) -> None:
    import importlib.resources as ir  # pylint: disable=import-outside-toplevel

    from . import emr  # pylint: disable=import-outside-toplevel

    script_bytes = (
        ir.files('hailtop.hailctl.emr').joinpath('resources/install-hail-emr.sh').read_bytes()
    )
    emr.upload_to_s3(bootstrap_s3_uri, script_bytes)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def stop(
    ctx: typer.Context,
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
):
    """List EMR clusters."""
    from . import emr  # pylint: disable=import-outside-toplevel

    resolved_region = emr.resolve_region(region)
    resp = emr.emr_client(resolved_region).list_clusters()
    for cluster in resp.get('Clusters', []):
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
    no_wait: Ann[bool, Opt('--no-wait', help='Return immediately after submitting, without waiting for completion.')] = False,
):
    """Submit a Python job to an EMR cluster configured for Hail."""
    from hailtop.config import ConfigVariable, configuration_of  # pylint: disable=import-outside-toplevel

    from . import submit as submit_mod  # pylint: disable=import-outside-toplevel

    scratch = configuration_of(ConfigVariable.EMR_REMOTE_TMPDIR, s3_scratch, None)
    if scratch is None:
        raise typer.BadParameter('Provide --s3-scratch or set `hailctl config set emr/remote_tmpdir`.')

    submit_mod.submit(cluster_id, script, scratch, region, ctx.args, wait=not no_wait)
