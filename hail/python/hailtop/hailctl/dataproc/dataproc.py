import sys
import click

from . import gcloud
from ..hailctl import hailctl

MINIMUM_REQUIRED_GCLOUD_VERSION = (285, 0, 0)


@hailctl.group(
    help="""Manage and monitor Hail deployments.

When invoking 'gcloud dataproc' commands, the region is taken from the
gcloud 'dataproc/region' setting, and if 'dataproc/region' is not set,
determined from the zone.""")
@click.option('--beta', is_flag=True,
              help="Use of 'beta' in gcloud commands.")
@click.option('--dry-run', is_flag=True,
              help="Print gcloud commands, but don't run them.")
@click.option('--gcloud-configuration',
              help="Google Cloud configuration to when invoking gcloud.  If not specified, use gcloud default configuration.")
@click.option('--project',
              metavar='GCP_PROJECT',
              help="Google Cloud project to use.")
@click.option('--zone', '-z',
              metavar='GCP_ZONE',
              help="Compute zone to use.  If not specified, use gcloud 'compute/zone' setting.")
@click.pass_context
def dataproc(ctx, beta, dry_run, gcloud_configuration, project, zone):  # pylint: disable=unused-argument
    try:
        gcloud_version = gcloud.get_version()
        if gcloud_version < MINIMUM_REQUIRED_GCLOUD_VERSION:
            print(f"hailctl dataproc requires Google Cloud SDK (gcloud) version {'.'.join(map(str, MINIMUM_REQUIRED_GCLOUD_VERSION))} or higher", file=sys.stderr)
            sys.exit(1)
    except Exception:
        # If gcloud's output format changes in the future and the version can't be parsed,
        # then continue and attempt to run gcloud.
        print("Warning: unable to determine Google Cloud SDK version", file=sys.stderr)

    ctx.obj = gcloud.GCloudRunner(beta, dry_run, gcloud_configuration, project, zone)
