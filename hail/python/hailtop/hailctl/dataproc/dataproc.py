import sys
import click

from . import gcloud
from ..hailctl import hailctl

MINIMUM_REQUIRED_GCLOUD_VERSION = (285, 0, 0)


@hailctl.group(
    help='Manage and monitor Hail deployments')
@click.option('--beta',
              help='Force use of `beta` in gcloud commands')
def dataproc(beta):  # pylint: disable=unused-argument
    try:
        gcloud_version = gcloud.get_version()
        if gcloud_version < MINIMUM_REQUIRED_GCLOUD_VERSION:
            print(f"hailctl dataproc requires Google Cloud SDK (gcloud) version {'.'.join(map(str, MINIMUM_REQUIRED_GCLOUD_VERSION))} or higher", file=sys.stderr)
            sys.exit(1)
    except Exception:
        # If gcloud's output format changes in the future and the version can't be parsed,
        # then continue and attempt to run gcloud.
        print("Warning: unable to determine Google Cloud SDK version", file=sys.stderr)
