import sys

import argparse

from . import connect
from . import describe
from . import diagnose
from . import gcloud
from . import list_clusters
from . import modify
from . import start
from . import stop
from . import submit


MINIMUM_REQUIRED_GCLOUD_VERSION = (285, 0, 0)


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'dataproc',
        help='Manage and monitor Hail deployments.',
        description='Manage and monitor Hail deployments.')
    parser.add_argument(
        '--beta',
        action='store_true',
        help='Force use of `beta` in gcloud commands')

    subparsers = parser.add_subparsers(
        title='hailctl dataproc subcommand',
        dest='hailctl dataproc subcommand',
        required=True)

    connect.init_parser(subparsers)
    describe.init_parser(subparsers)
    diagnose.init_parser(subparsers)
    list_clusters.init_parser(subparsers)
    modify.init_parser(subparsers)
    start.init_parser(subparsers)
    stop.init_parser(subparsers)
    submit.init_parser(subparsers)


def main(args):
    try:
        gcloud_version = gcloud.get_version()
        if gcloud_version < MINIMUM_REQUIRED_GCLOUD_VERSION:
            print(f"hailctl dataproc requires Google Cloud SDK (gcloud) version {'.'.join(map(str, MINIMUM_REQUIRED_GCLOUD_VERSION))} or higher", file=sys.stderr)
            sys.exit(1)
    except Exception:
        # If gcloud's output format changes in the future and the version can't be parsed,
        # then continue and attempt to run gcloud.
        print("Warning: unable to determine Google Cloud SDK version", file=sys.stderr)

    if args.module.startswith('hailctl dataproc connect'):
        connect.main(args)
    elif args.module.startswith('hailctl dataproc describe'):
        describe.main(args)
    elif args.module.startswith('hailctl dataproc diagnose'):
        diagnose.main(args)
    elif args.module.startswith('hailctl dataproc list'):
        list_clusters.main(args)
    elif args.module.startswith('hailctl dataproc modify'):
        modify.main(args)
    elif args.module.startswith('hailctl dataproc start'):
        start.main(args)
    elif args.module.startswith('hailctl dataproc stop'):
        stop.main(args)
    else:
        assert args.module.startswith('hailctl dataproc submit')
        submit.main(args)
