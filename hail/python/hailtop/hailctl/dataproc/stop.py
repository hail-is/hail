import argparse

from . import gcloud


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'stop',
        help='Shut down a Dataproc cluster.',
        description='Shut down a Dataproc cluster.')
    parser.set_defaults(module='hailctl dataproc stop', allow_unknown_args=True)

    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('--async', action='store_true', dest='asink',
                        help="Do not wait for cluster deletion.")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print gcloud dataproc command, but don't run it.")


def main(args):
    print("Stopping cluster '{}'...".format(args.name))

    cmd = ['dataproc', 'clusters', 'delete', '--quiet', args.name]
    if args.asink:
        cmd.append('--async')

    if args.unknown_args:
        cmd.extend(args.unknown_args)

    # print underlying gcloud command
    print('gcloud ' + ' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[6:]))

    if not args.dry_run:
        gcloud.run(cmd)
