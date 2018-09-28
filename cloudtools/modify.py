import sys
from subprocess import check_call


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('--jar', type=str, help='New Hail JAR.')
    parser.add_argument('--zip', type=str, help='New Hail ZIP.')
    parser.add_argument('--num-workers', '--n-workers', '-w', type=int,
                        help='New number of worker machines (min. 2).')
    parser.add_argument('--num-preemptible-workers', '--n-pre-workers', '-p', type=int,
                        help='New number of preemptible worker machines.')
    parser.add_argument('--graceful-decommission-timeout', '--graceful', type=str,
                        help='If set, cluster size downgrade will use graceful decommissionnig with the given timeout (e.g. "60m").')
    parser.add_argument('--max-idle', type=str, help='New maximum idle time before shutdown (e.g. "60m").')
    parser.add_argument('--dry-run', action='store_true', help="Print gcloud dataproc command, but don't run it.")
    parser.add_argument('--zone', '-z', default='us-central1-b', type=str,
                        help='Compute zone for Dataproc cluster (default: %(default)s).')

def main(args):

    modify_args = []
    if args.num_workers is not None:
        modify_args.append('--num-workers={}'.format(args.num_workers))

    if args.num_preemptible_workers is not None:
        modify_args.append('--num-preemptible-workers={}'.format(args.num_preemptible_workers))

    if args.graceful_decommission_timeout:
        if not modify_args:
            sys.exit("Error: Cannot use --graceful-decommission-timeout without resizing the cluster.")
        modify_args.append('--graceful-decommission-timeout={}'.format(args.graceful_decommission_timeout))

    if args.max_idle:
        modify_args.append('--max-idle={}'.format(args.max_idle))

    if modify_args:
        cmd = [
            'gcloud',
            'dataproc',
            'clusters',
            'update',
            args.name] + modify_args

        if args.max_idle or args.graceful_decommission_timeout:
            cmd.insert(1, 'beta')

        # print underlying gcloud command
        print('gcloud update config command:')
        print(' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[5:]))

        # Update cluster
        if not args.dry_run:
            print("Updating cluster '{}'...".format(args.name))
            check_call(cmd)

    if (args.jar is not None):
        print('gcloud jar update command(s):')
        _scp_and_sudo_move(args.jar, args.name, '/home/hail/hail.jar', args.zone)
    if (args.zip is not None):
        print('gcloud zip update command(s):')
        _scp_and_sudo_move(args.zip, args.name, '/home/hail/hail.zip', args.zone)


# user doesn't have access to /home/hail/ so we copy then use sudo
def _scp_and_sudo_move(source, destination_host, destination, zone):
    cmds = []
    if source.startswith("gs://"):
        cmds.append([
            'gcloud',
            'compute',
            'ssh',
            '{}-m'.format(destination_host),
            '--zone={}'.format(zone),
            '--',
            'sudo gsutil cp {} {}'.format(source, destination)
        ])
    else:
        cmds.extend([
            [
                'gcloud',
                'compute',
                'scp',
                '--zone={}'.format(zone),
                source,
                '{}-m:/tmp/foo'.format(destination_host)
            ],
            [
                'gcloud',
                'compute',
                'ssh',
                '{}-m'.format(destination_host),
                '--zone={}'.format(zone),
                '--',
                'sudo mv /tmp/foo {}'.format(destination)
            ]
        ])

    for cmd in cmds:
        print(cmd)
        check_call(cmd)
