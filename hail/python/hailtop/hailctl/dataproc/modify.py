import os.path
import sys
from subprocess import check_call
import yaml
import pkg_resources


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('--wheel', type=str, help='New Hail installation.')
    parser.add_argument('--num-workers', '--n-workers', '-w', type=int,
                        help='New number of worker machines (min. 2).')
    parser.add_argument('--num-preemptible-workers', '--n-pre-workers', '-p', type=int,
                        help='New number of preemptible worker machines.')
    parser.add_argument('--graceful-decommission-timeout', '--graceful', type=str,
                        help='If set, cluster size downgrade will use graceful decommissioning with the given timeout (e.g. "60m").')
    parser.add_argument('--max-idle', type=str, help='New maximum idle time before shutdown (e.g. "60m").')
    parser.add_argument('--dry-run', action='store_true', help="Print gcloud dataproc command, but don't run it.")
    parser.add_argument('--zone', '-z', default='us-central1-b', type=str,
                        help='Compute zone for Dataproc cluster (default: %(default)s).')
    parser.add_argument('--update-hail-version', action='store_true', help="Update the version of hail running on cluster to match "
                        "the currently installed version.")


def main(args, pass_through_args):
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

    if args.update_hail_version and args.wheel:
        sys.exit("Error: Cannot specify both --update-hail-version and --wheel")

    if modify_args:
        cmd = ['gcloud',
               'dataproc',
               'clusters',
               'update',
               args.name] + modify_args

        if args.beta:
            cmd.insert(1, 'beta')

        cmd.extend(pass_through_args)

        # print underlying gcloud command
        print(' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[5:]))

        # Update cluster
        if not args.dry_run:
            print("Updating cluster '{}'...".format(args.name))
            check_call(cmd)

    wheel = None
    if args.update_hail_version:
        assert pkg_resources.resource_exists('hailtop.hailctl', "deploy.yaml")
        updated_wheel = yaml.safe_load(
            pkg_resources.resource_stream('hailtop.hailctl', "deploy.yaml"))['dataproc']['wheel']
        wheel = updated_wheel
    else:
        wheel = args.wheel

    if wheel is not None:
        wheelfile = os.path.basename(wheel)
        cmds = []
        if wheel.startswith("gs://"):
            cmds.append([
                'gcloud',
                'compute',
                'ssh',
                '{}-m'.format(args.name),
                '--zone={}'.format(args.zone),
                '--',
                f'sudo gsutil cp {wheel} /tmp/ && '
                'sudo /opt/conda/default/bin/pip uninstall -y hail && '
                f'sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/{wheelfile}'
            ])
        else:
            cmds.extend([
                [
                    'gcloud',
                    'compute',
                    'scp',
                    '--zone={}'.format(args.zone),
                    wheel,
                    '{}-m:/tmp/'.format(args.name)
                ],
                [
                    'gcloud',
                    'compute',
                    'ssh',
                    f'{args.name}-m',
                    f'--zone={args.zone}',
                    '--',
                    'sudo /opt/conda/default/bin/pip uninstall -y hail && '
                    f'sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/{wheelfile}'
                ]
            ])

        for cmd in cmds:
            print(cmd)
            check_call(cmd)
