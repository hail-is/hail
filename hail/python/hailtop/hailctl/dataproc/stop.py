from . import gcloud


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('--async', action='store_true', dest='asink',
                        help="Do not wait for cluster deletion.")
    parser.add_argument('--region', help='Region.', required=True)
    parser.add_argument('--dry-run', action='store_true',
                        help="Print gcloud dataproc command, but don't run it.")


async def main(args, pass_through_args):
    print("Stopping cluster '{}'...".format(args.name))

    cmd = [
        'dataproc', 
        'clusters', 
        'delete', 
        '--region={}'.format(args.region), 
        '--quiet', 
        args.name
    ]
    if args.asink:
        cmd.append('--async')

    cmd.extend(pass_through_args)

    # print underlying gcloud command
    print('gcloud ' + ' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[6:]))

    if not args.dry_run:
        gcloud.run(cmd)
