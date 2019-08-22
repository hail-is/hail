from subprocess import check_call


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('--async', action='store_true', dest='asink',
                        help="Do not wait for cluster deletion.")


def main(args, pass_through_args):  # pylint: disable=unused-argument
    print("Stopping cluster '{}'...".format(args.name))

    cmd = ['gcloud', 'dataproc', 'clusters', 'delete', '--quiet', args.name]
    if args.asink:
        cmd.append('--async')

    check_call(cmd)
