from subprocess import check_call

def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('--async', action='store_true', help="Do not wait for cluster deletion.")

def main(args):
    print("Stopping cluster '{}'...".format(args.name))

    cmd = ['gcloud', 'dataproc', 'clusters', 'delete', '--quiet', args.name]
    if vars(args)['async']:
        cmd.append('--async')

    check_call(cmd)
