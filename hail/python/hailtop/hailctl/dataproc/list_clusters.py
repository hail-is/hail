from . import gcloud


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'list',
        help='List active Dataproc clusters.',
        description='List active Dataproc clusters.')
    parser.set_defaults(module='hailctl dataproc list', allow_unknown_args=True)


def main(args):
    gcloud.run(['dataproc', 'clusters', 'list'] + args.unknown_args)
