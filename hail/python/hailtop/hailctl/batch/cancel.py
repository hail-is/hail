from .batch_cli_utils import get_batch_if_exists


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'cancel',
        help='Cancel a batch',
        description='Cancel a batch')
    parser.set_defaults(module='hailctl batch cancel')
    parser.add_argument('id', type=int)


def main(args, client):
    maybe_batch = get_batch_if_exists(client, args.id)
    if maybe_batch is None:
        print(f"Batch with id {args.id} not found")
        return

    batch = maybe_batch

    batch.cancel()
    print(f"Batch with id {args.id} was cancelled successfully")
