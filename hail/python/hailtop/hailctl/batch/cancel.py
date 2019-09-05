from .batch_cli_utils import get_batch_if_exists


def init_parser(parser):
    parser.add_argument('id', type=int)


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    maybe_batch = get_batch_if_exists(client, args.id)
    if maybe_batch is None:
        print(f"Batch with id {args.id} not found")
        return

    batch = maybe_batch

    batch.cancel()
    print(f"Batch with id {args.id} was cancelled successfully")
