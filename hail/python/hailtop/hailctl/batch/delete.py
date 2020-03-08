from .batch_cli_utils import get_batch_if_exists


def init_parser(parser):
    parser.add_argument('batch_id', type=int, help="ID number of batch to be deleted")


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    maybe_batch = get_batch_if_exists(client, args.batch_id)
    if maybe_batch is None:
        print(f"Batch with batch_id {args.batch_id} not found")
        exit(1)

    batch = maybe_batch

    batch.delete()
    print(f"Batch with batch_id {args.batch_id} was deleted successfully")
