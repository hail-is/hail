from .batch_cli_utils import get_batch_if_exists

def init_parser(parser):
    parser.add_argument('id', type=int)

def main(args, pass_through_args, client):
    maybe_batch = get_batch_if_exists(client, args.id)
    if maybe_batch is None:
        print("Batch with id {} not found".format(args.id))
        return

    batch = maybe_batch

    batch.delete()
    print("Batch with id {} was deleted successfully".format(args.id))
