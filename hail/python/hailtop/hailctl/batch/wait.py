from .batch_cli_utils import get_batch_if_exists

def init_parser(parser):
    parser.add_argument('batch_id', type=int)

def main(args, pass_through_args, client):
    maybe_batch = get_batch_if_exists(client, args.batch_id)
    if maybe_batch is None:
        print("Batch with id {} not found".format(args.batch_id))
        return

    batch = maybe_batch
    print(batch.wait())
