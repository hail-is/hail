import tabulate
import argparse

def init_parser(parser):
    parser.add_argument('--success', type=str, help="true or false")
    parser.add_argument('--complete', type=str, help="true or false")

def bool_string_to_bool(bool_string):
    if bool_string in ["True", "true", "t"]:
        return True
    elif bool_string in ['False', 'false', 'f']:
        return False
    else:
        raise ValueError("Input could not be resolved to a bool")

def main(args, passthrough_args, client):
    success = None
    if args.success:
        try:
            success = bool_string_to_bool(args.success)
        except:
            raise argparse.ArgumentTypeError("Boolean value expected for success")

    complete = None
    if args.complete:
        try:
            complete = bool_string_to_bool(args.complete)
        except:
            raise argparse.ArgumentTypeError("Boolean value expected for complete")

    batch_list = client.list_batches(success=success, complete=complete)
    pretty_batches = [[batch.id, batch.status()['state'].capitalize()] for batch in batch_list]

    print(tabulate.tabulate(pretty_batches, headers=["ID", "STATUS"], tablefmt='orgtbl'))
