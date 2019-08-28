import argparse
import tabulate

from .batch_cli_utils import bool_string_to_bool


def init_parser(parser):
    parser.add_argument('--success', '-s', type=str, help="true or false")
    parser.add_argument('--complete', '-c', type=str, help="true or false")
    parser.add_argument('--attributes', '-a', metavar="KEY=VALUE", nargs='+',
                        help="Filters list to specified attributes. Specify attributes using"
                        "KEY=VALUE form, do not put spaces before or after the equal sign.")


def main(args, passthrough_args, client):  # pylint: disable=unused-argument
    success = None
    if args.success:
        try:
            success = bool_string_to_bool(args.success)
        except ValueError:
            raise argparse.ArgumentTypeError("Boolean value expected for success")

    complete = None
    if args.complete:
        try:
            complete = bool_string_to_bool(args.complete)
        except ValueError:
            raise argparse.ArgumentTypeError("Boolean value expected for complete")

    attributes = {}
    if args.attributes:
        for att in args.attributes:
            att = att.strip()
            key_value = att.split('=')
            if len(key_value) != 2:
                raise argparse.ArgumentTypeError(f'Attribute {att!r} should contain exactly one equal sign')
            attributes[key_value[0]] = key_value[1]

    batch_list = client.list_batches(success=success, complete=complete, attributes=attributes)
    pretty_batches = [[batch.id, batch.status()['state'].capitalize()] for batch in batch_list]

    print(tabulate.tabulate(pretty_batches, headers=["ID", "STATUS"], tablefmt='orgtbl'))
