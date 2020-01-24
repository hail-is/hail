import tabulate


def init_parser(parser):
    parser.add_argument('--query', '-q', type=str, help="see docs at https://batch.hail.is/batches")


def main(args, passthrough_args, client):  # pylint: disable=unused-argument
    batch_list = client.list_batches(q=args.query)
    pretty_batches = [[batch.id, batch.status()['state'].capitalize()] for batch in batch_list]

    print(tabulate.tabulate(pretty_batches, headers=["ID", "STATUS"], tablefmt='orgtbl'))
