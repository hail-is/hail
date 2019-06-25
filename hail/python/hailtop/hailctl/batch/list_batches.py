import tabulate

def init_parser(parser):
        parser.add_argument('')

def main(args, passthrough_args, client):
    batch_list = client.list_batches()
    pretty_batches = [[batch.id, batch.status()['state'].capitalize()] for batch in batch_list]

    print(tabulate.tabulate(pretty_batches, headers=["ID", "STATUS"], tablefmt='orgtbl'))