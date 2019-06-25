def init_parser(parser):
        parser.add_argument('')

def main(args, passthrough_args, client):
    print(client.list_batches())