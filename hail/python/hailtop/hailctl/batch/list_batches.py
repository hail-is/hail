from .batch_cli_utils import make_formatter, TABLE_FORMAT_OPTIONS


def init_parser(parser):
    parser.add_argument('--query', '-q', type=str, help="see docs at https://batch.hail.is/batches")
    parser.add_argument('--limit', '-l', type=int, default=50,
                        help='number of batches to return (default 50)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='list all batches (overrides --limit)')
    parser.add_argument('--before', type=int, help='start listing before supplied id', default=None)
    parser.add_argument('--full', action='store_true',
                        help='when output is tabular, print more information')
    parser.add_argument('--no-header', action='store_true', help='do not print a table header')
    parser.add_argument('-o', type=str, default='grid',
                        choices=TABLE_FORMAT_OPTIONS)


def main(args, passthrough_args, client):  # pylint: disable=unused-argument
    batch_list = client.list_batches(q=args.query, last_batch_id=args.before, limit=args.limit)
    statuses = [batch.last_known_status() for batch in batch_list]

    if len(statuses) == 0:
        print("No batches to display.")
        return

    for status in statuses:
        status['state'] = status['state'].capitalize()

    if args.full:
        statuses = [
            {k: v for k, v in status.items() if k != 'attributes'}
            for status in statuses
        ]
    else:
        statuses = [
            {'id': status['id'], 'state': status['state']}
            for status in statuses
        ]

    format = make_formatter(args.o)
    print(format(statuses))
