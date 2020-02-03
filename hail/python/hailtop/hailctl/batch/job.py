from .batch_cli_utils import get_job_if_exists, make_formatter


def init_parser(parser):
    parser.add_argument('batch_id', type=int, help="ID number of the desired batch")
    parser.add_argument('job_id', type=int, help="ID number of the desired job")
    parser.add_argument('-o', type=str, default='yaml', help="Specify output format",
                        choices=["yaml", "json"])


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    maybe_job = get_job_if_exists(client, args.batch_id, args.job_id)
    if maybe_job is None:
        print(f"Job with ID {args.job_id} on batch {args.batch_id} not found")
        return

    print(make_formatter(args.o)(maybe_job._status))
