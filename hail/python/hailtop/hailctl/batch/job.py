from .batch_cli_utils import get_job_if_exists, make_formatter


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'job',
        help='Get the status and specification for a job',
        description='Get the status and specification for a job'
    )
    parser.set_defaults(module='hailctl batch job')
    parser.add_argument('batch_id', type=int, help="ID number of the desired batch")
    parser.add_argument('job_id', type=int, help="ID number of the desired job")
    parser.add_argument('-o', type=str, default='yaml', help="Specify output format",
                        choices=["yaml", "json"])


def job(args, client):
    maybe_job = get_job_if_exists(client, args.batch_id, args.job_id)
    if maybe_job is None:
        print(f"Job with ID {args.job_id} on batch {args.batch_id} not found")
        return

    print(make_formatter(args.o)(maybe_job._status))
