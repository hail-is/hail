from .batch_cli_utils import get_job_if_exists

def init_parser(parser):
    parser.add_argument('batch_id', type=int)
    parser.add_argument('job_id', type=int)

def main(args, pass_through_args, client):
    maybe_job= get_job_if_exists(client, args.batch_id, args.job_id)
    if maybe_job is None:
        print("Job with id {} on batch {} not found".format(args.job_id, args.batch_id))
        return

    print(maybe_job.log())