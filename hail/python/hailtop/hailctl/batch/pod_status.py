from .batch_cli_utils import get_job_if_exists


def init_parser(parser):
    parser.add_argument('batch_id', type=int, help="ID number of the desired batch")
    parser.add_argument('job_id', type=int, help="ID number of the desired job")


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    maybe_job = get_job_if_exists(client, args.batch_id, args.job_id)
    if maybe_job is None:
        print(f"Job with ID {args.job_id} on batch {args.batch_id} not found")
        return

    pod_status = maybe_job.pod_status()

    if 'setup' in pod_status:
        print("Setup Pod Status:")
        print(pod_status['setup'])

    if 'main' in pod_status:
        print("Main Pod Status:")
        print(pod_status['main'])

    if 'cleanup' in pod_status:
        print("Cleanup Pod Status:")
        print(pod_status['cleanup'])
