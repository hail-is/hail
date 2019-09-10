from subprocess import check_call


def main(args, pass_through_args):  # pylint: disable=unused-argument
    check_call(['gcloud', 'dataproc', 'clusters', 'list'] + pass_through_args)
