from subprocess import check_call

def main(args, pass_through_args):
    check_call(['gcloud', 'dataproc', 'clusters', 'list'])
