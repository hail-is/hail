from subprocess import check_call

def main(args):
    check_call(['gcloud', 'dataproc', 'clusters', 'list'])
