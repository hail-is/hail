from subprocess import call

def main(args):
    call(['gcloud', 'dataproc', 'clusters', 'list'])
