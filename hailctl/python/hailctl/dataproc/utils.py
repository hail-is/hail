import subprocess as sp

from .cluster_config import ClusterConfig

LATEST_ROOT = 'gs://hail-common/builds/0.2/latest-hash/cloudtools-4-spark-{spark}.txt'
CONFIG_ROOT = 'gs://hail-common/builds/0.2/config/hail-config-0.2-{sha}.json'


def latest_sha(spark):
    # FIXME
    hash_file = LATEST_ROOT.format(spark=spark)
    return sp.check_output(['gsutil', 'cat', hash_file]).decode().strip()


def get_config_filename(sha):
    fname = CONFIG_ROOT.format(sha=sha)
    if sp.call(['gsutil', '-q', 'stat', fname]) != 0:
        return 'gs://hail-common/builds/0.2/config/hail-config-0.2-default.json'
    return fname


def load_config_file(fname):
    if fname.startswith('gs://'):
        return ClusterConfig(sp.check_output(['gsutil', 'cat', fname]).strip())
    return ClusterConfig(sp.check_output(['cat', fname]).strip())


def load_config(sha):
    return load_config_file(get_config_filename(sha))


def safe_call(*args):
    # only print output on error
    try:
        sp.check_output(args, stderr=sp.STDOUT)
    except sp.CalledProcessError as e:
        print(e.output.decode())
        raise e
