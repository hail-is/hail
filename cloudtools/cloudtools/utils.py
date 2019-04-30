from .cluster_config import ClusterConfig
import subprocess as sp
import sys
from . import __version__


if sys.version_info >= (3,0):
    decode = lambda s: s.decode()
    # Python 3 check_output returns a byte string
else:
    # In Python 2, bytes and str are the same
    decode = lambda s: s

def latest_sha(version, spark):
    cloudtools_version = __version__.strip().split('.')
    hash_file = 'gs://hail-common/builds/{}/latest-hash/cloudtools-{}-spark-{}.txt'.format(
        version,
        cloudtools_version[0],
        spark)
    return decode(sp.check_output(['gsutil', 'cat', hash_file]).strip())


def get_config_filename(sha, version):
    fname = 'gs://hail-common/builds/{version}/config/hail-config-{version}-{hash}.json'.format(
        version=version, hash=sha)
    if sp.call(['gsutil', '-q', 'stat', fname]) != 0:
        return 'gs://hail-common/builds/{version}/config/hail-config-{version}-default.json'.format(
            version=version)
    return fname


def load_config_file(fname):
    if fname.startswith('gs://'):
        return ClusterConfig(sp.check_output(['gsutil', 'cat', fname]).strip())
    return ClusterConfig(sp.check_output(['cat', fname]).strip())


def load_config(sha, version):
    return load_config_file(get_config_filename(sha, version))
