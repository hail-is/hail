from collections import namedtuple
import re

from hailtop.config import ConfigVariable


_config_variables = None

ConfigVariableInfo = namedtuple('ConfigVariableInfo', ['help_msg', 'validation'])


def config_variables():
    from hailtop.batch_client.parse import CPU_REGEXPAT, MEMORY_REGEXPAT  # pylint: disable=import-outside-toplevel
    from hailtop.fs.router_fs import RouterAsyncFS  # pylint: disable=import-outside-toplevel

    global _config_variables

    if _config_variables is None:
        _config_variables = {
            ConfigVariable.DOMAIN: ConfigVariableInfo(
                help_msg='Domain of the Batch service',
                validation=(lambda x: re.fullmatch(r'.+\..+', x) is not None, 'should be valid domain'),
            ),
            ConfigVariable.GCS_REQUESTER_PAYS_PROJECT: ConfigVariableInfo(
                help_msg='Project when using requester pays buckets in GCS',
                validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid GCS project name'),
            ),
            ConfigVariable.GCS_REQUESTER_PAYS_BUCKETS: ConfigVariableInfo(
                help_msg='Allowed buckets when using requester pays in GCS',
                validation=(
                    lambda x: re.fullmatch(r'[^:/\s]+(,[^:/\s]+)*', x) is not None,
                    'should be comma separated list of bucket names'),
            ),
            ConfigVariable.BATCH_BUCKET: ConfigVariableInfo(
                help_msg='Deprecated - Name of GCS bucket to use as a temporary scratch directory',
                validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None,
                            'should be valid Google Bucket identifier, with no gs:// prefix'),
            ),
            ConfigVariable.BATCH_REMOTE_TMPDIR: ConfigVariableInfo(
                help_msg='Cloud storage URI to use as a temporary scratch directory',
                validation=(RouterAsyncFS.valid_url, 'should be valid cloud storage URI such as gs://my-bucket/batch-tmp/'),
            ),
            ConfigVariable.BATCH_REGIONS: ConfigVariableInfo(
                help_msg='Comma-separated list of regions to run jobs in',
                validation=(
                    lambda x: re.fullmatch(r'[^\s]+(,[^\s]+)*', x) is not None, 'should be comma separated list of regions'),
            ),
            ConfigVariable.BATCH_BILLING_PROJECT: ConfigVariableInfo(
                help_msg='Batch billing project',
                validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid Batch billing project name'),
            ),
            ConfigVariable.BATCH_BACKEND: ConfigVariableInfo(
                help_msg='Backend to use. One of local or service.',
                validation=(lambda x: x in ('local', 'service'), 'should be one of "local" or "service"'),
            ),
            ConfigVariable.QUERY_BACKEND: ConfigVariableInfo(
                help_msg='Backend to use for Hail Query. One of spark, local, batch.',
                validation=(lambda x: x in ('local', 'spark', 'batch'), 'should be one of "local", "spark", or "batch"'),
            ),
            ConfigVariable.QUERY_JAR_URL: ConfigVariableInfo(
                help_msg='Cloud storage URI to a Query JAR',
                validation=(RouterAsyncFS.valid_url, 'should be valid cloud storage URI such as gs://my-bucket/jars/sha.jar')
            ),
            ConfigVariable.QUERY_BATCH_DRIVER_CORES: ConfigVariableInfo(
                help_msg='Cores specification for the query driver',
                validation=(lambda x: re.fullmatch(CPU_REGEXPAT, x) is not None,
                            'should be an integer which is a power of two from 1 to 16 inclusive'),
            ),
            ConfigVariable.QUERY_BATCH_WORKER_CORES: ConfigVariableInfo(
                help_msg='Cores specification for the query worker',
                validation=(lambda x: re.fullmatch(CPU_REGEXPAT, x) is not None,
                            'should be an integer which is a power of two from 1 to 16 inclusive'),
            ),
            ConfigVariable.QUERY_BATCH_DRIVER_MEMORY: ConfigVariableInfo(
                help_msg='Memory specification for the query driver',
                validation=(lambda x: re.fullmatch(MEMORY_REGEXPAT, x) is not None or x in ('standard', 'lowmem', 'highmem'),
                            'should be a valid string specifying memory "[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?" or one of standard, lowmem, highmem'),
            ),
            ConfigVariable.QUERY_BATCH_WORKER_MEMORY: ConfigVariableInfo(
                help_msg='Memory specification for the query worker',
                validation=(lambda x: re.fullmatch(MEMORY_REGEXPAT, x) is not None or x in ('standard', 'lowmem', 'highmem'),
                            'should be a valid string specifying memory "[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?" or one of standard, lowmem, highmem'),
            ),
            ConfigVariable.QUERY_NAME_PREFIX: ConfigVariableInfo(
                help_msg='Name used when displaying query progress in a progress bar',
                validation=(lambda x: re.fullmatch(r'[^\s]+', x) is not None, 'should be single word without spaces'),
            ),
            ConfigVariable.QUERY_DISABLE_PROGRESS_BAR: ConfigVariableInfo(
                help_msg='Disable the progress bar with a value of 1. Enable the progress bar with a value of 0',
                validation=(lambda x: x in ('0', '1'), 'should be a value of 0 or 1'),
            ),
        }

    return _config_variables
