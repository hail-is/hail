from collections import namedtuple
from enum import Enum
import re


_config_variables = None

ConfigVariableInfo = namedtuple('ConfigVariable', ['section', 'option', 'help_msg', 'validation'])


class ConfigVariable(str, Enum):
    DOMAIN = 'domain'
    GCS_REQUESTER_PAYS_PROJECT = 'gcs_requester_pays/project',
    GCS_REQUESTER_PAYS_BUCKETS = 'gcs_requester_pays/buckets',
    BATCH_BUCKET = 'batch/bucket'
    BATCH_REMOTE_TMPDIR = 'batch/remote_tmpdir'
    BATCH_REGIONS = 'batch/regions',
    BATCH_BILLING_PROJECT = 'batch/billing_project'
    BATCH_BACKEND = 'batch/backend'
    QUERY_BACKEND = 'query/backend'
    QUERY_JAR_URL = 'query/jar_url'
    QUERY_BATCH_DRIVER_CORES = 'query/batch_driver_cores'
    QUERY_BATCH_WORKER_CORES = 'query/batch_worker_cores'
    QUERY_BATCH_DRIVER_MEMORY = 'query/batch_driver_memory'
    QUERY_BATCH_WORKER_MEMORY = 'query/batch_worker_memory'
    QUERY_NAME_PREFIX = 'query/name_prefix'
    QUERY_DISABLE_PROGRESS_BAR = 'query/disable_progress_bar'


def config_variables():
    from hailtop.batch_client.parse import CPU_REGEXPAT, MEMORY_REGEXPAT  # pylint: disable=import-outside-toplevel
    from hailtop.fs.router_fs import RouterAsyncFS  # pylint: disable=import-outside-toplevel

    global _config_variables

    if _config_variables is None:
        _config_variables = {
            ConfigVariable.DOMAIN: ConfigVariableInfo(
                section='global',
                option='domain',
                help_msg='Domain of the Batch service',
                validation=(lambda x: re.fullmatch(r'.+\..+', x) is not None, 'should be valid domain'),
            ),
            ConfigVariable.GCS_REQUESTER_PAYS_PROJECT: ConfigVariableInfo(
                section='gcs_requester_pays',
                option='project',
                help_msg='Project when using requester pays buckets in GCS',
                validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid GCS project name'),
            ),
            ConfigVariable.GCS_REQUESTER_PAYS_BUCKETS: ConfigVariableInfo(
                section='gcs_requester_pays',
                option='buckets',
                help_msg='Allowed buckets when using requester pays in GCS',
                validation=(
                    lambda x: re.fullmatch(r'[^:/\s]+(,[^:/\s]+)*', x) is not None,
                    'should be comma separated list of bucket names'),
            ),
            ConfigVariable.BATCH_BUCKET: ConfigVariableInfo(
                section='batch',
                option='bucket',
                help_msg='Deprecated - Name of GCS bucket to use as a temporary scratch directory',
                validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None,
                            'should be valid Google Bucket identifier, with no gs:// prefix'),
            ),
            ConfigVariable.BATCH_REMOTE_TMPDIR: ConfigVariableInfo(
                section='batch',
                option='remote_tmpdir',
                help_msg='Cloud storage URI to use as a temporary scratch directory',
                validation=(RouterAsyncFS.valid_url, 'should be valid cloud storage URI such as gs://my-bucket/batch-tmp/'),
            ),
            ConfigVariable.BATCH_REGIONS: ConfigVariableInfo(
                section='batch',
                option='regions',
                help_msg='Comma-separated list of regions to run jobs in',
                validation=(
                    lambda x: re.fullmatch(r'[^\s]+(,[^\s]+)*', x) is not None, 'should be comma separated list of regions'),
            ),
            ConfigVariable.BATCH_BILLING_PROJECT: ConfigVariableInfo(
                section='batch',
                option='billing_project',
                help_msg='Batch billing project',
                validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid Batch billing project name'),
            ),
            ConfigVariable.BATCH_BACKEND: ConfigVariableInfo(
                section='batch',
                option='backend',
                help_msg='Backend to use. One of local or service.',
                validation=(lambda x: x in ('local', 'service'), 'should be one of "local" or "service"'),
            ),
            ConfigVariable.QUERY_BACKEND: ConfigVariableInfo(
                section='query',
                option='backend',
                help_msg='Backend to use for Hail Query. One of spark, local, batch.',
                validation=(lambda x: x in ('local', 'spark', 'batch'), 'should be one of "local", "spark", or "batch"'),
            ),
            ConfigVariable.QUERY_JAR_URL: ConfigVariableInfo(
                section='query',
                option='jar_url',
                help_msg='Cloud storage URI to a Query JAR',
                validation=(RouterAsyncFS.valid_url, 'should be valid cloud storage URI such as gs://my-bucket/jars/sha.jar')
            ),
            ConfigVariable.QUERY_BATCH_DRIVER_CORES: ConfigVariableInfo(
                section='query',
                option='batch_driver_cores',
                help_msg='Cores specification for the query driver',
                validation=(lambda x: re.fullmatch(CPU_REGEXPAT, x) is not None,
                            'should be an integer which is a power of two from 1 to 16 inclusive'),
            ),
            ConfigVariable.QUERY_BATCH_WORKER_CORES: ConfigVariableInfo(
                section='query',
                option='batch_worker_cores',
                help_msg='Cores specification for the query worker',
                validation=(lambda x: re.fullmatch(CPU_REGEXPAT, x) is not None,
                            'should be an integer which is a power of two from 1 to 16 inclusive'),
            ),
            ConfigVariable.QUERY_BATCH_DRIVER_MEMORY: ConfigVariableInfo(
                section='query',
                option='batch_driver_memory',
                help_msg='Memory specification for the query driver',
                validation=(lambda x: re.fullmatch(MEMORY_REGEXPAT, x) is not None or x in ('standard', 'lowmem', 'highmem'),
                            'should be a valid string specifying memory "[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?" or one of standard, lowmem, highmem'),
            ),
            ConfigVariable.QUERY_BATCH_WORKER_MEMORY: ConfigVariableInfo(
                section='query',
                option='batch_worker_memory',
                help_msg='Memory specification for the query worker',
                validation=(lambda x: re.fullmatch(MEMORY_REGEXPAT, x) is not None or x in ('standard', 'lowmem', 'highmem'),
                            'should be a valid string specifying memory "[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?" or one of standard, lowmem, highmem'),
            ),
            ConfigVariable.QUERY_NAME_PREFIX: ConfigVariableInfo(
                section='query',
                option='name_prefix',
                help_msg='Name used when displaying query progress in a progress bar',
                validation=(lambda x: re.fullmatch(r'[^\s]+', x) is not None, 'should be single word without spaces'),
            ),
            ConfigVariable.QUERY_DISABLE_PROGRESS_BAR: ConfigVariableInfo(
                section='query',
                option='disable_progress_bar',
                help_msg='Disable the progress bar with a value of 1. Enable the progress bar with a value of 0',
                validation=(lambda x: x in ('0', '1'), 'should be a value of 0 or 1'),
            ),
        }

    return _config_variables
