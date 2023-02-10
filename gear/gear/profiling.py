import logging
import os

import googlecloudprofiler

HAIL_SHA = os.environ['HAIL_SHA']
HAIL_SHOULD_PROFILE = 'HAIL_SHOULD_PROFILE' in os.environ
DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
CLOUD = os.environ['CLOUD']


def install_profiler_if_requested(service: str):
    if HAIL_SHOULD_PROFILE and CLOUD == 'gcp':
        profiler_tag = DEFAULT_NAMESPACE
        if profiler_tag == 'default':
            profiler_tag = DEFAULT_NAMESPACE + f'-{HAIL_SHA[0:12]}'
        googlecloudprofiler.start(
            service=service,
            service_version=profiler_tag,
            # https://cloud.google.com/profiler/docs/profiling-python#agent_logging
            verbose=3,
        )

        def ignore_failed_to_collect_and_upload_profile(record):
            if 'Failed to collect and upload profile: [Errno 32] Broken pipe' in record.msg:
                record.levelno = logging.INFO
                record.levelname = "INFO"
            return record

        googlecloudprofiler.logger.addFilter(ignore_failed_to_collect_and_upload_profile)
