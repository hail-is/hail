import base64
import json
import logging
import os
from shlex import quote as shq
from typing import Dict, Optional

from ....file_store import FileStore
from ....instance_config import InstanceConfig
from ...utils import ACCEPTABLE_QUERY_JAR_URL_PREFIX

log = logging.getLogger('create_instance')

BATCH_WORKER_IMAGE = os.environ['HAIL_BATCH_WORKER_IMAGE']


log.info(f'BATCH_WORKER_IMAGE {BATCH_WORKER_IMAGE}')
log.info(f'ACCEPTABLE_QUERY_JAR_URL_PREFIX {ACCEPTABLE_QUERY_JAR_URL_PREFIX}')


def create_vm_config(
    file_store: FileStore,  # FIXME: how does Lambda get the logs into the right place in GCP bucket?
    resource_rates: Dict[str, float],
    zone: str,
    machine_name: str,
    machine_type: str,
    activation_token: str,
    max_idle_time_msecs: int,
    local_ssd_data_disk: bool,
    data_disk_size_gb: int,
    boot_disk_size_gb: int,
    preemptible: bool,
    job_private: bool,
    project: str,
    instance_config: InstanceConfig,
    gpu_config: Optional[GPUConfig] = None,
) -> dict:
    raise NotImplementedError
