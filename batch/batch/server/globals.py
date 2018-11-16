import os
import uuid
from .log import log

KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))
REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 5 * 60))
POD_NAMESPACE = os.environ.get('POD_NAMESPACE', 'batch-pods')
INSTANCE_ID = uuid.uuid4().hex

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')
log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')
log.info(f'POD_NAMESPACE {POD_NAMESPACE}')
log.info(f'INSTANCE_ID = {INSTANCE_ID}')

pod_name_job = {}
job_id_job = {}
batch_id_batch = {}
dag_id_dag = {}


def _log_path(id):
    return f'logs/job-{id}.log'


def _read_file(fname):
    with open(fname, 'r') as f:
        return f.read()


_counter = 0


def max_id():
    return _counter


def next_id():
    global _counter

    _counter = _counter + 1
    return _counter
