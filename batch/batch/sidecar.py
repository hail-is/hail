import asyncio
import concurrent
import json
import logging
import os
import subprocess as sp
import sys
import time
import traceback

import kubernetes as kube
from hailtop import gear

from .batch_configuration import REFRESH_INTERVAL_IN_SECONDS, HAIL_POD_NAMESPACE
from .batch_configuration import KUBERNETES_TIMEOUT_IN_SECONDS
from .blocking_to_async import blocking_to_async
from .datetime_json import JSON_ENCODER
from .k8s import K8s
from .log_store import LogStore


pod_name = os.environ['POD_NAME']
batch_instance = os.environ['INSTANCE_ID']
copy_output_cmd = os.environ.get('COPY_OUTPUT_CMD')
batch_id = os.environ['BATCH_ID']
job_id = os.environ['JOB_ID']
token = os.environ['TOKEN']
batch_bucket_name = os.environ['BATCH_BUCKET_NAME']

gear.configure_logging()
log = logging.getLogger('batch-sidecar')

kube.config.load_incluster_config()
v1 = kube.client.CoreV1Api()

log_store = LogStore(concurrent.futures.ThreadPoolExecutor(),
                     instance_id=batch_instance,
                     batch_bucket_name=batch_bucket_name)
directory = log_store.gs_job_output_directory(batch_id, job_id, token)


def container_statuses(pod):
    statuses = {status.name: status for status in pod.status.init_container_statuses}
    statuses.update({status.name: status for status in pod.status.container_statuses})
    return statuses


async def process_container(pod, container_name):
    status = container_statuses(pod)[container_name]

    if status.state is None:
        log.error(f'trying to process a container status without a state: {status}')
        return {'exit_code': None, 'log': None, 'duration': None}
    if status.state.terminated is None:
        log.error(f'trying to process a non-terminated container: {status}')
        return {'exit_code': None, 'log': None, 'duration': None}

    state_terminated = status.state.terminated
    ec = state_terminated.exit_code

    pod_log, err = await k8s.read_pod_log(pod_name, container=container_name)
    if err:
        traceback.print_tb(err.__traceback__)
        log.info(f'no logs for {pod_name} due to previous error '
                 f'Error: {err}')

    if state_terminated.finished_at is not None and state_terminated.started_at is not None:
        duration = (state_terminated.finished_at - state_terminated.started_at).total_seconds()
    else:
        log.warning(
            f'{container_name} container is terminated but has no timing '
            f'information. {status}')
        duration = None

    result = {'exit_code': ec,
              'log': pod_log,
              'duration': duration}

    return result


async def process_pod(pod, failed=False, failure_reason=None):
    start = time.time()
    setup = await process_container(pod, 'setup')

    if failed:
        main = {'exit_code': 999,  # FIXME hack
                'log': failure_reason,
                'duration': None}
    else:
        assert pod is not None
        assert pod.metadata.name == pod_name
        main = await process_container(pod, 'main')

    ec = 0
    pod_log = None
    if main['exit_code'] == 0 and copy_output_cmd is not None:
        try:
            pod_log = sp.check_output(copy_output_cmd, shell=True, stderr=sp.STDOUT)
            pod_log = pod_log.decode('ascii')
            ec = 0
        except sp.CalledProcessError as e:
            pod_log = str(e) + '\n' + e.output
            ec = e.returncode

    pod_status, err = await k8s.read_pod_status(pod_name, pretty=True)
    if err is None:
        log_store.write_gs_file(
            directory,
            LogStore.pod_status_file_name,
            JSON_ENCODER.encode(pod_status.to_dict()))

    cleanup = {'exit_code': ec,
               'log': pod_log,
               'duration': round(time.time() - start)}

    exit_codes = [result['exit_code'] for result in (setup, main, cleanup)]
    durations = [result['duration'] for result in (setup, main, cleanup)]
    log_store.write_gs_file(
        directory,
        LogStore.results_file_name,
        json.dumps({'exit_codes': exit_codes, 'durations': durations}))

    log_store.write_gs_file(
        directory,
        LogStore.log_file_name,
        json.dumps({'setup': setup['log'],
                    'main': main['log'],
                    'cleanup': cleanup['log']}))

    sys.exit(0)


async def pod_changed(pod):
    if not pod:
        return

    assert pod.metadata.name == pod_name

    if pod.status and pod.status.container_statuses:
        main_status = container_statuses(pod)['main']
        if main_status.state:
            if main_status.state.terminated:
                await process_pod(pod)
            elif (main_status.state.waiting
                  and main_status.state.waiting.reason == 'ImagePullBackOff'):
                await process_pod(None,
                                  failed=True,
                                  failure_reason=main_status.state.waiting.reason)


async def kube_event_loop(pool):
    while True:
        try:
            stream = kube.watch.Watch().stream(
                v1.list_namespaced_pod,
                field_selector=f'metadata.name={pod_name}',
                namespace=HAIL_POD_NAMESPACE)
            async for event in DeblockedIterator(pool, stream):
                type = event['type']
                pod = event['object']
                name = pod.metadata.name
                log.info(f'event {type} named {name}')
                await pod_changed(pod)
        except Exception as exc:  # pylint: disable=W0703
            log.exception(f'k8s event stream failed due to: {exc}')
        await asyncio.sleep(5)


async def refresh_k8s_pods():
    await asyncio.sleep(1)
    while True:
        pods, err = await k8s.get_pod(
            namespace=HAIL_POD_NAMESPACE,
            name=pod_name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            log.info(f'could not refresh pods due to {err}, will try again later')
            return

        for pod in pods.items:
            name = pod['metadata']['name']
            log.info(f'refreshing from pod {name}')
            await pod_changed(pod)
        await asyncio.sleep(REFRESH_INTERVAL_IN_SECONDS)


class DeblockedIterator:
    def __init__(self, pool, it):
        self.pool = pool
        self.it = it

    def __aiter__(self):
        return self

    def __anext__(self):
        return blocking_to_async(self.pool, self.it.__next__)


if __name__ == '__main__':
    pool = concurrent.futures.ThreadPoolExecutor()
    loop = asyncio.get_event_loop()
    try:
        asyncio.ensure_future(kube_event_loop(pool))
        asyncio.ensure_future(refresh_k8s_pods())
        k8s = K8s(pool, KUBERNETES_TIMEOUT_IN_SECONDS, HAIL_POD_NAMESPACE, v1)
        loop.run_forever()
    finally:
        loop.close()
        pool.shutdown()
