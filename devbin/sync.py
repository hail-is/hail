#!/usr/bin/env python3
import argparse
import asyncio
import concurrent.futures
import logging
import os
import re
import signal
import sys
from contextlib import closing
from threading import Thread
from typing import List, Set, Tuple

import kubernetes_asyncio.client
import kubernetes_asyncio.config
from fswatch import Monitor, libfswatch

from hailtop.aiotools import BackgroundTaskManager
from hailtop.hail_logging import configure_logging
from hailtop.utils import CalledProcessError, check_shell, retry_transient_errors

configure_logging()
log = logging.getLogger('sync.py')
RSYNC_ARGS = "-av --progress --stats --exclude='*.log' --exclude='.mypy_cache' --exclude='__pycache__' --exclude='*~' --exclude='flycheck_*' --exclude='.#*'"


DEVBIN = os.path.abspath(os.path.dirname(__file__))


class Sync:
    def __init__(self, paths: List[Tuple[str, str]]):
        self.pods: Set[Tuple[str, str]] = set()
        self.paths = paths
        self.should_sync_event = asyncio.Event()
        self.update_loop_task = asyncio.create_task(self.update_loop())

    async def close(self):
        await cancel_and_retrieve_all_exceptions([self.update_loop_task])

    async def sync_and_restart_pod(self, pod, namespace):
        log.info(f'reloading {pod}@{namespace}')
        try:
            await asyncio.gather(
                *[
                    check_shell(f'{DEVBIN}/krsync.sh {RSYNC_ARGS} {local} {pod}@{namespace}:{remote}')
                    for local, remote in self.paths
                ]
            )
            await check_shell(f'kubectl exec {pod} --namespace {namespace} -- kill -2 1')
        except CalledProcessError:
            log.warning(f'could not synchronize {namespace}/{pod}, removing from active pods', exc_info=True)
            self.pods.remove((pod, namespace))
            return
        log.info(f'reloaded {pod}@{namespace}')

    async def initialize_pod(self, pod, namespace):
        log.info(f'initializing {pod}@{namespace}')
        try:
            await asyncio.gather(
                *[
                    check_shell(f'{DEVBIN}/krsync.sh {RSYNC_ARGS} {local} {pod}@{namespace}:{remote}')
                    for local, remote in self.paths
                ]
            )
            await check_shell(f'kubectl exec {pod} --namespace {namespace} -- kill -2 1')
        except CalledProcessError:
            log.warning(f'could not initialize {namespace}/{pod}', exc_info=True)
            return
        self.pods.add((pod, namespace))
        log.info(f'initialized {pod}@{namespace}')

    async def monitor_pods(self, apps, namespace):
        await kubernetes_asyncio.config.load_kube_config()
        k8s = kubernetes_asyncio.client.CoreV1Api()
        try:
            while True:
                log.info('monitor_pods: start loop')
                updated_pods = await retry_transient_errors(
                    k8s.list_namespaced_pod, namespace, label_selector=f'app in ({",".join(apps)})'
                )
                updated_pods = [
                    x
                    for x in updated_pods.items
                    if x.status.phase == 'Running'
                    if all(s.ready for s in x.status.container_statuses)
                ]
                updated_pods = {(pod.metadata.name, namespace) for pod in updated_pods}
                fresh_pods = updated_pods - self.pods
                dead_pods = self.pods - updated_pods
                log.info(f'monitor_pods: fresh_pods: {fresh_pods}')
                log.info(f'monitor_pods: dead_pods: {dead_pods}')
                self.pods = self.pods - dead_pods
                await asyncio.gather(*[self.initialize_pod(name, namespace) for name, namespace in fresh_pods])
                await asyncio.sleep(5)
        finally:
            await k8s.api_client.rest_client.pool_manager.close()

    async def update_loop(self):
        while True:
            await self.should_sync_event.wait()
            self.should_sync_event.clear()
            await asyncio.gather(*[self.sync_and_restart_pod(pod, namespace) for pod, namespace in self.pods])

    async def should_sync(self):
        self.should_sync_event.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='sync.py',
        description='Develop locally, deploy cloudly.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--app', action='append', type=str, help='An app label to watch.')
    parser.add_argument('--namespace', required=False, type=str, help='The namespace in which to watch.')
    parser.add_argument(
        '--path',
        action='append',
        nargs='+',
        metavar=('local', 'remote'),
        help='The local path will be kept in sync with the remote path.',
    )
    parser.add_argument(
        '--ignore',
        required=False,
        type=str,
        default='flycheck_.*|.*~|\.#.*',
        help='A regular expression indicating in which files to ignore changes.',
    )

    args = parser.parse_args(sys.argv[1:])

    with closing(asyncio.get_event_loop()) as loop:
        monitor = Monitor()
        sync_futs: Set[concurrent.futures.Future] = set()
        try:
            sync = Sync(args.path)

            for local, _ in args.path:
                monitor.add_path(local)

            ignore_re = re.compile(args.ignore)

            def callback(path: bytes, evt_time, flags, flags_num, event_num):
                if not ignore_re.fullmatch(os.path.basename(path.decode())):
                    sync_futs.add(asyncio.run_coroutine_threadsafe(sync.should_sync(), loop))

            monitor.set_callback(callback)

            signal.signal(signal.SIGINT, monitor._handle_signal)
            thread = Thread(target=libfswatch.fsw_start_monitor, args=(monitor.handle,), daemon=True)
            thread.start()
            loop.run_until_complete(sync.monitor_pods(args.app, args.namespace))
        finally:
            try:
                loop.run_until_complete(sync.close())
            finally:
                try:
                    for f in sync_futs:
                        f.cancel()
                finally:
                    thread.join()
