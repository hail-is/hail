import argparse
import asyncio
import concurrent.futures
import sys
import traceback

import uvloop
from kubernetes_asyncio import client, config

uvloop.install()


async def timeout(timeout_seconds):
    print('info: in timeout', file=sys.stderr)
    await asyncio.sleep(timeout_seconds)
    print('error: timed out', file=sys.stderr)
    sys.exit(1)


async def wait_for_pod_complete(v1, namespace, name):
    print('info: in wait_for_pod_complete', file=sys.stderr)
    while True:
        try:
            try:
                pod = await v1.read_namespaced_pod(name, namespace, _request_timeout=5.0)
                if pod and pod.status and pod.status.container_statuses:
                    container_statuses = pod.status.container_statuses
                    if all(cs.state and cs.state.terminated for cs in container_statuses):
                        if all(cs.state.terminated.exit_code == 0 for cs in container_statuses):
                            print('info: success')
                            sys.exit(0)
                        else:
                            print('error: a container failed')
                            sys.exit(1)
            except client.ApiException as exc:
                if exc.status == 404:
                    print('info: 404', file=sys.stderr)
                else:
                    raise
        except concurrent.futures.CancelledError:
            print('info: CancelledError', file=sys.stderr)
            raise
        except Exception as e:
            print(f'wait_for_pod_complete failed due to exception {traceback.format_exc()}{e}', file=sys.stderr)

        await asyncio.sleep(1)


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('timeout_seconds', type=int)
    parser.add_argument('namespace', type=str)

    subparsers = parser.add_subparsers(dest='kind')

    pod_parser = subparsers.add_parser('Pod')
    pod_parser.add_argument('name', type=str)

    args = parser.parse_args()

    assert args.kind == 'Pod'
    await config.load_kube_config()
    v1 = client.CoreV1Api()
    try:
        t = wait_for_pod_complete(v1, args.namespace, args.name)

        await asyncio.gather(timeout(args.timeout_seconds), t)
    finally:
        await v1.api_client.rest_client.pool_manager.close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
