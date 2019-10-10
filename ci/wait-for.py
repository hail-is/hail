import os
import sys
import traceback
import argparse
import concurrent.futures
import asyncio
import aiohttp
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
                pod = await v1.read_namespaced_pod(
                    name,
                    namespace,
                    _request_timeout=5.0)
                if pod and pod.status and pod.status.container_statuses:
                    container_statuses = pod.status.container_statuses
                    if all(cs.state and cs.state.terminated for cs in container_statuses):
                        if all(cs.state.terminated.exit_code == 0 for cs in container_statuses):
                            print('info: success')
                            sys.exit(0)
                        else:
                            print('error: a container failed')
                            sys.exit(1)
            except client.rest.ApiException as exc:
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


# this needs to agree with hailtop.config
def internal_base_url(namespace, service, port):
    if namespace == 'default':
        return f'http://{service}.default:{port}'
    return f'http://{service}.{namespace}:{port}/{namespace}/{service}'


async def wait_for_service_alive(namespace, name, port):
    print('info: in wait_for_service_alive', file=sys.stderr)
    base_url = internal_base_url(namespace, name, port)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
        while True:
            try:
                async with session.get(f'{base_url}/healthcheck') as resp:
                    if resp.status >= 200 and resp.status < 300:
                        print('info: success')
                        sys.exit(0)
            except concurrent.futures.CancelledError:
                print('info: CancelledError', file=sys.stderr)
                raise
            except Exception as e:
                print(f'wait_for_service_alive failed due to exception {traceback.format_exc()}{e}', file=sys.stderr)

            await asyncio.sleep(1)


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('timeout_seconds', type=int)
    parser.add_argument('namespace', type=str)

    subparsers = parser.add_subparsers(dest='kind')

    pod_parser = subparsers.add_parser('Pod')
    pod_parser.add_argument('name', type=str)

    service_parser = subparsers.add_parser('Service')
    service_parser.add_argument('name', type=str)
    service_parser.add_argument('--port', '-p', type=int, default=80)

    args = parser.parse_args()

    if args.kind == 'Pod':
        if 'USE_KUBE_CONFIG' in os.environ:
            await config.load_kube_config()
        else:
            config.load_incluster_config()
        v1 = client.CoreV1Api()

        t = wait_for_pod_complete(v1, args.namespace, args.name)
    else:
        assert args.kind == 'Service'
        t = wait_for_service_alive(args.namespace, args.name, args.port)

    await asyncio.gather(timeout(args.timeout_seconds), t)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
