import os
import sys
import traceback
import asyncio
import concurrent.futures
import uvloop
from kubernetes_asyncio import client, config

namespace = sys.argv[1]
name = sys.argv[2]

print(f'info: wait for {namespace}:{name}', file=sys.stderr)

uvloop.install()


async def timeout():
    print('info: in timeout', file=sys.stderr)
    await asyncio.sleep(300)
    print('error: timed out', file=sys.stderr)
    sys.exit(1)


async def poll():
    print('info: in poll', file=sys.stderr)
    if 'USE_KUBE_CONFIG' in os.environ:
        await config.load_kube_config()
    else:
        config.load_incluster_config()
    v1 = client.CoreV1Api()
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
                    pass
                else:
                    raise
        except concurrent.futures.CancelledError:
            print('info: CancelledError', file=sys.stderr)
            raise
        except Exception as e:
            print(f'poll failed due to exception {traceback.format_exc()}{e}', file=sys.stderr)

        await asyncio.sleep(1)

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(timeout(), poll()))
