import kubernetes as kube
import prometheus_client as pc
from hailtop.utils import blocking_to_async

READ_SECRET_TIME = pc.Summary('batch2_read_secret',
                              'Batch2 k8s read secret latency in seconds')


class K8s:
    def __init__(self, thread_pool, timeout, namespace, k8s_client):
        self.thread_pool = thread_pool
        self.timeout = timeout
        self.namespace = namespace
        self.k8s_client = k8s_client

    async def read_secret(self, *args, **kwargs):
        if '_request_timeout' not in kwargs:
            kwargs['_request_timeout'] = self.timeout
        if 'namespace' not in kwargs:
            kwargs['namespace'] = self.namespace
        with READ_SECRET_TIME.time():
            try:
                v = await blocking_to_async(
                    self.thread_pool,
                    self.k8s_client.read_namespaced_secret, *args, **kwargs)
                return (v, None)
            except kube.client.rest.ApiException as err:
                return (None, err)
