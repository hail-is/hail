import logging

import kubernetes as kube
import prometheus_client as pc

from .blocking_to_async import blocking_to_async

log = logging.getLogger('batch.k8s')


class EmptyContextManager:
    def __enter__(self):
        return None

    def __exit__(self, type, value, callback):
        return None


class NoSummary:
    EMPTY_CONTEXT_MANAGER = EmptyContextManager()

    def time(self):  # pylint: disable=R0201
        return NoSummary.EMPTY_CONTEXT_MANAGER


class K8s:
    def __init__(self, blocking_pool, timeout, namespace, k8s_api):
        self.blocking_pool = blocking_pool
        self.timeout = timeout
        self.namespace = namespace
        self._delete_pod = self._wrap_k8s_delete(k8s_api.delete_namespaced_pod)
        self._delete_pvc = self._wrap_k8s_delete(k8s_api.delete_namespaced_persistent_volume_claim)
        self._create_pod = self._wrap_k8s(k8s_api.create_namespaced_pod,
                                          pc.Summary('batch_create_pod_seconds',
                                                     'Batch k8s create pod latency in seconds'))
        self._create_pvc = self._wrap_k8s(k8s_api.create_namespaced_persistent_volume_claim,
                                          pc.Summary('batch_create_pvc_seconds',
                                                     'Batch k8s create pvc latency in seconds'))
        self._read_pod_log = self._wrap_k8s(k8s_api.read_namespaced_pod_log)
        self._read_pod_status = self._wrap_k8s(k8s_api.read_namespaced_pod_status)
        self._list_pods = self._wrap_k8s(k8s_api.list_namespaced_pod)
        self._list_pvcs = self._wrap_k8s(k8s_api.list_namespaced_persistent_volume_claim)
        self._get_pod = self._wrap_k8s(k8s_api.read_namespaced_pod)
        self._get_pvc = self._wrap_k8s(k8s_api.read_namespaced_persistent_volume_claim)

    async def delete_pod(self, name):
        assert name is not None
        return await self._delete_pod(name=name)

    async def delete_pvc(self, name):
        assert name is not None
        return await self._delete_pvc(name=name)

    async def create_pod(self, *args, **kwargs):
        return await self._create_pod(*args, **kwargs)

    async def create_pvc(self, *args, **kwargs):
        return await self._create_pvc(*args, **kwargs)

    async def read_pod_log(self, *args, **kwargs):
        return await self._read_pod_log(*args, **kwargs)

    async def read_pod_status(self, *args, **kwargs):
        return await self._read_pod_status(*args, **kwargs)

    async def list_pods(self, *args, **kwargs):
        return await self._list_pods(*args, **kwargs)

    async def list_pvcs(self, *args, **kwargs):
        return await self._list_pvcs(*args, **kwargs)

    async def get_pod(self, *args, **kwargs):
        return await self._get_pod(*args, **kwargs)

    async def get_pvc(self, *args, **kwargs):
        return await self._get_pvc(*args, **kwargs)

    def _wrap_k8s(self, fun, pc_summary=NoSummary()):
        async def wrapped(*args, **kwargs):
            try:
                if '_request_timeout' not in kwargs:
                    kwargs['_request_timeout'] = self.timeout
                if 'namespace' not in kwargs:
                    kwargs['namespace'] = self.namespace
                with pc_summary.time():
                    return (await blocking_to_async(self.blocking_pool,
                                                    fun,
                                                    *args,
                                                    **kwargs),
                            None)
            except kube.client.rest.ApiException as err:
                return (None, err)
        wrapped.__name__ = fun.__name__
        return wrapped

    def _wrap_k8s_delete(self, fun, pc_summary=NoSummary()):
        k8s_fun = self._wrap_k8s(fun, pc_summary)

        async def wrapped(*args, **kwargs):
            _, err = await k8s_fun(*args, **kwargs)
            if err is None or err.status == 404:
                log.debug(f'ignore already deleted {fun.__name__}(*{args}, **{kwargs})')
                return None
            return err
        wrapped.__name__ = fun.__name__
        return wrapped
