import kubernetes as kube

from .blocking_to_async import blocking_to_async


class K8s:
    def __init__(self, blocking_pool, timeout, namespace, k8s_api, log):
        self.blocking_pool = blocking_pool
        self.timeout = timeout
        self.namespace = namespace
        self.log = log
        self._delete_pod = self._wrap_k8s_delete(k8s_api.delete_namespaced_pod)
        self._delete_pvc = self._wrap_k8s_delete(k8s_api.delete_namespaced_persistent_volume_claim)
        self._create_pod = self._wrap_k8s(k8s_api.create_namespaced_pod)
        self._create_pvc = self._wrap_k8s(k8s_api.create_namespaced_persistent_volume_claim)
        self._read_pod_log = self._wrap_k8s(k8s_api.read_namespaced_pod_log)
        self._list_pods = self._wrap_k8s(k8s_api.list_namespaced_pod)
        self._list_pvcs = self._wrap_k8s(k8s_api.list_namespaced_persistent_volume_claim)

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

    async def list_pods(self, *args, **kwargs):
        return await self._list_pods(*args, **kwargs)

    async def list_pvcs(self, *args, **kwargs):
        return await self._list_pvcs(*args, **kwargs)

    def _wrap_k8s(self, fun):
        async def wrapped(*args, **kwargs):
            try:
                if '_request_timeout' not in kwargs:
                    kwargs['_request_timeout'] = self.timeout
                if 'namespace' not in kwargs:
                    kwargs['namespace'] = self.namespace
                return (await blocking_to_async(self.blocking_pool,
                                                fun,
                                                *args,
                                                **kwargs),
                        None)
            except kube.client.rest.ApiException as err:
                return (None, err)
        wrapped.__name__ = fun.__name__
        return wrapped

    def _wrap_k8s_delete(self, fun):
        k8s_fun = self._wrap_k8s(fun)

        async def wrapped(*args, **kwargs):
            _, err = await k8s_fun(args, kwargs)
            if err is None or err.status == 404:
                self.log.debug(f'ignore already deleted {fun.__name__}(*{args}, **{kwargs})')
                return None
            return err
        wrapped.__name__ = fun.__name__
        return wrapped
