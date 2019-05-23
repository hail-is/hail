from .globals import blocking_to_async


class K8s:
    def __init__(self, blocking_pool, timeout, namespace, k8s_api):
        self.blocking_pool = blocking_pool
        self.timeout = timeout
        self.namespace = namespace
        self.__delete_pod = self._wrap_k8s_delete(k8s_api.delete_namespaced_pod)
        self.__delete_pvc = self._wrap_k8s_delete(k8s_api.delete_namespaced_persistent_volume_claim)
        self.__create_pod = self._wrap_k8s(k8s_api.create_namespaced_pod)
        self.__create_pvc = self._wrap_k8s(k8s_api.create_namespaced_persistent_volume_claim)
        self.__read_pod_log = self._wrap_k8s(k8s_api.read_namespaced_pod_log)
        self.__list_pods = self._wrap_k8s(k8s_api.list_namespaced_pod)
        self.__list_pvcs = self._wrap_k8s(k8s_api.list_namespaced_persistent_volume_claim)

    def _wrap_k8s(fun):
        async def wrapped(*args, **kwargs):
            try:
                if '_request_timeout' not in kwargs:
                    kwargs['_request_timeout'] = self.timeout
                if 'namespace' not in kwargs:
                    kwargs['namespace'] = self.namespace
                return (await blocking_to_async(self.blocking_pool,
                                                f,
                                                *args,
                                                **kwargs),
                        None)
            except kube.client.rest.ApiException as err:
                return (None, err)
        wrapped.__name__ = fun.__name__
        return wrapped

    def _wrap_k8s_delete(fun):
        k8s_fun = self._wrap_k8s(fun)

        async def wrapped(*args, **kwargs):
            _, err = await k8s_fun(args, kwargs)
            if err is None or err.status is 404:
                log.debug(f'ignore already deleted {fun.__name__}(*{args}, **{kwargs})')
                return None
            return err
        wrapped.__name__ = fun.__name__
        return wrapped

    async def delete_pod(name):
        assert name is not None
        return await self.__delete_pod(name=name)

    async def delete_pvc(name):
        assert name is not None
        return await self.__delete_pvc(name=name)

    async def create_pod(*args, **kwargs):
        return await self.__create_pod(*args, **kwargs)

    async def create_pvc(*args, **kwargs):
        return await self.__create_pvc(*args, **kwargs)

    async def read_pod_log(*args, **kwargs):
        return await self.__read_pod_log(*args, **kwargs)

    async def list_pods(*args, **kwargs):
        return await self.__list_pods(*args, **kwargs)

    async def list_pvcs(*args, **kwargs):
        return await self.__list_pvcs(*args, **kwargs)
