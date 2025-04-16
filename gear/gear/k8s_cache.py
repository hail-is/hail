import os
from typing import Tuple

from hailtop.utils import retry_transient_errors

from .time_limited_max_size_cache import TimeLimitedMaxSizeCache

FIVE_SECONDS_NS = 5 * 1000 * 1000 * 1000


class K8sCache:
    def __init__(self, client, refresh_time_ns: int = FIVE_SECONDS_NS, max_size: int = 100):
        self.client = client
        self.secret_cache = TimeLimitedMaxSizeCache(
            self._get_secret_from_k8s, refresh_time_ns, max_size, 'K8s secret cache'
        )
        self.service_account_cache = TimeLimitedMaxSizeCache(
            self._get_service_account_from_k8s, refresh_time_ns, max_size, 'K8s service account cache'
        )
        self.k8s_timeout = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))

    async def _get_secret_from_k8s(self, name_and_namespace: Tuple[str, str]):
        return await retry_transient_errors(
            self.client.read_namespaced_secret, *name_and_namespace, _request_timeout=self.k8s_timeout
        )

    async def _get_service_account_from_k8s(self, name_and_namespace: Tuple[str, str]):
        return await retry_transient_errors(
            self.client.read_namespaced_service_account, *name_and_namespace, _request_timeout=self.k8s_timeout
        )

    async def read_secret(self, name, namespace):
        return await self.secret_cache.lookup((name, namespace))

    async def read_service_account(self, name, namespace):
        return await self.service_account_cache.lookup((name, namespace))
