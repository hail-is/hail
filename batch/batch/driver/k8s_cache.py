import time
import sortedcontainers


class K8sCache:
    def __init__(self, client, refresh_time, max_size=100):
        self.client = client
        self.refresh_time = refresh_time
        self.max_size = max_size

        self.secrets_update_time = {}
        self.secrets = sortedcontainers.SortedDict(
            key=lambda id: self.secrets_update_time[id])

        self.service_accounts_update_time = {}
        self.service_accounts = sortedcontainers.SortedDict(
            key=lambda id: self.service_accounts_update_time[id])

    async def read_secret(self, name, namespace, timeout):
        id = (name, namespace)

        time_updated = self.secrets_update_time.get(id, None)
        if time_updated and time.time() < time_updated + self.refresh_time:
            secret = self.secrets[id]
            return secret

        if len(self.secrets) == self.max_size:
            self.secrets.popitem(0)

        secret = await self.client.read_namespaced_secret(
            name, namespace, _request_timeout=timeout)
        self.secrets_update_time[id] = time.time()
        self.secrets[id] = secret

        return secret

    async def read_service_account(self, name, namespace, timeout):
        id = (name, namespace)

        time_updated = self.service_accounts_update_time.get(id, None)
        if time_updated and time.time() < time_updated + self.refresh_time:
            sa = self.service_accounts[id]
            return sa

        if len(self.service_accounts) == self.max_size:
            self.service_accounts.popitem(0)

        sa = await self.client.read_namespaced_service_account(
            name, namespace, _request_timeout=timeout)
        self.service_accounts_update_time[id] = time.time()
        self.service_accounts[id] = sa

        return sa
