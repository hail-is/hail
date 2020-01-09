import time
import asyncio
import logging

log = logging.getLogger('k8s_cache')


class K8sCache:
    def __init__(self, client, refresh_time):
        self.client = client
        self.refresh_time = refresh_time
        self.secrets = {}
        self.service_accounts = {}

    async def async_init(self):
        asyncio.ensure_future(self.cleanup_loop())

    async def read_secret(self, name, namespace, timeout):
        secret, time_updated = self.secrets.get((name, namespace), (None, None))
        if time_updated and time.time() < time_updated + self.refresh_time:
            return secret

        secret = await self.client.read_namespaced_secret(
            name, namespace, _request_timeout=timeout)
        self.secrets[(name, namespace)] = (secret, time.time())
        return secret

    async def read_service_account(self, name, namespace, timeout):
        sa, time_updated = self.service_accounts.get((name, namespace), (None, None))
        if time_updated and time.time() < time_updated + self.refresh_time:
            return sa

        sa = await self.client.read_namespaced_service_account(
            name, namespace, _request_timeout=timeout)
        self.service_accounts[(name, namespace)] = (sa, time.time())
        return sa

    async def cleanup_loop(self):
        while True:
            await asyncio.sleep(300)

            try:
                secrets_to_delete = [id for id, (_, time_updated) in self.secrets.items()
                                     if time.time() >= time_updated + self.refresh_time]
                for id in secrets_to_delete:
                    del self.secrets[id]

                sas_to_delete = [(id, time_updated) for id, (_, time_updated) in self.service_accounts.items()
                                 if time.time() >= time_updated + self.refresh_time]
                for id in sas_to_delete:
                    del self.service_accounts[id]
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except:  # pylint: disable=bare-except
                log.exception(f'error while cleaning up the k8s cache')
