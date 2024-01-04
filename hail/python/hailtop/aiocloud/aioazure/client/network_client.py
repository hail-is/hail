from typing import Optional

import aiohttp

from ..session import AzureSession
from .base_client import AzureBaseClient


class AzureNetworkClient(AzureBaseClient):
    def __init__(self, subscription_id, resource_group_name, session: Optional[AzureSession] = None, **kwargs):
        if 'params' not in kwargs:
            kwargs['params'] = {}
        params = kwargs['params']
        if 'api-version' not in params:
            params['api-version'] = '2021-03-01'
        session = session or AzureSession(**kwargs)
        super().__init__(
            f'https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.Network',
            session=session,
        )

    async def delete_nic(self, nic_name: str, ignore_not_found: bool = False):
        try:
            await self.delete(f'/networkInterfaces/{nic_name}')
        except aiohttp.ClientResponseError as e:
            if ignore_not_found and e.status == 404:
                pass
            raise

    async def delete_public_ip(self, public_ip_name: str, ignore_not_found: bool = False):
        try:
            await self.delete(f'/publicIPAddresses/{public_ip_name}')
        except aiohttp.ClientResponseError as e:
            if ignore_not_found and e.status == 404:
                pass
            raise
