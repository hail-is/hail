from typing import Optional

from ..credentials import AzureCredentials
from ..session import AzureSession
from .base_client import AzureBaseClient


class AzureNetworkClient(AzureBaseClient):
    def __init__(self, subscription_id, resource_group_name, session: Optional[AzureSession] = None,
                 credentials: Optional[AzureCredentials] = None):
        session = session or AzureSession(credentials=credentials)
        super().__init__(f'https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.Network',
                         session=session)

    async def get_nic(self, nic_name):
        params = {
            'api-version': '2021-03-01'
        }
        return await self.get(f'/networkInterfaces/{nic_name}', params=params)

    async def delete_nic(self, nic_name):
        params = {
            'api-version': '2021-03-01',
        }
        return await self.delete(f'/networkInterfaces/{nic_name}', params=params)

    async def delete_nic_and_wait(self, nic_name):
        params = {
            'api-version': '2021-03-01',
        }
        return await self.delete_and_wait(f'/networkInterfaces/{nic_name}', params=params)

    async def get_public_ip(self, ip_name):
        params = {
            'api-version': '2021-03-01',
        }
        return await self.get(f'/publicIPAddresses/{ip_name}', params=params)

    async def delete_public_ip(self, ip_name):
        params = {
            'api-version': '2021-03-01',
        }
        return await self.delete(f'/publicIPAddresses/{ip_name}', params=params)

    async def delete_public_ip_and_wait(self, ip_name):
        params = {
            'api-version': '2021-03-01',
        }
        return await self.delete_and_wait(f'/publicIPAddresses/{ip_name}', params=params)

    async def get_network_security_group(self, nsg_name):
        params = {
            'api-version': '2021-03-01',
        }
        return await self.get(f'/networkSecurityGroups/{nsg_name}', params=params)

    async def delete_network_security_group(self, nsg_name):
        params = {
            'api-version': '2021-03-01',
        }
        return await self.delete(f'/networkSecurityGroups/{nsg_name}', params=params)

    async def delete_network_security_group_and_wait(self, nsg_name):
        params = {
            'api-version': '2021-03-01',
        }
        return await self.delete_and_wait(f'/networkSecurityGroups/{nsg_name}', params=params)
