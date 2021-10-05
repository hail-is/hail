from typing import Optional

from ..credentials import AzureCredentials
from ..session import AzureSession
from .base_client import AzureBaseClient


class AzureComputeClient(AzureBaseClient):
    def __init__(self, subscription_id, resource_group_name, session: Optional[AzureSession] = None,
                 credentials: Optional[AzureCredentials] = None):
        session = session or AzureSession(credentials=credentials)
        super().__init__(f'https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.Compute',
                         session=session)

    async def get_vm_instance_view(self, vm_name):
        params = {
            'api-version': '2021-07-01',
        }
        return await self.get(f'/virtualMachines/{vm_name}/instanceView', params=params)

    async def get_vm(self, vm_name):
        params = {
            'api-version': '2021-07-01',
        }
        return await self.get(f'/virtualMachines/{vm_name}', params=params)

    async def delete_vm(self, vm_name):
        params = {
            'api-version': '2021-07-01',
        }
        return await self.delete(f'/virtualMachines/{vm_name}', params=params)

    async def delete_vm_and_wait(self, vm_name):
        params = {
            'api-version': '2021-07-01',
        }
        return await self.delete_and_wait(f'/virtualMachines/{vm_name}', params=params)

    async def get_disk(self, disk_name):
        params = {
            'api-version': '2020-12-01',
        }
        return await self.get(f'/disks/{disk_name}', params=params)

    async def delete_disk(self, disk_name):
        params = {
            'api-version': '2020-12-01',
        }
        return await self.delete(f'/disks/{disk_name}', params=params)

    async def delete_disk_and_wait(self, disk_name):
        params = {
            'api-version': '2020-12-01',
        }
        return await self.delete_and_wait(f'/disks/{disk_name}', params=params)
