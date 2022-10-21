import os

from ....common import CloudBaseClient, Session
from .....auth import hail_credentials


class TerraClient(CloudBaseClient):
    def __init__(self):
        base_url = (
            f"{os.environ['WORKSPACE_MANAGER_URL']}/api/workspaces/v1/{os.environ['WORKSPACE_ID']}/resources/controlled/azure"
        )
        super().__init__(base_url, Session(credentials=hail_credentials()))

    async def get_storage_container_sas_token(self, container_resource_id: str, blob_name: str, permissions: str = 'racwdl', expires_after: int = 3600) -> str:
        headers = {'Content-Type': 'application/json'}
        params = {'sasPermissions': permissions, 'sasExpirationDuration': expires_after, 'sasBlobName': blob_name}
        resp = await self.post(
            f'/storageContainer/{container_resource_id}/getSasToken',
            headers=headers,
            params=params,
        )
        return resp['url']
