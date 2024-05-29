import base64
from typing import Optional

import orjson

from hailtop.httpx import ClientResponseError

from .base_client import GoogleBaseClient


class GoogleSecretManagerClient(GoogleBaseClient):
    def __init__(self, project, **kwargs):
        super().__init__(f'https://secretmanager.googleapis.com/v1/projects/{project}', **kwargs)

    # https://cloud.google.com/secret-manager/docs/reference/rest

    async def create_secret(self, secret_id: str, expiration_seconds: Optional[int] = None):
        json: dict = {'replication': {'automatic': {}}}
        if expiration_seconds:
            json['ttl'] = f'{expiration_seconds}s'
        await self.post('/secrets', params={'secret_id': secret_id}, json=json)

    async def create_secret_if_not_exists(self, secret_id: str, expiration_seconds: Optional[int] = None):
        try:
            await self.create_secret(secret_id, expiration_seconds)
        except ClientResponseError as e:
            if not e.status == 409 or orjson.loads(e.body)['error'].get('status') != 'ALREADY_EXISTS':
                raise

    async def create_secret_version(self, secret_id: str, secret_data: bytes):
        encoded_data = base64.b64encode(secret_data).decode('utf-8')
        await self.post(f'/secrets/{secret_id}:addVersion', json={'payload': {'data': encoded_data}})

    async def get_latest_secret_version(self, secret_id: str) -> bytes:
        resp = await self.get(f'/secrets/{secret_id}/versions/latest:access')
        return base64.b64decode(resp['payload']['data'])
