from typing import Any, Dict, List, Optional

import botocore.exceptions

from .base_client import AwsBaseClient


class AwsIamClient(AwsBaseClient):
    def __init__(self, **kwargs):
        super().__init__('iam', **kwargs)

    # https://docs.aws.amazon.com/IAM/latest/APIReference/API_Operations.html

    @staticmethod
    def is_no_such_entity(e: botocore.exceptions.ClientError) -> bool:
        return e.response.get('Error', {}).get('Code') == 'NoSuchEntity'

    async def get_user(self, user_name: str) -> Dict[str, Any]:
        resp = await self._call(self._client.get_user, UserName=self._normalize_user_identifier(user_name))
        return resp['User']

    async def create_user(
        self, user_name: str, *, path: str = '/', tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {'UserName': self._normalize_user_identifier(user_name), 'Path': path}
        if tags:
            kwargs['Tags'] = [{'Key': k, 'Value': v} for k, v in tags.items()]
        resp = await self._call(self._client.create_user, **kwargs)
        return resp['User']

    async def delete_user(self, user_name: str) -> None:
        await self._call(self._client.delete_user, UserName=self._normalize_user_identifier(user_name))

    async def create_access_key(self, user_name: str) -> Dict[str, Any]:
        resp = await self._call(self._client.create_access_key, UserName=self._normalize_user_identifier(user_name))
        return resp['AccessKey']

    async def list_access_key_ids(self, user_name: str) -> List[str]:
        key_ids: List[str] = []
        kwargs: Dict[str, Any] = {'UserName': self._normalize_user_identifier(user_name)}
        while True:
            resp = await self._call(self._client.list_access_keys, **kwargs)
            key_ids.extend(k['AccessKeyId'] for k in resp['AccessKeyMetadata'])
            if not resp.get('IsTruncated'):
                return key_ids
            kwargs['Marker'] = resp['Marker']

    async def delete_access_key(self, user_name: str, access_key_id: str) -> None:
        await self._call(
            self._client.delete_access_key,
            UserName=self._normalize_user_identifier(user_name),
            AccessKeyId=access_key_id,
        )

    @staticmethod
    def _normalize_user_identifier(user_identifier: str) -> str:
        if user_identifier.startswith('arn:aws'):
            return user_identifier.split('/')[-1]
        return user_identifier
