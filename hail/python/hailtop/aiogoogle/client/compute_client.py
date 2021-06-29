import copy
import uuid
import asyncio
from typing import Mapping, Any, Optional, MutableMapping, List, Dict
import logging

from .base_client import BaseClient
from hailtop.utils import sleep_and_backoff, retry_transient_errors

log = logging.getLogger('compute_client')


class GCPError:
    def __init__(self, code, message):
        self.code = code
        self.message = message


class GCPOperationError(Exception):
    def __init__(self, status: int, message: str, errors: Optional[List[GCPError]], response: Dict[str, Any]):
        super().__init__(message)
        self.status = status
        self.message = message
        self.errors = errors or []
        self.error_codes = [e.code for e in errors]
        self.error_messages = [e.message for e in errors]
        self.response = response

    def __str__(self):
        return f'GCPOperationError: {self.status}:{self.message} {self.error_codes} {self.error_messages}; {self.response}'


class PagedIterator:
    def __init__(self, client: 'ComputeClient', path: str, request_params: Optional[Mapping[str, Any]], request_kwargs: Mapping[str, Any]):
        assert 'params' not in request_kwargs
        self._client = client
        self._path = path
        if request_params is None:
            request_params = {}
        self._request_params = request_params
        self._request_kwargs = request_kwargs
        self._page = None
        self._index = None

    def __aiter__(self) -> 'PagedIterator':
        return self

    async def __anext__(self):
        if self._page is None:
            assert 'pageToken' not in self._request_params
            self._page = await self._client.get(self._path, params=self._request_params, **self._request_kwargs)
            self._index = 0

        while True:
            if 'items' in self._page and self._index < len(self._page['items']):
                i = self._index
                self._index += 1
                return self._page['items'][i]

            next_page_token = self._page.get('nextPageToken')
            if next_page_token is not None:
                self._request_params['pageToken'] = next_page_token
                self._page = await self._client.get(self._path, params=self._request_params, **self._request_kwargs)
                self._index = 0
            else:
                raise StopAsyncIteration


class ComputeClient(BaseClient):
    def __init__(self, project, **kwargs):
        super().__init__(f'https://compute.googleapis.com/compute/v1/projects/{project}', **kwargs)

    # docs:
    # https://cloud.google.com/compute/docs/api/how-tos/api-requests-responses#handling_api_responses
    # https://cloud.google.com/compute/docs/reference/rest/v1
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/insert
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/get
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/delete
    # https://cloud.google.com/compute/docs/reference/rest/v1/disks

    async def list(self, path: str, *, params: MutableMapping[str, Any] = None, **kwargs) -> PagedIterator:
        return PagedIterator(self, path, params, kwargs)

    async def create_disk(self, path: str, *, params: MutableMapping[str, Any] = None, **kwargs):
        return await self._request_with_zonal_operations_response(self.post, path, params, **kwargs)

    async def attach_disk(self, path: str, *, params: MutableMapping[str, Any] = None, **kwargs):
        return await self._request_with_zonal_operations_response(self.post, path, params, **kwargs)

    async def detach_disk(self, path: str, *, params: MutableMapping[str, Any] = None, **kwargs):
        return await self._request_with_zonal_operations_response(self.post, path, params, **kwargs)

    async def delete_disk(self, path: str, *, params: MutableMapping[str, Any] = None, **kwargs):
        return await self.delete(path, params=params, **kwargs)

    async def _request_with_zonal_operations_response(self, request_f, path, params: MutableMapping[str, Any] = None, **kwargs):
        assert 'params' not in kwargs

        async def request_and_wait():
            local_params = copy.deepcopy(params)

            if local_params is None:
                local_params = {}

            request_uuid = str(uuid.uuid4())
            if 'requestId' not in local_params:
                local_params['requestId'] = request_uuid

            resp = await request_f(path, params=local_params, **kwargs)
            log.info(f'resp {resp}')
            operation_id = resp['id']
            zone = resp['zone'].rsplit('/', 1)[1]

            while True:
                result = await self.post(f'/zones/{zone}/operations/{operation_id}/wait')
                log.info(f'result {result}')
                if result['status'] == 'DONE':
                    http_error_status_code = result.get('httpErrorStatusCode')
                    http_error_message = result.get('httpErrorMessage')
                    errors = None

                    error = result.get('error')
                    if error:
                        errors = error['errors']
                        errors = [GCPError(e['code'], e['message']) for e in errors]

                    if http_error_status_code is not None:
                        raise GCPOperationError(http_error_status_code, http_error_message, errors, result)
                    return result

                await asyncio.sleep(1)

        await retry_transient_errors(request_and_wait)
