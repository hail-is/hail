import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, MutableMapping, Optional

import aiohttp

from hailtop.utils import retry_transient_errors, sleep_before_try

from .base_client import GoogleBaseClient

log = logging.getLogger('compute_client')


class GCPOperationError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        error_codes: Optional[List[str]],
        error_messages: Optional[List[str]],
        response: Dict[str, Any],
    ):
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_codes = error_codes
        self.error_messages = error_messages
        self.response = response

    def __str__(self):
        return (
            f'GCPOperationError: {self.status}:{self.message} {self.error_codes} {self.error_messages}; {self.response}'
        )


class GoogleComputeClient(GoogleBaseClient):
    def __init__(self, project, **kwargs):
        super().__init__(f'https://compute.googleapis.com/compute/v1/projects/{project}', **kwargs)

    # docs:
    # https://cloud.google.com/compute/docs/api/how-tos/api-requests-responses#handling_api_responses
    # https://cloud.google.com/compute/docs/reference/rest/v1
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/insert
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/get
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/delete
    # https://cloud.google.com/compute/docs/reference/rest/v1/disks

    async def list(
        self, path: str, *, params: Optional[MutableMapping[str, Any]] = None, **kwargs
    ) -> AsyncIterator[dict]:
        # Don't mutate the caller's params when we add the nextPageToken
        params = dict(params) if params is not None else {}
        first_page = True
        next = None
        while first_page or next is not None:
            page = await self.get(path, params=params, **kwargs)
            for item in page.get('items', []):
                yield item
            next = page.get('nextPageToken')
            params['pageToken'] = next
            first_page = False

    async def create_disk(self, path: str, *, params: Optional[MutableMapping[str, Any]] = None, **kwargs):
        return await self._request_with_zonal_operations_response(self.post, path, params, **kwargs)

    async def attach_disk(self, path: str, *, params: Optional[MutableMapping[str, Any]] = None, **kwargs):
        return await self._request_with_zonal_operations_response(self.post, path, params, **kwargs)

    async def detach_disk(self, path: str, *, params: Optional[MutableMapping[str, Any]] = None, **kwargs):
        return await self._request_with_zonal_operations_response(self.post, path, params, **kwargs)

    async def delete_disk(self, path: str, *, params: Optional[MutableMapping[str, Any]] = None, **kwargs):
        return await self.delete(path, params=params, **kwargs)

    async def _request_with_zonal_operations_response(
        self, request_f, path, maybe_params: Optional[MutableMapping[str, Any]] = None, **kwargs
    ):
        params = maybe_params or {}
        assert 'requestId' not in params

        async def request_and_wait():
            params['requestId'] = str(uuid.uuid4())

            resp = await request_f(path, params=params, **kwargs)

            operation_id = resp['id']
            zone = resp['zone'].rsplit('/', 1)[1]

            tries = 0
            while True:
                result = await self.post(
                    f'/zones/{zone}/operations/{operation_id}/wait', timeout=aiohttp.ClientTimeout(total=150)
                )
                if result['status'] == 'DONE':
                    error = result.get('error')
                    if error:
                        assert result.get('httpErrorStatusCode') is not None
                        assert result.get('httpErrorMessage') is not None

                        error_codes = [e['code'] for e in error['errors']]
                        error_messages = [e['message'] for e in error['errors']]

                        raise GCPOperationError(
                            result['httpErrorStatusCode'],
                            result['httpErrorMessage'],
                            error_codes,
                            error_messages,
                            result,
                        )

                    return result
                tries += 1
                await sleep_before_try(tries, base_delay_ms=2_000, max_delay_ms=15_000)

        return await retry_transient_errors(request_and_wait)
