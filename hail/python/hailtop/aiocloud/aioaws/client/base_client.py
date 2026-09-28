from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any, Callable, Optional, Type, TypeVar

import boto3
import botocore.config

from hailtop.utils import blocking_to_async

ClientType = TypeVar('ClientType', bound='AwsBaseClient')


class AwsBaseClient:
    def __init__(
        self,
        service_name: str,
        *,
        session: Optional[boto3.session.Session] = None,
        config: Optional[botocore.config.Config] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        max_workers: Optional[int] = None,
    ):
        if session is None:
            session = boto3.session.Session()
        self._owns_thread_pool = thread_pool is None
        if thread_pool is None:
            thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._thread_pool = thread_pool
        self._client = session.client(service_name, config=config)

    async def _call(self, fun: Callable[..., Any], **kwargs) -> Any:
        return await blocking_to_async(self._thread_pool, fun, **kwargs)

    async def close(self) -> None:
        if hasattr(self, '_client'):
            self._client.close()
            del self._client
        if self._owns_thread_pool:
            self._thread_pool.shutdown(wait=False)

    async def __aenter__(self: ClientType) -> ClientType:
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        await self.close()
