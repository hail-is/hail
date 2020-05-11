from types import TracebackType
from typing import Any, Optional, Type, TypeVar
from hailtop.utils import RateLimit
from hailtop.aiogoogle.auth import Session, RateLimitedSession

ClientType = TypeVar('ClientType', bound='BaseClient')


class BaseClient:
    def __init__(self, base_url: str, *, session: Session = None,
                 rate_limit: RateLimit = None, **kwargs):
        self._base_url = base_url
        if session is None:
            session = Session(**kwargs)
        if rate_limit is not None:
            session = RateLimitedSession(session=session, rate_limit=rate_limit)
        self._session = session

    async def get(self, path: str, **kwargs) -> Any:
        async with await self._session.get(
                f'{self._base_url}{path}', **kwargs) as resp:
            return await resp.json()

    async def post(self, path: str, **kwargs) -> Any:
        async with await self._session.post(
                f'{self._base_url}{path}', **kwargs) as resp:
            return await resp.json()

    async def delete(self, path: str, **kwargs) -> None:
        async with await self._session.delete(
                f'{self._base_url}{path}', **kwargs):
            pass

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self: ClientType) -> ClientType:
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        await self.close()
