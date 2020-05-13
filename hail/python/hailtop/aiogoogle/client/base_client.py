import hailtop.aiogoogle.auth as google_auth


class BaseClient:
    def __init__(self, base_url, *, session=None, **kwargs):
        self._base_url = base_url
        if session is None:
            session = google_auth.Session(**kwargs)
        self._session = session

    async def get(self, path, **kwargs):
        async with await self._session.get(
                f'{self._base_url}{path}', **kwargs) as resp:
            return await resp.json()

    async def post(self, path, **kwargs):
        async with await self._session.post(
                f'{self._base_url}{path}', **kwargs) as resp:
            return await resp.json()

    async def delete(self, path, **kwargs):
        async with await self._session.delete(
                f'{self._base_url}{path}', **kwargs):
            pass

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
