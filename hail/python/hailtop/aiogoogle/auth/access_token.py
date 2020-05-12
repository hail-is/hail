import time


class AccessToken:
    def __init__(self, credentials):
        self.credentials = credentials
        self._access_token = None
        self._expires_at = None

    async def auth_headers(self, session):
        now = time.time()
        if self._access_token is None or now > self._expires_at:
            self._access_token = await self.credentials.get_access_token(session)
            self._expires_at = now + self._access_token['expires_in'] // 2
        return {'Authorization': f'Bearer {self._access_token["access_token"]}'}
