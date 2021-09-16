import abc

import logging

log = logging.getLogger(__name__)


class Credentials(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def from_file(credentials_file):
        pass

    @staticmethod
    @abc.abstractmethod
    def default_credentials():
        pass

    @abc.abstractmethod
    async def get_access_token(self, session):
        pass

    @abc.abstractmethod
    async def close(self):
        pass


class AccessToken(abc.ABC):
    def __init__(self, credentials: 'Credentials'):
        self.credentials = credentials
        self._access_token = None
        self._expires_at = None

    @abc.abstractmethod
    async def auth_headers(self, session):
        pass
