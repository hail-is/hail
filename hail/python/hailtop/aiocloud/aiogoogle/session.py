from typing import Mapping, Optional, Union

from ..common import Session, AnonymousCloudCredentials

from .credentials import GoogleCredentials


class GoogleSession(Session):
    def __init__(
        self,
        *,
        credentials: Optional[Union[GoogleCredentials, AnonymousCloudCredentials]] = None,
        credentials_file: Optional[str] = None,
        params: Optional[Mapping[str, str]] = None,
        **kwargs,
    ):
        assert (
            credentials is None or credentials_file is None
        ), f'specify only one of credentials or credentials_file: {(credentials, credentials_file)}'
        if credentials is None:
            if credentials_file:
                credentials = GoogleCredentials.from_file(credentials_file)
            else:
                credentials = GoogleCredentials.default_credentials()
        super().__init__(credentials=credentials, params=params, **kwargs)
