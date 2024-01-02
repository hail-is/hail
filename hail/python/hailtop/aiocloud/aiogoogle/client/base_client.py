from typing import Optional, Union, Mapping

from hailtop.utils import RateLimit

from ..credentials import GoogleCredentials
from ...common import CloudBaseClient
from ...common.session import BaseSession, Session
from ...common.credentials import AnonymousCloudCredentials


class GoogleBaseClient(CloudBaseClient):
    def __init__(self,
                 base_url: str,
                 *,
                 session: Optional[BaseSession] = None,
                 rate_limit: Optional[RateLimit] = None,
                 credentials: Optional[Union[GoogleCredentials, AnonymousCloudCredentials]] = None,
                 credentials_file: Optional[str] = None,
                 params: Optional[Mapping[str, str]] = None,
                 **kwargs):
        if session is None:
            session = Session(
                credentials=credentials or GoogleCredentials.from_file_or_default(credentials_file),
                params=params,
                **kwargs
            )
        elif credentials_file is not None or credentials is not None:
            raise ValueError('Do not provide credentials_file or credentials when session is None')

        super().__init__(base_url=base_url, session=session, rate_limit=rate_limit)
