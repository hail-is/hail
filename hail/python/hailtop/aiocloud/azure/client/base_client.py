from typing import Optional

from hailtop.utils import RateLimit

from ...common import BaseClient
from ..session import AzureSession


class AzureBaseClient(BaseClient):
    _session: AzureSession

    def __init__(self, base_url: str, *, session: Optional[AzureSession] = None,
                 rate_limit: RateLimit = None, **kwargs):
        if session is None:
            session = AzureSession(**kwargs)
        super().__init__(base_url, session, rate_limit=rate_limit)
