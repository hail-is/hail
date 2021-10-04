from typing import Optional

from hailtop.utils import RateLimit

from ...common import CloudBaseClient
from ..session import AzureSession


class AzureBaseClient(CloudBaseClient):
    _session: AzureSession

    def __init__(self, base_url: str, *, session: Optional[AzureSession] = None,
                 rate_limit: RateLimit = None, **kwargs):
        if session is None:
            session = AzureSession(**kwargs)
        super().__init__(base_url, session, rate_limit=rate_limit)
