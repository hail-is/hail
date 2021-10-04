from typing import Optional

from hailtop.utils import RateLimit

from ...common import CloudBaseClient
from ..session import GCPSession


class GCPBaseClient(CloudBaseClient):
    _session: GCPSession

    def __init__(self, base_url: str, *, session: Optional[GCPSession] = None,
                 rate_limit: RateLimit = None, **kwargs):
        if session is None:
            session = GCPSession(**kwargs)
        super().__init__(base_url, session, rate_limit=rate_limit)
