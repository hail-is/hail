from typing import Optional

from hailtop.utils import RateLimit

from ...common import CloudBaseClient
from ..session import GoogleSession


class GoogleBaseClient(CloudBaseClient):
    def __init__(self, base_url: str, *, session: Optional[GoogleSession] = None,
                 rate_limit: Optional[RateLimit] = None, **kwargs):
        if session is None:
            session = GoogleSession(**kwargs)
        super().__init__(base_url, session, rate_limit=rate_limit)
