from typing import Optional

from .query_v1 import parse_batch_jobs_query_v1
from .query_v2 import parse_batch_jobs_query_v2

CURRENT_QUERY_VERSION = 1


def build_batch_jobs_query(batch_id: int, version: int, q: str, last_job_id: Optional[int]):
    if version == 1:
        return parse_batch_jobs_query_v1(batch_id, q, last_job_id)
    assert version == 2, version
    return parse_batch_jobs_query_v2(batch_id, q, last_job_id)
