from .query_v1 import parse_batch_jobs_query_v1
from .query_v2 import parse_batch_jobs_query_v2

CURRENT_QUERY_VERSION = 1


def build_batch_jobs_query(request, batch_id, version):
    if version == 1:
        return parse_batch_jobs_query_v1(request, batch_id)
    assert version == 2, version
    return parse_batch_jobs_query_v2(request, batch_id)
