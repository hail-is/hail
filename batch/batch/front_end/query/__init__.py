from .query_v1 import parse_batch_jobs_query_v1, parse_list_batches_query_v1
from .query_v2 import parse_batch_jobs_query_v2, parse_list_batches_query_v2

CURRENT_QUERY_VERSION = 2

__all__ = [
    'CURRENT_QUERY_VERSION',
    'parse_batch_jobs_query_v1',
    'parse_batch_jobs_query_v2',
    'parse_list_batches_query_v1',
    'parse_list_batches_query_v2',
]
