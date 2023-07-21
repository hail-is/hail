import collections
import logging
from typing import Dict, Literal, Tuple

import sortedcontainers

from hailtop.utils import secret_alnum_string

log = logging.getLogger('batch.spec_writer')

JOB_TOKEN_CACHE: Dict[int, sortedcontainers.SortedSet] = collections.defaultdict(
    lambda: sortedcontainers.SortedSet(key=lambda t: t[1])
)
JOB_TOKEN_CACHE_MAX_BATCHES = 100
JOB_TOKEN_CACHE_MAX_BUNCHES_PER_BATCH = 100


class SpecWriter:
    byteorder: Literal['little', 'big'] = 'little'
    signed = False
    bytes_per_offset = 8

    @staticmethod
    def get_index_file_offsets(job_id, start_job_id):
        assert job_id >= start_job_id
        idx_start = SpecWriter.bytes_per_offset * (job_id - start_job_id)
        idx_end = (
            idx_start + 2 * SpecWriter.bytes_per_offset
        ) - 1  # `end` parameter in gcs is inclusive of last byte to return
        return (idx_start, idx_end)

    @staticmethod
    def get_spec_file_offsets(offsets):
        assert len(offsets) == 2 * SpecWriter.bytes_per_offset
        spec_start = int.from_bytes(offsets[:8], byteorder=SpecWriter.byteorder, signed=SpecWriter.signed)
        next_spec_start = int.from_bytes(offsets[8:], byteorder=SpecWriter.byteorder, signed=SpecWriter.signed)
        return (spec_start, next_spec_start - 1)  # `end` parameter in gcs is inclusive of last byte to return

    @staticmethod
    async def get_token_start_id(db, batch_id, job_id) -> Tuple[str, int]:
        in_batch_cache = JOB_TOKEN_CACHE[batch_id]
        index = in_batch_cache.bisect_key_right(job_id) - 1
        assert index < len(in_batch_cache)
        if index >= 0:
            token, start, end = in_batch_cache[index]
            if job_id in range(start, end):
                return (token, start)

        token, start_job_id, end_job_id = await SpecWriter._get_token_start_id_and_end_id(db, batch_id, job_id)

        # It is least likely that old batches or early bunches in a given
        # batch will be needed again
        if len(JOB_TOKEN_CACHE) == JOB_TOKEN_CACHE_MAX_BATCHES:
            JOB_TOKEN_CACHE.pop(min(JOB_TOKEN_CACHE.keys()))
        elif len(JOB_TOKEN_CACHE[batch_id]) == JOB_TOKEN_CACHE_MAX_BUNCHES_PER_BATCH:
            JOB_TOKEN_CACHE[batch_id].pop(0)

        JOB_TOKEN_CACHE[batch_id].add((token, start_job_id, end_job_id))

        return (token, start_job_id)

    @staticmethod
    async def _get_token_start_id_and_end_id(db, batch_id, job_id) -> Tuple[str, int, int]:
        bunch_record = await db.select_and_fetchone(
            '''
SELECT
batch_bunches.start_job_id,
batch_bunches.token,
(SELECT start_job_id FROM batch_bunches WHERE batch_id = %s AND start_job_id > %s ORDER BY start_job_id LIMIT 1) AS next_start_job_id,
batches.n_jobs
FROM batch_bunches
JOIN batches ON batches.id = batch_bunches.batch_id
WHERE batch_bunches.batch_id = %s AND batch_bunches.start_job_id <= %s
ORDER BY batch_bunches.start_job_id DESC
LIMIT 1;
''',
            (batch_id, job_id, batch_id, job_id),
            'get_token_start_id',
        )
        token = bunch_record['token']
        start_job_id = bunch_record['start_job_id']
        end_job_id = bunch_record['next_start_job_id'] or (bunch_record['n_jobs'] + 1)
        return (token, start_job_id, end_job_id)

    def __init__(self, file_store, batch_id):
        self.file_store = file_store
        self.batch_id = batch_id
        self.token = secret_alnum_string(16)

        self._data_bytes = bytearray()
        self._offsets_bytes = bytearray()
        self._n_elements = 0

    def add(self, data):
        data_bytes = data.encode('utf-8')
        start = len(self._data_bytes)

        self._offsets_bytes.extend(start.to_bytes(8, byteorder=SpecWriter.byteorder, signed=SpecWriter.signed))
        self._data_bytes.extend(data_bytes)

        self._n_elements += 1

    async def write(self):
        end = len(self._data_bytes)
        self._offsets_bytes.extend(end.to_bytes(8, byteorder=SpecWriter.byteorder, signed=SpecWriter.signed))

        await self.file_store.write_spec_file(
            self.batch_id, self.token, bytes(self._data_bytes), bytes(self._offsets_bytes)
        )
        return self.token
