import secrets
import logging

log = logging.getLogger('batch.spec_writer')


class SpecWriter:
    byteorder = 'little'
    signed = False
    bytes_per_offset = 8

    @staticmethod
    def get_index_file_offsets(job_id, start_job_id):
        assert job_id >= start_job_id
        idx_start = SpecWriter.bytes_per_offset * (job_id - start_job_id)
        idx_end = (idx_start + 2 * SpecWriter.bytes_per_offset) - 1  # `end` parameter in gcs is inclusive of last byte to return
        return (idx_start, idx_end)

    @staticmethod
    def get_spec_file_offsets(offsets):
        assert len(offsets) == 2 * SpecWriter.bytes_per_offset
        spec_start = int.from_bytes(offsets[:8], byteorder=SpecWriter.byteorder, signed=SpecWriter.signed)
        next_spec_start = int.from_bytes(offsets[8:], byteorder=SpecWriter.byteorder, signed=SpecWriter.signed)
        return (spec_start, next_spec_start - 1)  # `end` parameter in gcs is inclusive of last byte to return

    @staticmethod
    async def get_token_start_id(db, batch_id, job_id):
        bunch_record = await db.select_and_fetchone(
            '''
SELECT start_job_id, token FROM batch_bunches
WHERE batch_id = %s AND start_job_id <= %s
ORDER BY start_job_id DESC
LIMIT 1;
''',
            (batch_id, job_id))
        token = bunch_record['token']
        start_job_id = bunch_record['start_job_id']
        return (token, start_job_id)

    def __init__(self, log_store, batch_id):
        self.log_store = log_store
        self.batch_id = batch_id
        self.token = ''.join([secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(16)])

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

        await self.log_store.write_spec_file(self.batch_id, self.token,
                                             bytes(self._data_bytes), bytes(self._offsets_bytes))
        return self.token
