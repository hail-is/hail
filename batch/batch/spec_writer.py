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
        bytes_per_job = 2 * SpecWriter.bytes_per_offset
        idx_start = bytes_per_job * (job_id - start_job_id)
        idx_end = (idx_start + bytes_per_job) - 1
        return (idx_start, idx_end)

    @staticmethod
    def read_spec_file_offsets(offsets):
        assert len(offsets) == 2 * SpecWriter.bytes_per_offset
        spec_start = int.from_bytes(offsets[:8], byteorder=SpecWriter.byteorder, signed=SpecWriter.signed)
        spec_end = int.from_bytes(offsets[8:], byteorder=SpecWriter.byteorder, signed=SpecWriter.signed)
        return (spec_start, spec_end)

    def __init__(self, log_store, batch_id):
        self.log_store = log_store
        self.batch_id = batch_id
        self.token = ''.join([secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(16)])

        self._data_bytes = bytearray()
        self._data_bytes.append(ord('['))

        self._offsets_bytes = bytearray()
        self._n_elements = 0

    def add(self, data):
        if self._n_elements > 0:
            self._data_bytes.append(ord(','))

        data_bytes = data.encode('utf-8')
        start = len(self._data_bytes)
        self._data_bytes.extend(data_bytes)
        end = len(self._data_bytes) - 1

        self._offsets_bytes.extend(start.to_bytes(8, byteorder=SpecWriter.byteorder, signed=SpecWriter.signed))
        self._offsets_bytes.extend(end.to_bytes(8, byteorder=SpecWriter.byteorder, signed=SpecWriter.signed))

        self._n_elements += 1

    async def write(self):
        self._data_bytes.append(ord(']'))
        await self.log_store.write_spec_file(self.batch_id, self.token,
                                             bytes(self._data_bytes), bytes(self._offsets_bytes))
        return self.token
