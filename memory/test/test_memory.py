import concurrent
import unittest
import uuid
from memory.client import BlockingMemoryClient

from hailtop.config import get_user_config
from hailtop.google_storage import GCS


class FileTests(unittest.TestCase):
    def setUp(self):
        bucket_name = get_user_config().get('batch', 'bucket')
        token = uuid.uuid4()
        self.test_path = f'gs://{ bucket_name }/memory-tests/{ token }'

        self.fs = GCS(concurrent.futures.ThreadPoolExecutor(), project='hail-vdc')
        self.client = BlockingMemoryClient('hail-vdc', fs=self.fs)
        self.temp_files = set()

    def tearDown(self):
        self.fs.delete_gs_files(self.test_path)
        self.client.close()

    def add_temp_file_from_string(self, name: str, str_value: str):
        handle = f'{ self.test_path }/{ name }'
        self.fs._write_gs_file_from_string(handle, str_value)
        return handle

    def test_non_existent(self):
        for _ in range(3):
            self.assertIsNone(self.client._get_file_if_exists(f'{ self.test_path }/nonexistent'))

    def test_small(self):
        cases = [('empty_file', b''), ('null', b'\0'), ('small', b'hello world')]
        for file, data in cases:
            handle = self.add_temp_file_from_string(file, data)
            i = 0
            cached = self.client.read_file(handle)
            while cached is None and i < 10:
                cached = self.client.read_file(handle)
                i += 1
            self.assertEqual(cached, data)