import os
import unittest

from batch.server.database import Database


class Test(unittest.TestCase):
    def setUp(self):
        self.db = Database.create_synchronous(os.environ.get('CLOUD_SQL_CONFIG_PATH',
                                                             '/batch-secrets/batch-production-cloud-sql-config.json'))

    def test_create_drop(self):
        schema = {'id': 'BIGINT'}
        key = ['id']

        t1_name = self.db.create_temp_table_sync("t1", schema, key)
        t2_name = self.db.create_temp_table_sync("t2", schema, key)
        assert t1_name.startswith("t1")
        assert t2_name.startswith("t2")

        self.db.drop_table_sync(t1_name, t2_name)
        assert not self.db.has_table_sync(t1_name)
        assert not self.db.has_table_sync(t2_name)
