import os
import unittest

from batch.server.database import Database


class Test(unittest.TestCase):
    def setUp(self):
        self.db = Database.create_synchronous(os.environ.get('CLOUD_SQL_CONFIG_PATH',
                                                             '/batch-secrets/batch-production-cloud-sql-config.json'))

    def test_create_drop(self):
        table_root = "foo"
        uid1 = self.db.temp_table_name_sync(table_root)
        uid2 = self.db.temp_table_name_sync(table_root)
        assert uid1.startswith(table_root)
        assert uid2.startswith(table_root)

        schema = {'id': 'BIGINT'}
        key = ['id']
        self.db.create_table_sync(uid1, schema, key)
        self.db.create_table_sync(uid2, schema, key)
        assert self.db.has_table_sync(uid1)
        assert self.db.has_table_sync(uid2)

        self.db.drop_table_sync(uid1, uid2)
        assert not self.db.has_table_sync(uid1)
        assert not self.db.has_table_sync(uid2)
