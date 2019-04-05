import os
import unittest

from batch.server.database import Database, run_synchronous


class Test(unittest.TestCase):
    def setUp(self):
        self.db = Database.create_synchronous(os.environ.get('CLOUD_SQL_CONFIG_PATH'))

    def temp_table(self):
        schema = {'id': 'INT',
                  'less_than_two': 'BOOLEAN',
                  'name': 'VARCHAR(100)'}
        keys = ['id']

        t = self.db.create_temporary_table_sync("temp", schema, keys)

        for i in range(1, 6):
            id = run_synchronous(t.new_record({'id': i,
                                                'less_than_two': (i < 2),
                                                'name': f"{i}"}))
            assert id == 0
        return t

    def test_create_drop(self):
        schema = {'id': 'BIGINT'}
        key = ['id']

        try:
            t1_name = self.db.create_temporary_table_sync("t1", schema, key)
            t2_name = self.db.create_temporary_table_sync("t2", schema, key)
            assert t1_name.startswith("t1")
            assert t2_name.startswith("t2")
        finally:
            self.db.drop_table_sync(t1_name, t2_name)
            assert not self.db.has_table_sync(t1_name)
            assert not self.db.has_table_sync(t2_name)

    def test_new_record(self):
        t = self.temp_table()
        try:
            run_synchronous(t.new_record({'id': 10,
                                           'less_than_two': False,
                                           'name': f"hello"}))
        finally:
            self.db.drop_table_sync(t.name)

    def test_update_record(self):
        t = self.temp_table()
        try:
            run_synchronous(t.update_record({'id': 3, 'name': '3'},
                                            {'name': 'hello'}))

            updated_record = run_synchronous(t.get_record({'id': 3}))
            assert len(updated_record) == 1
            assert updated_record[0]['name'] == 'hello'
        finally:
            self.db.drop_table_sync(t.name)

    def test_get_records(self):
        t = self.temp_table()
        try:
            records = run_synchronous(t.get_record({'id': [1, 2]}))
            assert len(records) == 2
            assert records == [{'id': 1, 'less_than_two': True, 'name': '1'},
                               {'id': 2, 'less_than_two': False, 'name': '2'}]
        finally:
            self.db.drop_table_sync(t.name)

    def test_select_records(self):
        t = self.temp_table()
        try:
            records = run_synchronous(t.get_record({'id': 1}, ['less_than_two']))
            assert len(records) == 1
            assert records == [{'less_than_two': True}]
        finally:
            self.db.drop_table_sync(t.name)

    def test_delete_records(self):
        t = self.temp_table()
        try:
            run_synchronous(t.delete_record({'id': 1}))
            records = run_synchronous(t.get_record({'id': 1}))
            assert len(records) == 0
        finally:
            self.db.drop_table_sync(t.name)

    def test_has_record(self):
        t = self.temp_table()
        try:
            assert run_synchronous(t.has_record({'id': 4}))
        finally:
            self.db.drop_table_sync(t.name)

    def test_backtick_names(self):
        schema = {'id': 'INT',
                  '`less_than_two`': 'BOOLEAN',
                  'name': 'VARCHAR(100)'}
        keys = ['id']

        t = self.db.create_temporary_table_sync("temp", schema, keys)

        try:
            records = run_synchronous(t.get_record({'`less_than_two`': True}))
            assert len(records) == 0
            run_synchronous(t.new_record({'id': 5, '`less_than_two`': False, 'name': "foo"}))
            assert run_synchronous(t.has_record({'`less_than_two`': False}))
            run_synchronous(t.update_record({'id': 5}, {'`less_than_two`': True}))
            run_synchronous(t.delete_record({'`less_than_two`': True}))
        finally:
            self.db.drop_table_sync(t.name)
