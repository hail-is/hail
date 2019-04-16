import os
import json

from ..database import Database, Table, run_synchronous


class BatchDatabase(Database):
    @staticmethod
    def create_synchronous(config_file):
        db = run_synchronous(BatchDatabase(config_file))
        return db

    async def __init__(self, config_file):
        await super().__init__(config_file)

        self.jobs = await JobsTable(self, os.environ.get('JOBS_TABLE', 'jobs'))
        self.jobs_parents = await JobsParentsTable(self, os.environ.get('JOBS_PARENTS_TABLE', 'jobs-parents'))
        self.batch = await BatchTable(self, os.environ.get('BATCH_TABLE', 'batch'))
        self.batch_jobs = await BatchJobsTable(self, os.environ.get('BATCH_JOBS_TABLE', 'batch-jobs'))


class JobsTable(Table):
    async def __init__(self, db, name='jobs'):
        schema = {'id': 'BIGINT NOT NULL AUTO_INCREMENT',
                  'state': 'VARCHAR(40) NOT NULL',
                  'exit_code': 'INT',
                  'batch_id': 'BIGINT',
                  'scratch_folder': 'VARCHAR(1000)',
                  'pod_name': 'VARCHAR(1000)',
                  'pvc': 'TEXT(65535)',
                  'callback': 'TEXT(65535)',
                  'task_idx': 'INT NOT NULL',
                  'always_run': 'BOOLEAN',
                  'time_created': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                  'time_ended': 'TIMESTAMP',
                  'user': 'VARCHAR(1000)',
                  'attributes': 'TEXT(65535)',
                  'tasks': 'TEXT(65535)',
                  'parent_ids': 'TEXT(65535)',
                  'input_log_uri': 'VARCHAR(1000)',
                  'main_log_uri': 'VARCHAR(1000)',
                  'output_log_uri': 'VARCHAR(1000)'}

        keys = ['id']

        await super().__init__(db, name, schema, keys)

    async def update_record(self, id, **items):
        await super().update_record({'id': id}, items)

    async def get_records(self, ids, fields=None):
        assert isinstance(ids, list)
        return await super().get_record({'id': id}, fields)

    async def get_record(self, id, fields=None):
        records = await self.get_records({'id': id}, fields)
        assert len(records) == 1
        return records[0]

    async def has_record(self, id):
        return await super().has_record({'id': id})

    async def delete_record(self, id):
        await super().delete_record({'id': id})

    async def get_incomplete_parents(self, id):
        parent_ids = await self.get_record(id, ['parent_ids'])
        parent_ids = json.loads(parent_ids['parent_ids'])
        parent_records = await self.get_records(parent_ids)
        incomplete_parents = [pr['id'] for pr in parent_records.result()
                              if pr['state'] == 'Created' or pr['state'] == 'Ready']
        return incomplete_parents


class JobsParentsTable(Table):
    async def __init__(self, db, name='jobs-parents'):
        schema = {'job_id': 'BIGINT',
                  'parent_id': 'BIGINT'}
        keys = ['job_id', 'parent_id']

        await super().__init__(db, name, schema, keys)

    async def get_parents(self, job_id):
        result = await super().get_record({'job_id': job_id}, ['parent_id'])
        return [record['parent_id'] for record in result]

    async def get_children(self, parent_id):
        result = await super().get_record({'parent_id': parent_id}, ['job_id'])
        return [record['job_id'] for record in result]


class BatchTable(Table):
    async def __init__(self, db, name='batch'):
        schema = {'id': 'BIGINT NOT NULL AUTO_INCREMENT',
                  'attributes': 'TEXT(65535)',
                  'callback': 'TEXT(65535)',
                  'ttl': 'INT',
                  'is_open': 'BOOLEAN NOT NULL'
                  }
        keys = ['id']

        await super().__init__(db, name, schema, keys)

    async def update_record(self, id, **items):
        await super().update_record({'id': id}, items)

    async def get_records(self, ids, fields=None):
        assert isinstance(ids, list)
        return await super().get_record({'id': id}, fields)

    async def get_record(self, id, fields=None):
        records = await self.get_records({'id': id}, fields)
        assert len(records) == 1
        return records[0]

    async def has_record(self, id):
        return await super().has_record({'id': id})


class BatchJobsTable(Table):
    async def __init__(self, db, name='batch-jobs'):
        schema = {'batch_id': 'BIGINT',
                  'job_id': 'BIGINT'}
        keys = ['batch_id', 'job_id']

        await super().__init__(db, name, schema, keys)

    async def get_jobs(self, batch_id):
        result = await super().get_record({'batch_id': batch_id})
        return [record['job_id'] for record in result]

    async def delete_record(self, batch_id, job_id):
        await super().delete_record({'batch_id': batch_id, 'job_id': job_id})

    async def has_record(self, batch_id, job_id):
        return await super().has_record({'batch_id': batch_id, 'job_id': job_id})
