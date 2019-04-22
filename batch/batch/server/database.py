import os

from ..database import Database, Table


class BatchDatabase(Database):
    async def __init__(self, config_file):
        await super().__init__(config_file)

        self.jobs = await JobsTable(self, os.environ.get('JOBS_TABLE', 'jobs'))
        self.jobs_parents = await JobsParentsTable(self, os.environ.get('JOBS_PARENTS_TABLE', 'jobs-parents'))
        self.batch = await BatchTable(self, os.environ.get('BATCH_TABLE', 'batch'))
        self.batch_jobs = await BatchJobsTable(self, os.environ.get('BATCH_JOBS_TABLE', 'batch-jobs'))


class JobsTable(Table):
    uri_log_mapping = {'input': 'input_log_uri',
                       'main': 'main_log_uri',
                       'output': 'output_log_uri'}

    async def __init__(self, db, name='jobs'):
        schema = {'id': 'BIGINT NOT NULL AUTO_INCREMENT',
                  'state': 'VARCHAR(40) NOT NULL',
                  'exit_code': 'INT',
                  'batch_id': 'BIGINT',
                  'scratch_folder': 'VARCHAR(1024)',
                  'pod_name': 'VARCHAR(1024)',
                  'pvc': 'TEXT(65535)',
                  'callback': 'TEXT(65535)',
                  'task_idx': 'INT NOT NULL',
                  'always_run': 'BOOLEAN',
                  'cancelled': 'BOOLEAN',
                  'time_created': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                  'duration': 'BIGINT',
                  'userdata': 'TEXT(65535)',
                  'attributes': 'TEXT(65535)',
                  'tasks': 'TEXT(65535)',
                  'input_log_uri': 'VARCHAR(1024)',
                  'main_log_uri': 'VARCHAR(1024)',
                  'output_log_uri': 'VARCHAR(1024)'}

        keys = ['id']

        await super().__init__(db, name, schema, keys)

    async def update_record(self, id, **items):
        await super().update_record({'id': id}, items)

    async def get_all_records(self):
        return await super().get_all_records()

    async def get_records(self, ids, fields=None):
        return await super().get_record({'id': ids}, fields)

    async def has_record(self, id):
        return await super().has_record({'id': id})

    async def delete_record(self, id):
        await super().delete_record({'id': id})

    async def get_incomplete_parents(self, id):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                jobs_parents_name = self._db.jobs_parents.name
                sql = f"""SELECT id FROM `{self.name}`
                          INNER JOIN `{jobs_parents_name}`
                          ON `{self.name}`.id = `{jobs_parents_name}`.parent_id
                          WHERE `{self.name}`.state IN %s AND `{jobs_parents_name}`.job_id = %s"""

                await cursor.execute(sql.replace('\n', ' '), (('Created', 'Ready'), id))
                result = await cursor.fetchall()
                return [record['id'] for record in result]

    async def get_record_by_pod(self, pod):
        records = await self.get_records_where({'pod_name': pod})
        if len(records) == 0:  # pylint: disable=R1705
            return None
        elif len(records) == 1:
            return records[0]
        else:
            jobs_w_pod = [record['id'] for record in records]
            raise Exception("'jobs' table error. Cannot have the same pod in more than one record.\n"
                            f"Found the following jobs matching pod name '{pod}':\n" + ",".join(jobs_w_pod))

    async def get_records_where(self, condition):
        return await super().get_record(condition)

    async def update_log_uri(self, id, task_name, uri):
        await self.update_record(id, **{JobsTable.uri_log_mapping[task_name]: uri})

    async def get_log_uri(self, id, task_name):
        uri_field = JobsTable.uri_log_mapping[task_name]
        records = await self.get_records(id, fields=[uri_field])
        if records:
            assert len(records) == 1
            return records[0][uri_field]
        return None


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

    async def has_record(self, job_id, parent_id):
        return await super().has_record({'job_id': job_id, 'parent_id': parent_id})

    async def delete_records_where(self, condition):
        return await super().delete_record(condition)


class BatchTable(Table):
    async def __init__(self, db, name='batch'):
        schema = {'id': 'BIGINT NOT NULL AUTO_INCREMENT',
                  'attributes': 'TEXT(65535)',
                  'callback': 'TEXT(65535)',
                  'ttl': 'INT',
                  'is_open': 'BOOLEAN NOT NULL',
                  'time_created': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                  }
        keys = ['id']

        await super().__init__(db, name, schema, keys)

    async def update_record(self, id, **items):
        await super().update_record({'id': id}, items)

    async def get_all_records(self):
        return await super().get_all_records()

    async def get_records(self, ids, fields=None):
        return await super().get_record({'id': ids}, fields)

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
