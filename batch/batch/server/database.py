from ..database import Database, Table


class BatchDatabase(Database):
    async def __init__(self, config_file):
        await super().__init__(config_file)

        self.jobs = JobsTable(self)
        self.jobs_parents = JobsParentsTable(self)
        self.batch = BatchTable(self)
        self.batch_jobs = BatchJobsTable(self)


class JobsTable(Table):
    uri_log_mapping = {'input': 'input_log_uri',
                       'main': 'main_log_uri',
                       'output': 'output_log_uri'}

    def __init__(self, db):
        super().__init__(db, 'jobs')

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

                await cursor.execute(sql, (('Created', 'Ready'), id))
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

    async def get_parents(self, job_id):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                jobs_parents_name = self._db.jobs.name
                sql = f"""SELECT * FROM `{self.name}`
                          INNER JOIN `{jobs_parents_name}`
                          ON `{self.name}`.id = `{jobs_parents_name}`.parent_id
                          WHERE `{jobs_parents_name}`.job_id = %s"""
                await cursor.execute(sql, job_id)
                return await cursor.fetchall()

    async def get_children(self, parent_id):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                jobs_parents_name = self._db.jobs.name
                sql = f"""SELECT * FROM `{self.name}`
                          INNER JOIN `{jobs_parents_name}`
                          ON `{self.name}`.id = `{jobs_parents_name}`.job_id
                          WHERE `{jobs_parents_name}`.parent_id = %s"""
                await cursor.execute(sql, parent_id)
                return await cursor.fetchall()


class JobsParentsTable(Table):
    def __init__(self, db):
        super().__init__(db, 'jobs-parents')

    async def has_record(self, job_id, parent_id):
        return await super().has_record({'job_id': job_id, 'parent_id': parent_id})

    async def delete_records_where(self, condition):
        return await super().delete_record(condition)


class BatchTable(Table):
    def __init__(self, db):
        super().__init__(db, 'batch')

    async def update_record(self, id, **items):
        await super().update_record({'id': id}, items)

    async def get_all_records(self):
        return await super().get_all_records()

    async def get_records(self, ids, fields=None):
        return await super().get_record({'id': ids}, fields)

    async def has_record(self, id):
        return await super().has_record({'id': id})

    async def get_jobs(self, batch_id):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                batch_jobs_name = self._db.batch_jobs.name
                sql = f"""SELECT * FROM `{self.name}`
                                  INNER JOIN `{batch_jobs_name}`
                                  ON `{self.name}`.id = `{batch_jobs_name}`.job_id
                                  WHERE `{batch_jobs_name}`.batch_id = %s"""
                await cursor.execute(sql, batch_id)
                return await cursor.fetchall()


class BatchJobsTable(Table):
    def __init__(self, db):
        super().__init__(db, 'batch-jobs')

    async def delete_record(self, batch_id, job_id):
        await super().delete_record({'batch_id': batch_id, 'job_id': job_id})

    async def has_record(self, batch_id, job_id):
        return await super().has_record({'batch_id': batch_id, 'job_id': job_id})
