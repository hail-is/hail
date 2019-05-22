from ..database import Database, Table, make_where_statement


class BatchDatabase(Database):
    async def __init__(self, config_file):
        await super().__init__(config_file)

        self.jobs = JobsTable(self)
        self.jobs_parents = JobsParentsTable(self)
        self.batch = BatchTable(self)


class JobsTable(Table):
    log_uri_mapping = {'input': 'input_log_uri',
                       'main': 'main_log_uri',
                       'output': 'output_log_uri'}

    exit_code_mapping = {'input': 'input_exit_code',
                         'main': 'main_exit_code',
                         'output': 'output_exit_code'}

    def __init__(self, db):
        super().__init__(db, 'jobs')

    async def update_record(self, id, **items):
        await super().update_record({'id': id}, items)

    async def get_all_records(self):
        return await super().get_all_records()

    async def get_records(self, ids, fields=None):
        return await super().get_records({'id': ids}, fields)

    async def get_undeleted_records(self, ids, user):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                batch_name = self._db.batch.name
                where_template, where_values = make_where_statement({'id': ids, 'user': user})
                sql = f"""SELECT * FROM `{self.name}` WHERE {where_template} AND EXISTS 
                (SELECT id from `{batch_name}` WHERE `{batch_name}`.id = batch_id AND `{batch_name}`.deleted = FALSE)"""
                await cursor.execute(sql, tuple(where_values))
                result = await cursor.fetchall()
        return result

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

    async def get_records_by_batch(self, batch_id):
        return await self.get_records_where({'batch_id': batch_id})

    async def get_records_where(self, condition):
        return await super().get_records(condition)

    async def update_with_log_ec(self, id, task_name, uri, exit_code, **items):
        await self.update_record(id,
                                 **{JobsTable.log_uri_mapping[task_name]: uri,
                                    JobsTable.exit_code_mapping[task_name]: exit_code},
                                 **items)

    async def get_log_uri(self, id, task_name):
        uri_field = JobsTable.log_uri_mapping[task_name]
        records = await self.get_records(id, fields=[uri_field])
        if records:
            assert len(records) == 1
            return records[0][uri_field]
        return None

    def exit_code_field(self, task_name):
        return JobsTable.exit_code_mapping[task_name]

    async def get_parents(self, job_id):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                jobs_parents_name = self._db.jobs_parents.name
                sql = f"""SELECT * FROM `{self.name}`
                          INNER JOIN `{jobs_parents_name}`
                          ON `{self.name}`.id = `{jobs_parents_name}`.parent_id
                          WHERE `{jobs_parents_name}`.job_id = %s"""
                await cursor.execute(sql, job_id)
                return await cursor.fetchall()

    async def get_children(self, parent_id):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                jobs_parents_name = self._db.jobs_parents.name
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
        return await super().get_records({'id': ids}, fields)

    async def get_records_where(self, condition):
        return await super().get_records(condition)

    async def has_record(self, id):
        return await super().has_record({'id': id})

    async def delete_record(self, id):
        return await super().delete_record({'id': id})

    async def get_finished_deleted_records(self):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = f"SELECT * FROM `{self.name}` WHERE `deleted` = TRUE AND `n_completed` = `n_jobs`"
                await cursor.execute(sql)
                result = await cursor.fetchall()
        return result

    async def get_undeleted_records(self, ids, user):
        return await super().get_records({'id': ids, 'user': user, 'deleted': False})
