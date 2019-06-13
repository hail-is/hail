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

    batch_view_fields = {'cancelled', 'user', 'userdata'}

    def _select_fields(self, fields=None):
        assert fields is None or len(fields) != 0
        select_fields = []
        if fields is not None:
            for f in fields:
                if f in JobsTable.batch_view_fields:
                    f = f'`{self._db.batch.name}`.{f}'
                else:
                    f = f'`{self.name}`.{f}'
                select_fields.append(f)
        else:
            select_fields.append(f'`{self.name}`.*')
            for f in JobsTable.batch_view_fields:
                select_fields.append(f'{self._db.batch.name}.{f}')
        return select_fields


    def __init__(self, db):
        super().__init__(db, 'jobs')

    async def update_record(self, batch_id, job_id, **items):
        assert 'cancelled' not in items
        await super().update_record({'batch_id': batch_id, 'job_id': job_id}, items)

    async def get_all_records(self):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                batch_name = self._db.batch.name
                fields = ', '.join(self._select_fields())
                sql = f"""SELECT {fields} FROM `{self.name}` 
                          INNER JOIN {batch_name} ON `{self.name}`.batch_id = `{batch_name}`.id"""
                await cursor.execute(sql)
                return await cursor.fetchall()

    async def get_records(self, batch_id, ids, fields=None):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                batch_name = self._db.batch.name
                where_items = {'batch_id': batch_id, 'job_id': ids}
                where_template, where_values = make_where_statement(where_items)
                fields = ', '.join(self._select_fields(fields))
                sql = f"""SELECT {fields} FROM `{self.name}` 
                          INNER JOIN `{batch_name}` ON `{self.name}`.batch_id = `{batch_name}`.id 
                          WHERE {where_template}"""
                await cursor.execute(sql, tuple(where_values))
                result = await cursor.fetchall()
        return result

    async def get_undeleted_records(self, batch_id, ids, user):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                batch_name = self._db.batch.name
                where_template, where_values = make_where_statement({'batch_id': batch_id, 'job_id': ids, f'user': user})
                fields = ', '.join(self._select_fields())
                sql = f"""SELECT {fields} FROM `{self.name}` 
                INNER JOIN `{batch_name}` ON `{self.name}`.batch_id = `{batch_name}`.id
                WHERE {where_template} AND EXISTS
                (SELECT id from `{batch_name}` WHERE `{batch_name}`.id = batch_id AND `{batch_name}`.deleted = FALSE)"""
                await cursor.execute(sql, tuple(where_values))
                result = await cursor.fetchall()
        return result

    async def has_record(self, batch_id, job_id):
        return await super().has_record({'batch_id': batch_id, 'job_id': job_id})

    async def delete_record(self, batch_id, job_id):
        await super().delete_record({'batch_id': batch_id, 'job_id': job_id})

    async def get_incomplete_parents(self, batch_id, job_id):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                jobs_parents_name = self._db.jobs_parents.name
                sql = f"""SELECT `{self.name}`.batch_id, `{self.name}`.job_id FROM `{self.name}`
                          INNER JOIN `{jobs_parents_name}`
                          ON `{self.name}`.batch_id = `{jobs_parents_name}`.batch_id AND `{self.name}`.job_id = `{jobs_parents_name}`.parent_id
                          WHERE `{self.name}`.state IN %s AND `{jobs_parents_name}`.batch_id = %s AND `{jobs_parents_name}`.job_id = %s"""

                await cursor.execute(sql, (('Pending', 'Ready', 'Running'), batch_id, job_id))
                result = await cursor.fetchall()
                return [(record['batch_id'], record['job_id']) for record in result]

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
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                batch_name = self._db.batch.name
                where_template, where_values = make_where_statement(condition)
                fields = ', '.join(self._select_fields())
                sql = f"""SELECT {fields} FROM `{self.name}`
                          INNER JOIN `{batch_name}` ON `{self.name}`.batch_id = `{batch_name}`.id 
                          WHERE {where_template}"""
                await cursor.execute(sql, tuple(where_values))
                return await cursor.fetchall()

    async def update_with_log_ec(self, batch_id, job_id, task_name, uri, exit_code, **items):
        await self.update_record(batch_id, job_id,
                                 **{JobsTable.log_uri_mapping[task_name]: uri,
                                    JobsTable.exit_code_mapping[task_name]: exit_code},
                                 **items)

    async def get_log_uri(self, batch_id, job_id, task_name):
        uri_field = JobsTable.log_uri_mapping[task_name]
        records = await self.get_records(batch_id, job_id, fields=[uri_field])
        if records:
            assert len(records) == 1
            return records[0][uri_field]
        return None

    @staticmethod
    def exit_code_field(task_name):
        return JobsTable.exit_code_mapping[task_name]

    async def get_parents(self, batch_id, job_id):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                jobs_parents_name = self._db.jobs_parents.name
                batch_name = self._db.batch.name
                fields = ', '.join(self._select_fields())
                sql = f"""SELECT {fields} FROM `{self.name}`
                          INNER JOIN `{batch_name}` ON `{self.name}`.batch_id = `{batch_name}`.id
                          INNER JOIN `{jobs_parents_name}`
                          ON `{self.name}`.batch_id = `{jobs_parents_name}`.batch_id AND `{self.name}`.job_id = `{jobs_parents_name}`.parent_id
                          WHERE `{jobs_parents_name}`.batch_id = %s AND `{jobs_parents_name}`.job_id = %s"""
                await cursor.execute(sql, (batch_id, job_id))
                return await cursor.fetchall()

    async def get_children(self, batch_id, parent_id):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                jobs_parents_name = self._db.jobs_parents.name
                batch_name = self._db.batch.name
                fields = ', '.join(self._select_fields())
                sql = f"""SELECT {fields} FROM `{self.name}`
                          INNER JOIN `{batch_name}` ON `{self.name}`.batch_id = `{batch_name}`.id
                          INNER JOIN `{jobs_parents_name}`
                          ON `{self.name}`.batch_id = `{jobs_parents_name}`.batch_id AND `{self.name}`.job_id = `{jobs_parents_name}`.job_id
                          WHERE `{jobs_parents_name}`.batch_id = %s AND `{jobs_parents_name}`.parent_id = %s"""
                await cursor.execute(sql, (batch_id, parent_id))
                return await cursor.fetchall()


class JobsParentsTable(Table):
    def __init__(self, db):
        super().__init__(db, 'jobs-parents')

    async def has_record(self, batch_id, job_id, parent_id):
        return await super().has_record({'batch_id': batch_id, 'job_id': job_id, 'parent_id': parent_id})

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
