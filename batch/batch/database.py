import json
import logging
import asyncio
import aiomysql
from asyncinit import asyncinit

log = logging.getLogger('batch.database')


def run_synchronous(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@asyncinit
class Database:
    @classmethod
    def create_synchronous(cls, config_file):
        db = object.__new__(cls)
        run_synchronous(cls.__init__(db, config_file))
        return db

    async def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.loads(f.read().strip())

        self.host = config['host']
        self.port = config['port']
        self.user = config['user']
        self.db = config['db']
        self.password = config['password']
        self.charset = 'utf8'

        self.pool = await aiomysql.create_pool(host=self.host,
                                               port=self.port,
                                               db=self.db,
                                               user=self.user,
                                               password=self.password,
                                               charset=self.charset,
                                               cursorclass=aiomysql.cursors.DictCursor,
                                               autocommit=True)


def make_where_statement(items):
    template = []
    values = []
    for k, v in items.items():
        if isinstance(v, list):
            if len(v) == 0:
                template.append("FALSE")
            else:
                template.append(f'`{k.replace("`", "``")}` IN %s')
                values.append(v)
        elif v is None:
            template.append(f'`{k.replace("`", "``")}` IS NULL')
        elif v == "NOT NULL":
            template.append(f'`{k.replace("`", "``")}` IS NOT NULL')
        else:
            template.append(f'`{k.replace("`", "``")}` = %s')
            values.append(v)

    template = " AND ".join(template)
    return template, values


class Table:  # pylint: disable=R0903
    def __init__(self, db, name):
        self.name = name
        self._db = db

    def new_record_template(self, *field_names):
        names = ", ".join([f'`{name.replace("`", "``")}`' for name in field_names])
        values = ", ".join([f"%({name})s" for name in field_names])
        sql = f"INSERT INTO `{self.name}` ({names}) VALUES ({values})"
        return sql

    async def new_record(self, **items):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = self.new_record_template(*items)
                await cursor.execute(sql, items)
                return cursor.lastrowid  # This returns 0 unless an autoincrement field is in the table

    async def update_record(self, where_items, set_items):
        if len(set_items) != 0:
            async with self._db.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    where_template, where_values = make_where_statement(where_items)
                    set_template = ", ".join([f'`{k.replace("`", "``")}` = %s' for k, v in set_items.items()])
                    set_values = set_items.values()
                    sql = f"UPDATE `{self.name}` SET {set_template} WHERE {where_template}"
                    await cursor.execute(sql, (*set_values, *where_values))

    async def get_records(self, where_items, select_fields=None):
        assert select_fields is None or len(select_fields) != 0
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                where_template, where_values = make_where_statement(where_items)
                select_fields = ",".join(select_fields) if select_fields is not None else "*"
                sql = f"SELECT {select_fields} FROM `{self.name}` WHERE {where_template}"
                await cursor.execute(sql, tuple(where_values))
                return await cursor.fetchall()

    async def get_all_records(self):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"SELECT * FROM `{self.name}`")
                return await cursor.fetchall()

    async def has_record(self, where_items):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                where_template, where_values = make_where_statement(where_items)
                sql = f"SELECT COUNT(1) FROM `{self.name}` WHERE {where_template}"
                await cursor.execute(sql, where_values)
                result = await cursor.fetchone()
                return result['COUNT(1)'] >= 1

    async def delete_record(self, where_items):
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                where_template, where_values = make_where_statement(where_items)
                sql = f"DELETE FROM `{self.name}` WHERE {where_template}"
                await cursor.execute(sql, tuple(where_values))


class JobsBuilder:
    jobs_fields = {'batch_id', 'job_id', 'state', 'pod_name',
                   'pvc_name', 'pvc_size', 'callback', 'attributes',
                   'tasks', 'task_idx', 'always_run', 'duration'}

    jobs_parents_fields = {'batch_id', 'job_id', 'parent_id'}

    def __init__(self, db):
        self._db = db
        self._is_open = True
        self._jobs = []
        self._jobs_parents = []

        self._jobs_sql = self._db.jobs.new_record_template(*JobsBuilder.jobs_fields)
        self._jobs_parents_sql = self._db.jobs_parents.new_record_template(*JobsBuilder.jobs_parents_fields)

    async def close(self):
        self._is_open = False

    def create_job(self, **items):
        assert self._is_open
        assert set(items) == JobsBuilder.jobs_fields, set(items)
        self._jobs.append(dict(items))

    def create_job_parent(self, **items):
        assert self._is_open
        assert set(items) == JobsBuilder.jobs_parents_fields, set(items)
        self._jobs_parents.append(dict(items))

    async def commit(self):
        assert self._is_open
        async with self._db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if len(self._jobs) > 0:
                    await cursor.executemany(self._jobs_sql, self._jobs)
                    n_jobs_inserted = cursor.rowcount
                    if n_jobs_inserted != len(self._jobs):
                        log.info(f'inserted {n_jobs_inserted} jobs, but expected {len(self._jobs)} jobs')
                        return False

                if len(self._jobs_parents) > 0:
                    await cursor.executemany(self._jobs_parents_sql, self._jobs_parents)
                    n_jobs_parents_inserted = cursor.rowcount
                    if n_jobs_parents_inserted != len(self._jobs_parents):
                        log.info(f'inserted {n_jobs_parents_inserted} jobs parents, but expected {len(self._jobs_parents)}')
                        return False
                return True


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
        assert not set(items).intersection(JobsTable.batch_view_fields)
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
            jobs_w_pod = [str((record['batch_id'], record['job_id'])) for record in records]
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
                await cursor.execute(sql, where_values)
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
