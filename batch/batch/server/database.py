import aiomysql


class Database:
    @classmethod
    def create(cls, host, port, db_name, user, password, charset='utf8mb4'):
        db = cls(host, port, db_name, user, password, charset)
        db.connection = aiomysql.connect(host=host,
                                         port=port,
                                         user=user,
                                         password=password,
                                               charset=charset,
                                               cursorclass=aiomysql.cursors.DictCursor,
                                               autocommit=True)

        async with db.connection.cursor() as cursor:
            cursor._defer_warnings = True
            await cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            await db.connection.select_db(db_name)

        db._jobs_table = await JobsTable.create(db)

        return db

    def __init__(self, host, port, db_name, user, password, charset='utf8mb4'):
        self.name = db_name
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.charset = charset
        self.connection = None
        self._jobs_table = None

    async def delete(self):
        async with self.connection.cursor() as cursor:
            await cursor.execute(f"DROP DATABASE IF EXISTS `{self.name}`")
            await self.connection.select_db(None)

    @property
    def jobs(self):
        return self._jobs_table


class Table:
    def __init__(self, db, table_name):
        self._table_name = table_name
        self._db = db

    async def _create_table(self, schema, keys):
        assert all([k in schema for k in keys])

        async with self._db.connection.cursor() as cursor:
            cursor._defer_warnings = True

            schema = ", ".join([f"`{n}` {t}" for n, t in schema.items()])

            key_names = ", ".join([f'`{name.replace("`", "``")}`' for name in keys])
            keys = f", PRIMARY KEY( {key_names} )" if keys else ''

            sql = f"CREATE TABLE IF NOT EXISTS `{self._table_name}` ( {schema} {keys})"
            await cursor.execute(sql)

    async def _new_record(self, items):
        names = ", ".join([f'`{name.replace("`", "``")}`' for name in items.keys()])
        values_template = ", ".join(["%s" for _ in items.values()])
        async with self._db.connection.cursor() as cursor:
            sql = f"INSERT INTO `{self._table_name}` ({names}) VALUES ({values_template})"
            await cursor.execute(sql, tuple(items.values()))
            id = await cursor.lastrowid
        return id

    async def _update_record(self, key, items):
        async with self._db.connection.cursor() as cursor:
            if len(items) != 0:
                items_template = ", ".join([f'`{k.replace("`", "``")}` = %s' for k, v in items.items()])
                key_template = ", ".join([f'`{k.replace("`", "``")}` = %s' for k, v in key.items()])

                values = items.values()
                key_values = key.values()

                sql = f"UPDATE `{self._table_name}` SET {items_template} WHERE {key_template}"

                await cursor.execute(sql, (*values, *key_values))

    async def _get_records(self, key):
        async with self._db.connection.cursor() as cursor:
            key_template = ", ".join([f'`{k.replace("`", "``")}` = %s' for k, v in key.items()])
            key_values = key.values()
            sql = f"SELECT * FROM `{self._table_name}` WHERE {key_template}"
            result = await cursor.execute(sql, tuple(key_values))
        return result

    async def _has_record(self, key):
        async with self._db.connection.cursor() as cursor:
            key_template = ", ".join([f'`{k.replace("`", "``")}` = %s' for k, v in key.items()])
            key_values = key.values()
            sql = f"SELECT COUNT(1) FROM `{self._table_name}` WHERE {key_template}"
            count = await cursor.execute(sql, tuple(key_values))
        return count == 1


class JobsTable(Table):
    @classmethod
    async def create(cls, db):
        jt = cls(db)
        await jt._create_table(jt._schema, jt._keys)
        return jt

    def __init__(self, db):
        super().__init__(db, table_name='jobs')

        self._schema = {'id': 'INT NOT NULL AUTO_INCREMENT',
                        'state': 'VARCHAR(40) NOT NULL',
                        'exit_code': 'INT',
                        'batch_id': 'INT',
                        'scratch_folder': 'VARCHAR(200)',
                        'pod_name': 'VARCHAR(200)'}

        self._keys = ['id']

    async def new_record(self, **items):
        assert all([k in self._schema for k in items.keys()])
        return await self._new_record(items)

    async def update_record(self, id, **items):
        assert all([k in self._schema for k in items.keys()])
        await self._update_record({'id': id}, items)

    async def has_record(self, id):
        return await self._has_record({'id': id})
