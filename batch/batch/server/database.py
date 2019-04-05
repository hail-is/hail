import json
import uuid
import asyncio
import aiomysql
import pymysql
from asyncinit import asyncinit


def run_synchronous(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@asyncinit
class Database:
    @staticmethod
    def create_synchronous(config_file):
        db = run_synchronous(Database(config_file))
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

    async def has_table(self, name):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = f"SELECT * FROM INFORMATION_SCHEMA.tables " \
                    f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s"
                await cursor.execute(sql, (self.db, name))
                result = cursor.fetchone()
        return result.result() is not None

    def has_table_sync(self, name):
        return run_synchronous(self.has_table(name))

    async def drop_table(self, *names):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("DROP TABLE IF EXISTS {}".format(",".join([f'`{name}`' for name in names])))

    def drop_table_sync(self, *names):
        return run_synchronous(self.drop_table(*names))

    async def create_table(self, name, schema, keys, can_exist=True):
        assert all([k in schema for k in keys])

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                schema = ", ".join([f"`{n}` {t}" for n, t in schema.items()])
                key_names = ", ".join([f'`{name.replace("`", "``")}`' for name in keys])
                keys = f", PRIMARY KEY( {key_names} )" if keys else ''
                exists = 'IF NOT EXISTS' if can_exist else ''
                sql = f"CREATE TABLE {exists} `{name}` ( {schema} {keys})"
                await cursor.execute(sql)

    def create_table_sync(self, name, schema, keys):
        return run_synchronous(self.create_table(name, schema, keys))

    async def create_temp_table(self, root_name, schema, keys):
        for i in range(5):
            try:
                suffix = uuid.uuid4().hex[:8]
                name = f'{root_name}-{suffix}'
                await self.create_table(name, schema, keys, can_exist=False)
                return name
            except pymysql.err.InternalError:
                pass
        raise Exception("Too many attempts to get temp table.")

    def create_temp_table_sync(self, root_name, schema, keys):
        return run_synchronous(self.create_temp_table(root_name, schema, keys))
