import json
import uuid
import asyncio
import aiomysql
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

    async def temp_table_name(self, root):
        suffix = uuid.uuid4().hex[:8]
        name = f'{root}-{suffix}'
        niter = 0
        while await self.has_table(name):
            suffix = uuid.uuid4().hex[:8]
            name = f'{root}-{suffix}'
            niter += 1
            if niter > 5:
                raise Exception("Too many attempts to get unique temp table.")
        return name

    def temp_table_name_sync(self, root):
        return run_synchronous(self.temp_table_name(root))

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

    async def create_table(self, name, schema, keys):
        assert all([k in schema for k in keys])

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                schema = ", ".join([f"`{n}` {t}" for n, t in schema.items()])
                key_names = ", ".join([f'`{name.replace("`", "``")}`' for name in keys])
                keys = f", PRIMARY KEY( {key_names} )" if keys else ''
                sql = f"CREATE TABLE IF NOT EXISTS `{name}` ( {schema} {keys})"
                await cursor.execute(sql)

    def create_table_sync(self, name, schema, keys):
        return run_synchronous(self.create_table(name, schema, keys))