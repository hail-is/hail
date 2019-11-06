import json
import aiomysql
import logging

log = logging.getLogger('gear.database')


async def create_database_pool(autocommit=True, maxsize=10):
    with open('/sql-config/sql-config.json', 'r') as f:
        sql_config = json.loads(f.read())
    return await aiomysql.create_pool(
        maxsize=maxsize,
        # connection args
        host=sql_config['host'], user=sql_config['user'], password=sql_config['password'],
        db=sql_config['db'], port=sql_config['port'], charset='utf8',
        cursorclass=aiomysql.cursors.DictCursor, autocommit=autocommit)


class Database:
    def __init__(self):
        self.pool = None

    async def async_init(self, autocommit=True, maxsize=10):
        self.pool = await create_database_pool(autocommit, maxsize)

    async def just_execute(self, sql, args=None):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, args)

    async def execute_and_fetchone(self, sql, args=None):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, args)
                return await cursor.fetchone()

    async def execute_and_fetchall(self, sql, args=None):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, args)
                while True:
                    rows = await cursor.fetchmany(100)
                    if not rows:
                        break
                    for row in rows:
                        yield row

    async def execute_insertone(self, sql, args=None):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, args)
                return cursor.lastrowid

    async def execute_update(self, sql, args=None):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                return await cursor.execute(sql, args)

    async def async_close(self):
        self.pool.close()
        await self.pool.wait_closed()
