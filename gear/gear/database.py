import json
import aiomysql


async def create_database_pool(autocommit=True):
    with open('/sql-config/sql-config.json', 'r') as f:
        sql_config = json.loads(f.read())
    return await aiomysql.create_pool(host=sql_config['host'],
                                      port=sql_config['port'],
                                      db=sql_config['db'],
                                      user=sql_config['user'],
                                      password=sql_config['password'],
                                      charset='utf8',
                                      cursorclass=aiomysql.cursors.DictCursor,
                                      autocommit=autocommit)


async def just_execute(pool, sql, args=None):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, args)


async def execute_and_fetchone(pool, sql, args=None):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, args)
            return await cursor.fetchone()


async def execute_and_fetchall(pool, query, args=None):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            while True:
                rows = await cursor.fetchmany(100)
                if rows is None:
                    break
                for row in rows:
                    yield row


async def execute_insertone(pool, sql, args=None):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, args)
            return cursor.lastrowid
