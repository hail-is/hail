import os
import json
import pymysql
import aiomysql
import logging
import functools
import ssl

from hailtop.utils import sleep_and_backoff, LoggingTimer


log = logging.getLogger('gear.database')


# 1213 - Deadlock found when trying to get lock; try restarting transaction
# 2003 - Can't connect to MySQL server on ...
# 2013 - Lost connection to MySQL server during query ([Errno 104] Connection reset by peer)
retry_codes = (1213, 2003, 2013)


def retry_transient_mysql_errors(f):
    @functools.wraps(f)
    async def wrapper(*args, **kwargs):
        delay = 0.1
        while True:
            try:
                return await f(*args, **kwargs)
            except pymysql.err.OperationalError as e:
                if e.args[0] in retry_codes:
                    log.warning(f'encountered pymysql error, retrying {e}', exc_info=True)
                else:
                    raise
            delay = await sleep_and_backoff(delay)
    return wrapper


def transaction(db, **transaction_kwargs):
    def transformer(fun):
        @functools.wraps(fun)
        @retry_transient_mysql_errors
        async def wrapper(*args, **kwargs):
            async with db.start(**transaction_kwargs) as tx:
                return await fun(tx, *args, **kwargs)
        return wrapper
    return transformer


async def aenter(acontext_manager):
    return await acontext_manager.__aenter__()


async def aexit(acontext_manager, exc_type=None, exc_val=None, exc_tb=None):
    return await acontext_manager.__aexit__(exc_type, exc_val, exc_tb)


def get_sql_config(config_file=None):
    if config_file is None:
        config_file = os.environ.get('HAIL_DATABASE_CONFIG_FILE',
                                     '/sql-config/sql-config.json')
    with open(config_file, 'r') as f:
        sql_config = json.loads(f.read())
    check_sql_config(sql_config)
    return sql_config


def check_sql_config(sql_config):
    assert sql_config is not None
    for key in ('ssl-cert', 'ssl-key', 'ssl-ca', 'ssl-mode'):
        assert sql_config.get(key) is not None, key
    for key in ('ssl-cert', 'ssl-key', 'ssl-ca'):
        if not os.path.isfile(sql_config[key]):
            raise ValueError(f'specified {key}, {sql_config[key]} does not exist')
    log.info('using tls and verifying server certificates for MySQL')


database_ssl_context = None


def get_database_ssl_context(sql_config=None):
    global database_ssl_context
    if database_ssl_context is None:
        if sql_config is None:
            sql_config = get_sql_config()
        database_ssl_context = ssl.create_default_context(
            cafile=sql_config['ssl-ca'])
        database_ssl_context.load_cert_chain(sql_config['ssl-cert'],
                                             keyfile=sql_config['ssl-key'],
                                             password=None)
        database_ssl_context.verify_mode = ssl.CERT_REQUIRED
        database_ssl_context.check_hostname = False
    return database_ssl_context


@retry_transient_mysql_errors
async def create_database_pool(config_file=None, autocommit=True, maxsize=10):
    sql_config = get_sql_config(config_file)
    ssl_context = get_database_ssl_context(sql_config)
    assert ssl_context is not None
    return await aiomysql.create_pool(
        maxsize=maxsize,
        # connection args
        host=sql_config['host'], user=sql_config['user'], password=sql_config['password'],
        db=sql_config.get('db'), port=sql_config['port'], charset='utf8',
        ssl=ssl_context, cursorclass=aiomysql.cursors.DictCursor, autocommit=autocommit)


class TransactionAsyncContextManager:
    def __init__(self, db_pool, read_only):
        self.db_pool = db_pool
        self.read_only = read_only
        self.tx = None

    async def __aenter__(self):
        tx = Transaction()
        await tx.async_init(self.db_pool, self.read_only)
        self.tx = tx
        return tx

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.tx._aexit(exc_type, exc_val, exc_tb)
        self.tx = None


class Transaction:
    def __init__(self):
        self.conn_context_manager = None
        self.conn = None

    async def async_init(self, db_pool, read_only):
        self.conn_context_manager = db_pool.acquire()
        self.conn = await aenter(self.conn_context_manager)
        async with self.conn.cursor() as cursor:
            if read_only:
                await cursor.execute('START TRANSACTION READ ONLY;')
            else:
                await cursor.execute('START TRANSACTION;')

    async def _aexit(self, exc_type, exc_val, exc_tb):
        try:
            if self.conn is not None:
                try:
                    if exc_type:
                        await self.conn.rollback()
                    else:
                        await self.conn.commit()
                finally:
                    self.conn = None
        finally:
            if self.conn_context_manager is not None:
                try:
                    await aexit(self.conn_context_manager, exc_type, exc_val, exc_tb)
                finally:
                    self.conn_context_manager = None

    async def commit(self):
        assert self.conn
        self.conn.commit()
        self.conn = None

        await aexit(self.conn_context_manager)
        self.conn_context_manager = None

    async def rollback(self):
        assert self.conn
        self.conn.rollback()
        self.conn = None

        await aexit(self.conn_context_manager)
        self.conn_context_manager = None

    async def just_execute(self, sql, args=None):
        assert self.conn
        async with self.conn.cursor() as cursor:
            await cursor.execute(sql, args)

    async def execute_and_fetchone(self, sql, args=None):
        assert self.conn
        async with self.conn.cursor() as cursor:
            await cursor.execute(sql, args)
            return await cursor.fetchone()

    async def execute_and_fetchall(self, sql, args=None, timer_description=None):
        assert self.conn
        async with self.conn.cursor() as cursor:
            if timer_description is None:
                await cursor.execute(sql, args)
            else:
                async with LoggingTimer(f'{timer_description}: execute_and_fetchall: execute', threshold_ms=20):
                    await cursor.execute(sql, args)
            while True:
                if timer_description is None:
                    rows = await cursor.fetchmany(100)
                else:
                    async with LoggingTimer(f'{timer_description}: execute_and_fetchall: fetchmany', threshold_ms=20):
                        rows = await cursor.fetchmany(100)
                if not rows:
                    break
                for row in rows:
                    yield row

    async def execute_insertone(self, sql, args=None):
        assert self.conn
        async with self.conn.cursor() as cursor:
            await cursor.execute(sql, args)
            return cursor.lastrowid

    async def execute_update(self, sql, args=None):
        assert self.conn
        async with self.conn.cursor() as cursor:
            return await cursor.execute(sql, args)

    async def execute_many(self, sql, args_array):
        assert self.conn
        async with self.conn.cursor() as cursor:
            return await cursor.executemany(sql, args_array)


class Database:
    def __init__(self):
        self.pool = None

    async def async_init(self, config_file=None, maxsize=10):
        self.pool = await create_database_pool(config_file=config_file, autocommit=False, maxsize=maxsize)

    def start(self, read_only=False):
        return TransactionAsyncContextManager(self.pool, read_only)

    @retry_transient_mysql_errors
    async def just_execute(self, sql, args=None):
        async with self.start() as tx:
            await tx.just_execute(sql, args)

    @retry_transient_mysql_errors
    async def execute_and_fetchone(self, sql, args=None):
        async with self.start() as tx:
            return await tx.execute_and_fetchone(sql, args)

    @retry_transient_mysql_errors
    async def select_and_fetchone(self, sql, args=None):
        async with self.start(read_only=True) as tx:
            return await tx.execute_and_fetchone(sql, args)

    async def execute_and_fetchall(self, sql, args=None, timer_description=None):
        async with self.start() as tx:
            async for row in tx.execute_and_fetchall(sql, args, timer_description):
                yield row

    async def select_and_fetchall(self, sql, args=None, timer_description=None):
        async with self.start(read_only=True) as tx:
            async for row in tx.execute_and_fetchall(sql, args, timer_description):
                yield row

    @retry_transient_mysql_errors
    async def execute_insertone(self, sql, args=None):
        async with self.start() as tx:
            return await tx.execute_insertone(sql, args)

    @retry_transient_mysql_errors
    async def execute_update(self, sql, args=None):
        async with self.start() as tx:
            return await tx.execute_update(sql, args)

    @retry_transient_mysql_errors
    async def execute_many(self, sql, args_array):
        async with self.start() as tx:
            return await tx.execute_many(sql, args_array)

    async def async_close(self):
        self.pool.close()
        await self.pool.wait_closed()
