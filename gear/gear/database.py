import asyncio
import functools
import logging
import os
import ssl
import traceback
from typing import Optional

import aiomysql
import pymysql

from gear.metrics import DB_CONNECTION_QUEUE_SIZE, SQL_TRANSACTIONS, PrometheusSQLTimer
from hailtop.auth.sql_config import SQLConfig
from hailtop.utils import sleep_and_backoff

log = logging.getLogger('gear.database')


# 1040 - Too many connections
# 1213 - Deadlock found when trying to get lock; try restarting transaction
# 2003 - Can't connect to MySQL server on ...
# 2013 - Lost connection to MySQL server during query ([Errno 104] Connection reset by peer)
operational_error_retry_codes = (1040, 1213, 2003, 2013)
# 1205 - Lock wait timeout exceeded; try restarting transaction
internal_error_retry_codes = (1205,)


def retry_transient_mysql_errors(f):
    @functools.wraps(f)
    async def wrapper(*args, **kwargs):
        delay = 0.1
        while True:
            try:
                return await f(*args, **kwargs)
            except pymysql.err.InternalError as e:
                if e.args[0] in internal_error_retry_codes:
                    log.warning(
                        f'encountered pymysql error, retrying {e}',
                        exc_info=True,
                        extra={'full_stacktrace': '\n'.join(traceback.format_stack())},
                    )
                else:
                    raise
            except pymysql.err.OperationalError as e:
                if e.args[0] in operational_error_retry_codes:
                    log.warning(
                        f'encountered pymysql error, retrying {e}',
                        exc_info=True,
                        extra={'full_stacktrace': '\n'.join(traceback.format_stack())},
                    )
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


def get_sql_config(maybe_config_file: Optional[str] = None) -> SQLConfig:
    if maybe_config_file is None:
        config_file = os.environ.get('HAIL_DATABASE_CONFIG_FILE', '/sql-config/sql-config.json')
    else:
        config_file = maybe_config_file
    with open(config_file, 'r', encoding='utf-8') as f:
        sql_config = SQLConfig.from_json(f.read())
    sql_config.check()
    log.info('using tls and verifying server certificates for MySQL')
    return sql_config


database_ssl_context = None


def get_database_ssl_context(sql_config: Optional[SQLConfig] = None) -> ssl.SSLContext:
    global database_ssl_context
    if database_ssl_context is None:
        if sql_config is None:
            sql_config = get_sql_config()
        database_ssl_context = ssl.create_default_context(cafile=sql_config.ssl_ca)
        if sql_config.ssl_cert is not None and sql_config.ssl_key is not None:
            database_ssl_context.load_cert_chain(sql_config.ssl_cert, keyfile=sql_config.ssl_key, password=None)
        database_ssl_context.verify_mode = ssl.CERT_REQUIRED
        database_ssl_context.check_hostname = False
    return database_ssl_context


@retry_transient_mysql_errors
async def create_database_pool(config_file: str = None, autocommit: bool = True, maxsize: int = 10):
    sql_config = get_sql_config(config_file)
    ssl_context = get_database_ssl_context(sql_config)
    assert ssl_context is not None
    return await aiomysql.create_pool(
        maxsize=maxsize,
        # connection args
        host=sql_config.host,
        user=sql_config.user,
        password=sql_config.password,
        db=sql_config.db,
        port=sql_config.port,
        charset='utf8',
        ssl=ssl_context,
        cursorclass=aiomysql.cursors.DictCursor,
        autocommit=autocommit,
        # Discard stale connections, see https://stackoverflow.com/questions/69373128/db-connection-issue-with-encode-databases-library.
        pool_recycle=3600,
    )


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


async def _release_connection(conn_context_manager):
    if conn_context_manager is not None:
        try:
            if conn_context_manager._conn is not None:
                await aexit(conn_context_manager)
        except:
            log.exception('while releasing database connection')


class Transaction:
    def __init__(self):
        self.conn_context_manager = None
        self.conn = None

    async def async_init(self, db_pool, read_only):
        try:
            self.conn_context_manager = db_pool.acquire()
            DB_CONNECTION_QUEUE_SIZE.inc()
            SQL_TRANSACTIONS.inc()
            self.conn = await aenter(self.conn_context_manager)
            DB_CONNECTION_QUEUE_SIZE.dec()
            async with self.conn.cursor() as cursor:
                if read_only:
                    await cursor.execute('START TRANSACTION READ ONLY;')
                else:
                    await cursor.execute('START TRANSACTION;')
        except:
            self.conn = None
            conn_context_manager = self.conn_context_manager
            self.conn_context_manager = None
            asyncio.ensure_future(_release_connection(conn_context_manager))
            raise

    async def _aexit_1(self, exc_type):
        try:
            if self.conn is not None:
                if exc_type:
                    await self.conn.rollback()
                else:
                    await self.conn.commit()
        except:
            log.info('while exiting transaction', exc_info=True)
            raise
        finally:
            self.conn = None
            conn_context_manager = self.conn_context_manager
            self.conn_context_manager = None
            asyncio.ensure_future(_release_connection(conn_context_manager))

    async def _aexit(self, exc_type, exc_val, exc_tb):  # pylint: disable=unused-argument
        # cancelling cleanup could leak a connection
        # await shield becuase we want to wait for commit/rollback to finish
        await asyncio.shield(self._aexit_1(exc_type))

    async def just_execute(self, sql, args=None):
        assert self.conn
        async with self.conn.cursor() as cursor:
            await cursor.execute(sql, args)

    async def execute_and_fetchone(self, sql, args=None, query_name=None):
        assert self.conn
        async with self.conn.cursor() as cursor:
            if query_name is None:
                await cursor.execute(sql, args)
            else:
                async with PrometheusSQLTimer(query_name):
                    await cursor.execute(sql, args)
            return await cursor.fetchone()

    async def execute_and_fetchall(self, sql, args=None, query_name=None):
        assert self.conn
        async with self.conn.cursor() as cursor:
            if query_name is None:
                await cursor.execute(sql, args)
            else:
                async with PrometheusSQLTimer(query_name):
                    await cursor.execute(sql, args)
            while True:
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

    async def execute_update(self, sql, args=None, query_name=None):
        assert self.conn
        async with self.conn.cursor() as cursor:
            if query_name is None:
                return await cursor.execute(sql, args)
            async with PrometheusSQLTimer(query_name):
                return await cursor.execute(sql, args)

    async def execute_many(self, sql, args_array, query_name=None):
        assert self.conn
        async with self.conn.cursor() as cursor:
            if query_name is None:
                res = await cursor.executemany(sql, args_array)
            else:
                async with PrometheusSQLTimer(query_name):
                    res = await cursor.executemany(sql, args_array)
            return res


class CallError(Exception):
    def __init__(self, rv):
        super().__init__(rv)
        self.rv = rv


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
    async def execute_and_fetchone(self, sql, args=None, query_name=None):
        async with self.start() as tx:
            return await tx.execute_and_fetchone(sql, args, query_name)

    @retry_transient_mysql_errors
    async def select_and_fetchone(self, sql, args=None, query_name=None):
        async with self.start(read_only=True) as tx:
            return await tx.execute_and_fetchone(sql, args, query_name)

    async def execute_and_fetchall(self, sql, args=None, query_name=None):
        async with self.start() as tx:
            async for row in tx.execute_and_fetchall(sql, args, query_name):
                yield row

    async def select_and_fetchall(self, sql, args=None, query_name=None):
        async with self.start(read_only=True) as tx:
            async for row in tx.execute_and_fetchall(sql, args, query_name):
                yield row

    @retry_transient_mysql_errors
    async def execute_insertone(self, sql, args=None):
        async with self.start() as tx:
            return await tx.execute_insertone(sql, args)

    @retry_transient_mysql_errors
    async def execute_update(self, sql, args=None, query_name=None):
        async with self.start() as tx:
            return await tx.execute_update(sql, args, query_name)

    @retry_transient_mysql_errors
    async def execute_many(self, sql, args_array, query_name=None):
        async with self.start() as tx:
            return await tx.execute_many(sql, args_array, query_name=query_name)

    @retry_transient_mysql_errors
    async def check_call_procedure(self, sql, args=None, query_name=None):
        rv = await self.execute_and_fetchone(sql, args, query_name)
        if rv['rc'] != 0:
            raise CallError(rv)
        return rv

    async def async_close(self):
        self.pool.close()
        await self.pool.wait_closed()
