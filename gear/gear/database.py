import asyncio
import functools
import logging
import os
import ssl
import traceback
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, TypeVar

import aiomysql
import kubernetes_asyncio.client
import kubernetes_asyncio.config
import pymysql
from typing_extensions import Concatenate, ParamSpec

from gear.metrics import DB_CONNECTION_QUEUE_SIZE, SQL_TRANSACTIONS, PrometheusSQLTimer
from hailtop.aiotools import BackgroundTaskManager
from hailtop.auth.sql_config import SQLConfig
from hailtop.config import get_deploy_config
from hailtop.utils import first_extant_file, sleep_before_try

log = logging.getLogger('gear.database')


# 1040 - Too many connections
# 1213 - Deadlock found when trying to get lock; try restarting transaction
# 2003 - Can't connect to MySQL server on ...
# 2013 - Lost connection to MySQL server during query ([Errno 104] Connection reset by peer)
operational_error_retry_codes = (1040, 1213, 2003, 2013)
# 1205 - Lock wait timeout exceeded; try restarting transaction
internal_error_retry_codes = (1205,)


T = TypeVar("T")
P = ParamSpec('P')


def retry_transient_mysql_errors(f: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    @functools.wraps(f)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        tries = 0
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
            tries += 1
            await sleep_before_try(tries)

    return wrapper


def transaction(db: 'Database', read_only: bool = False):
    def transformer(fun: Callable[Concatenate['Transaction', P], Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(fun)
        @retry_transient_mysql_errors
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with db.start(read_only=read_only) as tx:
                return await fun(tx, *args, **kwargs)

        return wrapper

    return transformer


async def aenter(acontext_manager):
    return await acontext_manager.__aenter__()  # pylint: disable=unnecessary-dunder-call


async def aexit(acontext_manager, exc_type=None, exc_val=None, exc_tb=None):
    return await acontext_manager.__aexit__(exc_type, exc_val, exc_tb)


async def resolve_test_db_endpoint(sql_config: SQLConfig) -> SQLConfig:
    service_name, namespace = sql_config.host[: -len('.svc.cluster.local')].split('.', maxsplit=1)
    await kubernetes_asyncio.config.load_kube_config()
    async with kubernetes_asyncio.client.ApiClient() as api:
        client = kubernetes_asyncio.client.CoreV1Api(api)
        db_service = await client.read_namespaced_service(service_name, namespace)  # type: ignore
        db_pod = await client.read_namespaced_pod(f'{db_service.spec.selector["app"]}-0', namespace)  # type: ignore
        sql_config_dict = sql_config.to_dict()
        sql_config_dict['host'] = db_pod.status.host_ip  # type: ignore
        sql_config_dict['port'] = db_service.spec.ports[0].node_port  # type: ignore
        return SQLConfig.from_dict(sql_config_dict)


def get_sql_config(maybe_config_file: Optional[str] = None) -> SQLConfig:
    config_file = first_extant_file(
        maybe_config_file,
        os.environ.get('HAIL_DATABASE_CONFIG_FILE'),
        '/sql-config/sql-config.json',
    )
    if config_file is not None:
        with open(config_file, 'r', encoding='utf-8') as f:
            sql_config = SQLConfig.from_json(f.read())
    else:
        sql_config = SQLConfig.local_insecure_config()
    sql_config.check()
    log.info('using tls and verifying server certificates for MySQL')
    return sql_config


def get_database_ssl_context(sql_config: SQLConfig) -> ssl.SSLContext:
    database_ssl_context = ssl.create_default_context(cafile=sql_config.ssl_ca)
    if sql_config.ssl_cert is not None and sql_config.ssl_key is not None:
        database_ssl_context.load_cert_chain(sql_config.ssl_cert, keyfile=sql_config.ssl_key, password=None)
    database_ssl_context.verify_mode = ssl.CERT_REQUIRED
    database_ssl_context.check_hostname = False
    return database_ssl_context


@retry_transient_mysql_errors
async def create_database_pool(
    config_file: Optional[str] = None, autocommit: bool = True, maxsize: int = 10
) -> aiomysql.Pool:
    sql_config = get_sql_config(config_file)
    if get_deploy_config().location() != 'k8s' and sql_config.host.endswith('svc.cluster.local'):
        sql_config = await resolve_test_db_endpoint(sql_config)
    if sql_config.host == 'localhost':
        ssl_context = None
    else:
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
    def __init__(self, db_pool, read_only, task_manager: BackgroundTaskManager):
        self.db_pool = db_pool
        self.read_only = read_only
        self.tx: Optional['Transaction'] = None
        self.task_manager = task_manager

    async def __aenter__(self):
        tx = Transaction(self.task_manager)
        await tx.async_init(self.db_pool, self.read_only)
        self.tx = tx
        return tx

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        assert self.tx is not None
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
    def __init__(self, task_manager: BackgroundTaskManager):
        self.conn_context_manager = None
        self.conn = None
        self._task_manager = task_manager

    async def async_init(self, db_pool, read_only):
        try:
            self.conn_context_manager = db_pool.acquire()
            DB_CONNECTION_QUEUE_SIZE.inc()
            SQL_TRANSACTIONS.inc()
            self.conn = await aenter(self.conn_context_manager)
            assert self.conn is not None
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
            self._task_manager.ensure_future(_release_connection(conn_context_manager))
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
            self._task_manager.ensure_future(_release_connection(conn_context_manager))

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

    async def execute_and_fetchall(self, sql: str, args=None, query_name=None) -> AsyncIterator[Dict[str, Any]]:
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

    async def execute_insertone(self, sql, args=None, *, query_name=None) -> Optional[int]:
        assert self.conn
        async with self.conn.cursor() as cursor:
            if query_name is None:
                await cursor.execute(sql, args)
            else:
                async with PrometheusSQLTimer(query_name):
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
        self.connection_release_task_manager = None

    async def async_init(self, config_file=None, maxsize=10):
        self.pool = await create_database_pool(config_file=config_file, autocommit=False, maxsize=maxsize)
        self.connection_release_task_manager = BackgroundTaskManager()

    def start(self, read_only=False):
        assert self.connection_release_task_manager
        return TransactionAsyncContextManager(self.pool, read_only, self.connection_release_task_manager)

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
        assert self.pool
        assert self.connection_release_task_manager
        self.connection_release_task_manager.shutdown()
        self.pool.close()
        await self.pool.wait_closed()
