"""Unit tests for billing_dao against a real MySQL instance.

Requires local MySQL (make local-mysql) and the HAIL_SQL_DATABASE env var,
which the pytest fixture sets automatically.
"""

import os
import warnings

import aiomysql
import pytest
import pytest_asyncio

from batch.billing_dao import create_quote
from batch.exceptions import BatchOperationAlreadyCompletedError
from gear import Database

_TEST_DB = 'test_billing_dao'
_SCHEMA = os.path.join(os.path.dirname(__file__), 'billing_dao_schema.sql')


@pytest_asyncio.fixture(scope='module')
async def db():
    """Create a fresh test database, yield a connected Database, then drop it."""

    async def run_ddl(cur, sql):
        for statement in (s.strip() for s in sql.split(';')):
            if statement:
                await cur.execute(statement)

    async def admin_conn():
        return await aiomysql.connect(host='localhost', port=3306, user='root', password='pw')

    conn = await admin_conn()
    try:
        async with conn.cursor() as cur:
            # Suppress the benign "database doesn't exist" warning from DROP IF EXISTS.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                await cur.execute(f'DROP DATABASE IF EXISTS `{_TEST_DB}`')
            await cur.execute(f'CREATE DATABASE `{_TEST_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci')
            await cur.execute(f'USE `{_TEST_DB}`')
            with open(_SCHEMA, encoding='utf-8') as f:
                await run_ddl(cur, f.read())
        await conn.commit()
    finally:
        conn.close()

    os.environ['HAIL_SQL_DATABASE'] = _TEST_DB
    database = Database()
    await database.async_init()
    yield database
    await database.async_exit_stack.aclose()

    conn = await admin_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(f'DROP DATABASE IF EXISTS `{_TEST_DB}`')
        await conn.commit()
    finally:
        conn.close()


@pytest_asyncio.fixture(autouse=True)
async def clean_tables(db):
    yield
    async with db.start() as tx:
        await tx.just_execute('DELETE FROM quote_events')
        await tx.just_execute('DELETE FROM quotes')


# ---------------------------------------------------------------------------
# create_quote
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_quote_returns_id(db):
    quote_id = await create_quote(db, 'q-basic', cost_object='CO-001', actor='admin')
    assert isinstance(quote_id, int)
    assert quote_id > 0


@pytest.mark.asyncio
async def test_create_quote_persists_fields(db):
    await create_quote(db, 'q-fields', cost_object='CO-002', actor='admin', authorized_amount=5000.0, pi_name='Jane')
    row = await db.select_and_fetchone('SELECT * FROM quotes WHERE name = %s', ('q-fields',))
    assert row['cost_object'] == 'CO-002'
    assert row['authorized_amount'] == 5000.0
    assert row['pi_name'] == 'Jane'


@pytest.mark.asyncio
async def test_create_quote_logs_creation_event(db):
    await create_quote(db, 'q-event', cost_object='CO-003', actor='admin', comment='initial')
    row = await db.select_and_fetchone(
        'SELECT qe.* FROM quote_events qe JOIN quotes q ON q.id = qe.quote_id WHERE q.name = %s',
        ('q-event',),
    )
    assert row['action'] == 'quote_created'
    assert row['actor'] == 'admin'
    assert row['detail'] == 'CO-003'
    assert row['comment'] == 'initial'


@pytest.mark.asyncio
async def test_create_quote_duplicate_raises(db):
    await create_quote(db, 'q-dup', cost_object='CO-004', actor='admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await create_quote(db, 'q-dup', cost_object='CO-004b', actor='admin')


@pytest.mark.asyncio
async def test_create_quote_unlimited_stored_as_null(db):
    await create_quote(db, 'q-unlimited', cost_object='CO-005', actor='admin', authorized_amount=None)
    row = await db.select_and_fetchone('SELECT authorized_amount FROM quotes WHERE name = %s', ('q-unlimited',))
    assert row['authorized_amount'] is None
