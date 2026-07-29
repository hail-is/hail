"""Unit tests for billing_reporting against a real MySQL instance.

Requires local MySQL (make local-mysql) and the HAIL_SQL_DATABASE env var,
which the pytest fixture sets automatically.
"""

import datetime
import os
import warnings

import aiomysql
import pytest
import pytest_asyncio

from batch.billing_project_management import (
    add_billing_project_user,
    add_quote_manager,
    create_billing_project,
    create_quote,
)
from batch.billing_reporting import (
    query_billing_breakdown,
    query_billing_history,
    query_billing_projects_with_cost,
    query_billing_projects_without_cost,
)
from gear import Database

_TEST_DB = 'test_billing_reporting'
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
        await tx.just_execute('DELETE FROM aggregated_billing_project_user_resources_by_date_v3')
        await tx.just_execute('DELETE FROM aggregated_billing_project_user_resources_v3')
        await tx.just_execute('DELETE FROM resources')
        await tx.just_execute('DELETE FROM batches')
        await tx.just_execute('DELETE FROM billing_project_events')
        await tx.just_execute('DELETE FROM billing_project_users')
        await tx.just_execute('DELETE FROM billing_projects')
        await tx.just_execute('DELETE FROM quote_managers')
        await tx.just_execute('DELETE FROM quote_events')
        await tx.just_execute("DELETE FROM quotes WHERE name != 'INTERNAL'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_bp(db, quote_name, bp_name, limit=None):
    await create_quote(db, quote_name, cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (quote_name,))
    await create_billing_project(db, bp_name, q_row['id'], limit, None, 'admin', 'global_bm')
    return q_row['id']


async def _insert_resource(db, resource_name, rate):
    async with db.start() as tx:
        return await tx.execute_insertone(
            'INSERT INTO resources (resource, rate) VALUES (%s, %s)',
            (resource_name, rate),
        )


async def _insert_spend(db, billing_date, billing_project, user, resource_id, usage):
    async with db.start() as tx:
        await tx.just_execute(
            """INSERT INTO aggregated_billing_project_user_resources_by_date_v3
               (billing_date, billing_project, `user`, resource_id, token, `usage`)
               VALUES (%s, %s, %s, %s, 0, %s) AS new_row
               ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_by_date_v3.`usage` + new_row.`usage`""",
            (billing_date, billing_project, user, resource_id, usage),
        )


_JAN1 = datetime.datetime(2026, 1, 1)
_JAN31 = datetime.datetime(2026, 1, 31)
_JAN15 = datetime.date(2026, 1, 15)


# ---------------------------------------------------------------------------
# query_billing_projects_with_cost — users field shape
# ---------------------------------------------------------------------------


async def test_query_bp_users_bp_member(db):
    await _make_bp(db, 'q-users-bpm', 'bp-users-bpm')
    await add_billing_project_user(db, 'bp-users-bpm', 'luna', 'admin')
    results = await query_billing_projects_with_cost(db, billing_project='bp-users-bpm')
    assert len(results) == 1
    users = results[0]['users']
    luna = next((u for u in users if u['user'] == 'luna'), None)
    assert luna is not None
    assert 'bp-users-bpm:member' in luna['roles']


async def test_query_bp_users_quote_manager_appears_without_direct_membership(db):
    await _make_bp(db, 'q-users-qm', 'bp-users-qm')
    await add_quote_manager(db, 'q-users-qm', 'mars', 'owner', actor='admin')
    results = await query_billing_projects_with_cost(db, billing_project='bp-users-qm')
    assert len(results) == 1
    users = results[0]['users']
    mars = next((u for u in users if u['user'] == 'mars'), None)
    assert mars is not None
    assert 'q-users-qm:owner' in mars['roles']


async def test_query_bp_users_dual_role(db):
    await _make_bp(db, 'q-users-dual', 'bp-users-dual')
    await add_quote_manager(db, 'q-users-dual', 'nova', 'manager', actor='admin')
    await add_billing_project_user(db, 'bp-users-dual', 'nova', 'admin')
    results = await query_billing_projects_with_cost(db, billing_project='bp-users-dual')
    assert len(results) == 1
    users = results[0]['users']
    nova = next((u for u in users if u['user'] == 'nova'), None)
    assert nova is not None
    assert 'bp-users-dual:member' in nova['roles']
    assert 'q-users-dual:manager' in nova['roles']


async def test_query_bp_users_outsider_absent(db):
    await _make_bp(db, 'q-users-out', 'bp-users-out')
    results = await query_billing_projects_with_cost(db, billing_project='bp-users-out')
    assert len(results) == 1
    users = results[0]['users']
    assert not any(u['user'] == 'outsider' for u in users)


async def test_query_billing_projects_without_cost_basic(db):
    await _make_bp(db, 'q-woc', 'bp-woc')
    results = await query_billing_projects_without_cost(db, billing_project='bp-woc')
    assert len(results) == 1
    assert results[0]['billing_project'] == 'bp-woc'


# ---------------------------------------------------------------------------
# query_billing_history — access control
# ---------------------------------------------------------------------------


async def test_billing_history_global_bm_sees_all(db):
    await _make_bp(db, 'q-hist-all', 'bp-hist-all')
    resource_id = await _insert_resource(db, 'cpu', 0.01)
    await _insert_spend(db, _JAN15, 'bp-hist-all', 'alice', resource_id, 100)
    await _insert_spend(db, _JAN15, 'bp-hist-all', 'bob', resource_id, 200)

    rows = await query_billing_history(db, _JAN1, _JAN31, user=None, quote_manager_user=None)
    users_seen = {r['user'] for r in rows if r['billing_project'] == 'bp-hist-all'}
    assert 'alice' in users_seen
    assert 'bob' in users_seen


async def test_billing_history_regular_user_sees_only_own_spend(db):
    await _make_bp(db, 'q-hist-own', 'bp-hist-own')
    resource_id = await _insert_resource(db, 'cpu-own', 0.01)
    await _insert_spend(db, _JAN15, 'bp-hist-own', 'alice', resource_id, 100)
    await _insert_spend(db, _JAN15, 'bp-hist-own', 'bob', resource_id, 200)

    rows = await query_billing_history(db, _JAN1, _JAN31, user='alice', quote_manager_user=None)
    users_seen = {r['user'] for r in rows}
    assert 'alice' in users_seen
    assert 'bob' not in users_seen


async def test_billing_history_quote_manager_sees_all_spend_in_managed_quote(db):
    await _make_bp(db, 'q-hist-qm', 'bp-hist-qm')
    await add_quote_manager(db, 'q-hist-qm', 'manager', 'manager', actor='admin')
    resource_id = await _insert_resource(db, 'cpu-qm', 0.01)
    await _insert_spend(db, _JAN15, 'bp-hist-qm', 'alice', resource_id, 100)
    await _insert_spend(db, _JAN15, 'bp-hist-qm', 'bob', resource_id, 200)

    rows = await query_billing_history(db, _JAN1, _JAN31, user='manager', quote_manager_user='manager')
    users_seen = {r['user'] for r in rows if r['billing_project'] == 'bp-hist-qm'}
    assert 'alice' in users_seen
    assert 'bob' in users_seen


async def test_billing_history_quote_manager_also_sees_own_spend_in_other_projects(db):
    await _make_bp(db, 'q-hist-other', 'bp-hist-other')
    resource_id = await _insert_resource(db, 'cpu-other', 0.01)
    await _insert_spend(db, _JAN15, 'bp-hist-other', 'manager', resource_id, 50)

    rows = await query_billing_history(db, _JAN1, _JAN31, user='manager', quote_manager_user='manager')
    projects_seen = {r['billing_project'] for r in rows if r['user'] == 'manager'}
    assert 'bp-hist-other' in projects_seen


async def test_billing_history_non_manager_cannot_see_others_spend(db):
    await _make_bp(db, 'q-hist-excl', 'bp-hist-excl')
    resource_id = await _insert_resource(db, 'cpu-excl', 0.01)
    await _insert_spend(db, _JAN15, 'bp-hist-excl', 'alice', resource_id, 100)

    rows = await query_billing_history(db, _JAN1, _JAN31, user='outsider', quote_manager_user=None)
    users_seen = {r['user'] for r in rows if r['billing_project'] == 'bp-hist-excl'}
    assert 'alice' not in users_seen


async def test_billing_history_date_filter_applied(db):
    await _make_bp(db, 'q-hist-date', 'bp-hist-date')
    resource_id = await _insert_resource(db, 'cpu-date', 0.01)
    await _insert_spend(db, datetime.date(2025, 12, 31), 'bp-hist-date', 'alice', resource_id, 999)
    await _insert_spend(db, _JAN15, 'bp-hist-date', 'alice', resource_id, 100)

    rows = await query_billing_history(db, _JAN1, _JAN31, user='alice', quote_manager_user=None)
    total = sum(r['cost'] for r in rows if r['billing_project'] == 'bp-hist-date')
    assert pytest.approx(total, abs=0.01) == 100 * 0.01


# ---------------------------------------------------------------------------
# query_billing_breakdown — access control mirrors query_billing_history
# ---------------------------------------------------------------------------


async def test_billing_breakdown_regular_user_sees_only_own(db):
    await _make_bp(db, 'q-brkdn-own', 'bp-brkdn-own')
    resource_id = await _insert_resource(db, 'mem-own', 0.005)
    await _insert_spend(db, _JAN15, 'bp-brkdn-own', 'alice', resource_id, 100)
    await _insert_spend(db, _JAN15, 'bp-brkdn-own', 'bob', resource_id, 200)

    rows = await query_billing_breakdown(db, _JAN1, _JAN31, user='alice', quote_manager_user=None)
    users_seen = {r['user'] for r in rows}
    assert 'alice' in users_seen
    assert 'bob' not in users_seen


async def test_billing_breakdown_quote_manager_sees_all_in_quote(db):
    await _make_bp(db, 'q-brkdn-qm', 'bp-brkdn-qm')
    await add_quote_manager(db, 'q-brkdn-qm', 'mgr', 'manager', actor='admin')
    resource_id = await _insert_resource(db, 'mem-qm', 0.005)
    await _insert_spend(db, _JAN15, 'bp-brkdn-qm', 'alice', resource_id, 100)
    await _insert_spend(db, _JAN15, 'bp-brkdn-qm', 'bob', resource_id, 200)

    rows = await query_billing_breakdown(db, _JAN1, _JAN31, user='mgr', quote_manager_user='mgr')
    users_seen = {r['user'] for r in rows if r['billing_project'] == 'bp-brkdn-qm'}
    assert 'alice' in users_seen
    assert 'bob' in users_seen
