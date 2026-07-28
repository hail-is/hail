"""Unit tests for billing_dao against a real MySQL instance.

Requires local MySQL (make local-mysql) and the HAIL_SQL_DATABASE env var,
which the pytest fixture sets automatically.
"""

import json
import os
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import aiomysql
import pytest
import pytest_asyncio
from aiohttp import web

from batch.billing_auth import BillingPermission, billing_permission_required
from batch.billing_dao import (
    add_billing_project_user,
    add_quote_manager,
    change_billing_project_quote,
    close_billing_project,
    close_quote,
    create_billing_project,
    create_quote,
    delete_billing_project,
    edit_quote,
    get_billing_project_events,
    get_billing_role_for_bp,
    get_billing_role_for_quote,
    get_billing_role_for_quote_id,
    get_quote,
    get_quote_events,
    list_quotes_for_user,
    patch_billing_project,
    remove_billing_project_user,
    remove_quote_manager,
    reopen_billing_project,
)
from batch.exceptions import BatchOperationAlreadyCompletedError, BatchUserError
from batch.utils import query_billing_projects_with_cost
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
        await tx.just_execute('DELETE FROM batches')
        await tx.just_execute('DELETE FROM billing_project_events')
        await tx.just_execute('DELETE FROM billing_project_users')
        await tx.just_execute('DELETE FROM billing_projects')
        await tx.just_execute('DELETE FROM quote_managers')
        await tx.just_execute('DELETE FROM quote_events')
        await tx.just_execute("DELETE FROM quotes WHERE name != 'INTERNAL'")


# ---------------------------------------------------------------------------
# create_quote
# ---------------------------------------------------------------------------


async def test_create_quote_returns_id(db):
    quote_id = await create_quote(db, 'q-basic', cost_object='CO-001', actor='admin')
    assert isinstance(quote_id, int)
    assert quote_id > 0


async def test_create_quote_persists_fields(db):
    await create_quote(db, 'q-fields', cost_object='CO-002', actor='admin', authorized_amount=5000.0, pi_name='Jane')
    row = await db.select_and_fetchone('SELECT * FROM quotes WHERE name = %s', ('q-fields',))
    assert row['cost_object'] == 'CO-002'
    assert row['authorized_amount'] == 5000.0
    assert row['pi_name'] == 'Jane'


async def test_create_quote_logs_creation_event(db):
    await create_quote(db, 'q-event', cost_object='CO-003', actor='admin', comment='initial')
    row = await db.select_and_fetchone(
        'SELECT qe.* FROM quote_events qe JOIN quotes q ON q.id = qe.quote_id WHERE q.name = %s',
        ('q-event',),
    )
    assert row['action'] == 'quote_created'
    assert row['actor'] == 'admin'
    assert json.loads(row['detail']) == 'CO-003'
    assert row['comment'] == 'initial'


async def test_create_quote_duplicate_raises(db):
    await create_quote(db, 'q-dup', cost_object='CO-004', actor='admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await create_quote(db, 'q-dup', cost_object='CO-004b', actor='admin')


async def test_create_quote_unlimited_stored_as_null(db):
    await create_quote(db, 'q-unlimited', cost_object='CO-005', actor='admin', authorized_amount=None)
    row = await db.select_and_fetchone('SELECT authorized_amount FROM quotes WHERE name = %s', ('q-unlimited',))
    assert row['authorized_amount'] is None


async def test_create_quote_persists_description(db):
    await create_quote(db, 'q-desc', cost_object='CO', actor='admin', description='My test quote')
    row = await db.select_and_fetchone('SELECT description FROM quotes WHERE name = %s', ('q-desc',))
    assert row['description'] == 'My test quote'


async def test_create_quote_description_defaults_null(db):
    await create_quote(db, 'q-no-desc', cost_object='CO', actor='admin')
    row = await db.select_and_fetchone('SELECT description FROM quotes WHERE name = %s', ('q-no-desc',))
    assert row['description'] is None


async def test_create_quote_persists_quote_number(db):
    await create_quote(db, 'q-num', cost_object='CO', actor='admin', quote_number='Q-2026-001')
    row = await db.select_and_fetchone('SELECT quote_number FROM quotes WHERE name = %s', ('q-num',))
    assert row['quote_number'] == 'Q-2026-001'


async def test_create_quote_quote_number_defaults_null(db):
    await create_quote(db, 'q-no-num', cost_object='CO', actor='admin')
    row = await db.select_and_fetchone('SELECT quote_number FROM quotes WHERE name = %s', ('q-no-num',))
    assert row['quote_number'] is None


# ---------------------------------------------------------------------------
# list_quotes_for_user
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'is_global_bm,add_manager,expected_count_delta',
    [
        pytest.param(True, False, 1, id='global_bm_sees_all'),
        pytest.param(False, True, 1, id='member_sees_own'),
        pytest.param(False, False, 0, id='non_member_sees_nothing'),
    ],
)
async def test_list_quotes_for_user(db, is_global_bm, add_manager, expected_count_delta):
    before = await list_quotes_for_user(db, 'alice', is_global_bm)
    await create_quote(db, 'q-list', cost_object='co', actor='admin')
    if add_manager:
        await add_quote_manager(db, 'q-list', 'alice', 'manager', actor='admin')
    after = await list_quotes_for_user(db, 'alice', is_global_bm)
    assert len(after) - len(before) == expected_count_delta


async def test_list_quotes_global_bm_sees_internal(db):
    quotes = await list_quotes_for_user(db, 'someone', True)
    names = [q['name'] for q in quotes]
    assert 'INTERNAL' in names


# ---------------------------------------------------------------------------
# get_quote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'kwargs,expected',
    [
        pytest.param({'authorized_amount': 500.0}, {'authorized_amount': 500.0}, id='with_amount'),
        pytest.param({}, {'authorized_amount': None}, id='unlimited'),
    ],
)
async def test_get_quote_fields(db, kwargs, expected):
    await create_quote(db, 'q-get', cost_object='CO', actor='admin', **kwargs)
    q = await get_quote(db, 'q-get')
    assert q is not None
    for key, val in expected.items():
        assert q[key] == val


async def test_get_quote_includes_managers(db):
    await create_quote(db, 'q-mgrs', cost_object='CO', actor='admin')
    await add_quote_manager(db, 'q-mgrs', 'bob', 'owner', actor='admin')
    q = await get_quote(db, 'q-mgrs')
    assert q is not None
    assert any(m['user'] == 'bob' and m['role'] == 'owner' for m in q['managers'])


async def test_get_quote_includes_billing_projects(db):
    await create_quote(db, 'q-bps', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-bps',))
    await create_billing_project(db, 'bp-under-q', q_row['id'], 100.0, None, 'admin', 'global_bm')
    q = await get_quote(db, 'q-bps')
    assert q is not None
    bp_names = [bp['billing_project'] for bp in q['billing_projects']]
    assert 'bp-under-q' in bp_names


async def test_get_quote_returns_none_if_missing(db):
    result = await get_quote(db, 'no-such-quote')
    assert result is None


# ---------------------------------------------------------------------------
# edit_quote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'field,new_value',
    [
        pytest.param('cost_object', 'UPDATED', id='cost_object'),
        pytest.param('pi_name', 'Dr. Jones', id='pi_name'),
        pytest.param('pm_designee', 'pm@acme.com', id='pm_designee'),
        pytest.param('description', 'updated desc', id='description'),
        pytest.param('authorized_amount', 750.0, id='authorized_amount'),
        pytest.param('quote_number', 'Q-2026-042', id='quote_number'),
    ],
)
async def test_edit_quote_field(db, field, new_value):
    await create_quote(db, 'q-edit', cost_object='original', actor='admin', authorized_amount=1000.0)
    await edit_quote(db, 'q-edit', {field: new_value}, actor='admin', billing_role='global_bm')
    row = await db.select_and_fetchone(f'SELECT `{field}` FROM quotes WHERE name = %s', ('q-edit',))
    assert row[field] == new_value


async def test_edit_quote_logs_event(db):
    await create_quote(db, 'q-edit-log', cost_object='CO', actor='admin')
    await edit_quote(db, 'q-edit-log', {'cost_object': 'NEW'}, actor='admin', billing_role='global_bm', comment='upd')
    events = [
        r
        async for r in db.select_and_fetchall(
            'SELECT action, comment, detail FROM quote_events qe JOIN quotes q ON q.id = qe.quote_id WHERE q.name = %s',
            ('q-edit-log',),
        )
    ]
    actions = [e['action'] for e in events]
    assert 'quote_edited' in actions
    edit_event = next(e for e in events if e['action'] == 'quote_edited')
    detail = json.loads(edit_event['detail'])
    assert detail['cost_object'] == {'old': 'CO', 'new': 'NEW'}


async def test_edit_quote_rejects_amount_below_bp_limits(db):
    await create_quote(db, 'q-cap', cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-cap',))
    await create_billing_project(db, 'bp-cap', q_row['id'], 400.0, None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='less than sum of BP limits'):
        await edit_quote(db, 'q-cap', {'authorized_amount': 300.0}, actor='admin', billing_role='global_bm')


async def test_edit_quote_rejects_finite_amount_when_unlimited_bp_exists(db):
    # A quote containing an unlimited BP cannot be given a finite cap — the BP
    # could keep accruing charges beyond the cap.
    await create_quote(db, 'q-ul-bp', cost_object='CO', actor='admin', authorized_amount=None)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-ul-bp',))
    await create_billing_project(db, 'bp-ul', q_row['id'], None, None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='unlimited billing project'):
        await edit_quote(db, 'q-ul-bp', {'authorized_amount': 1000.0}, actor='admin', billing_role='global_bm')


async def test_edit_quote_unlimited_requires_global_bm(db):
    await create_quote(db, 'q-ul-auth', cost_object='CO', actor='admin', authorized_amount=500.0)
    with pytest.raises(BatchUserError, match='Only global billing managers'):
        await edit_quote(db, 'q-ul-auth', {'authorized_amount': None}, actor='admin', billing_role='quote_owner')


async def test_edit_quote_global_bm_can_set_unlimited(db):
    await create_quote(db, 'q-ul-ok', cost_object='CO', actor='admin', authorized_amount=500.0)
    await edit_quote(db, 'q-ul-ok', {'authorized_amount': None}, actor='admin', billing_role='global_bm')
    row = await db.select_and_fetchone('SELECT authorized_amount FROM quotes WHERE name = %s', ('q-ul-ok',))
    assert row['authorized_amount'] is None


# ---------------------------------------------------------------------------
# close_quote
# ---------------------------------------------------------------------------


async def test_close_quote_sets_state(db):
    await create_quote(db, 'q-close', cost_object='CO', actor='admin')
    await close_quote(db, 'q-close', actor='admin')
    row = await db.select_and_fetchone('SELECT state FROM quotes WHERE name = %s', ('q-close',))
    assert row['state'] == 'closed'


async def test_close_quote_logs_event(db):
    await create_quote(db, 'q-close-log', cost_object='CO', actor='admin')
    await close_quote(db, 'q-close-log', actor='admin', comment='done')
    events = [
        r
        async for r in db.select_and_fetchall(
            'SELECT action, comment FROM quote_events qe JOIN quotes q ON q.id = qe.quote_id WHERE q.name = %s',
            ('q-close-log',),
        )
    ]
    actions = [e['action'] for e in events]
    assert 'quote_closed' in actions
    close_event = next(e for e in events if e['action'] == 'quote_closed')
    assert close_event['comment'] == 'done'


async def test_close_quote_already_closed_raises(db):
    await create_quote(db, 'q-close-dup', cost_object='CO', actor='admin')
    await close_quote(db, 'q-close-dup', actor='admin')
    with pytest.raises(BatchOperationAlreadyCompletedError, match='already closed'):
        await close_quote(db, 'q-close-dup', actor='admin')


async def test_close_quote_unknown_raises(db):
    with pytest.raises(BatchUserError, match='Unknown quote'):
        await close_quote(db, 'no-such-quote', actor='admin')


async def test_close_quote_blocked_by_open_bp(db):
    await create_quote(db, 'q-open-bp', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-open-bp',))
    await create_billing_project(db, 'bp-open', q_row['id'], None, None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='bp-open'):
        await close_quote(db, 'q-open-bp', actor='admin')


async def test_close_quote_allowed_when_all_bps_closed(db):
    await create_quote(db, 'q-all-closed', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-all-closed',))
    await create_billing_project(db, 'bp-cl', q_row['id'], None, None, 'admin', 'global_bm')
    await close_billing_project(db, 'bp-cl', actor='admin')
    await close_quote(db, 'q-all-closed', actor='admin')
    row = await db.select_and_fetchone('SELECT state FROM quotes WHERE name = %s', ('q-all-closed',))
    assert row['state'] == 'closed'


async def test_close_quote_allowed_when_bps_deleted(db):
    await create_quote(db, 'q-bp-deleted', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-bp-deleted',))
    await create_billing_project(db, 'bp-del', q_row['id'], None, None, 'admin', 'global_bm')
    await close_billing_project(db, 'bp-del', actor='admin')
    await delete_billing_project(db, 'bp-del')
    await close_quote(db, 'q-bp-deleted', actor='admin')
    row = await db.select_and_fetchone('SELECT state FROM quotes WHERE name = %s', ('q-bp-deleted',))
    assert row['state'] == 'closed'


async def test_get_quote_includes_state(db):
    await create_quote(db, 'q-state-field', cost_object='CO', actor='admin')
    q = await get_quote(db, 'q-state-field')
    assert q is not None
    assert q['state'] == 'open'
    await close_quote(db, 'q-state-field', actor='admin')
    q2 = await get_quote(db, 'q-state-field')
    assert q2 is not None
    assert q2['state'] == 'closed'


# ---------------------------------------------------------------------------
# add_quote_manager / remove_quote_manager
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('role', ['owner', 'manager'])
async def test_add_quote_manager_role(db, role):
    await create_quote(db, f'q-role-{role}', cost_object='CO', actor='admin')
    await add_quote_manager(db, f'q-role-{role}', 'carol', role, actor='admin')
    row = await db.select_and_fetchone(
        'SELECT qm.role FROM quote_managers qm JOIN quotes q ON q.id = qm.quote_id WHERE q.name = %s AND qm.user = %s',
        (f'q-role-{role}', 'carol'),
    )
    assert row['role'] == role


async def test_add_quote_manager_duplicate_raises(db):
    await create_quote(db, 'q-dup-mgr', cost_object='CO', actor='admin')
    await add_quote_manager(db, 'q-dup-mgr', 'dave', 'manager', actor='admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await add_quote_manager(db, 'q-dup-mgr', 'dave', 'owner', actor='admin')


async def test_remove_quote_manager(db):
    await create_quote(db, 'q-rm-mgr', cost_object='CO', actor='admin')
    await add_quote_manager(db, 'q-rm-mgr', 'eve', 'manager', actor='admin')
    await remove_quote_manager(db, 'q-rm-mgr', 'eve', actor='admin')
    row = await db.select_and_fetchone(
        'SELECT qm.role FROM quote_managers qm JOIN quotes q ON q.id = qm.quote_id WHERE q.name = %s AND qm.user = %s',
        ('q-rm-mgr', 'eve'),
    )
    assert row is None


async def test_remove_quote_manager_not_member_raises(db):
    await create_quote(db, 'q-rm-nonmember', cost_object='CO', actor='admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await remove_quote_manager(db, 'q-rm-nonmember', 'nobody', actor='admin')


async def test_add_remove_quote_manager_events_written(db):
    await create_quote(db, 'q-mgr-events', cost_object='CO', actor='admin')
    await add_quote_manager(db, 'q-mgr-events', 'frank', 'manager', actor='admin')
    await remove_quote_manager(db, 'q-mgr-events', 'frank', actor='admin')
    events = [
        r
        async for r in db.select_and_fetchall(
            'SELECT action, detail FROM quote_events qe JOIN quotes q ON q.id = qe.quote_id WHERE q.name = %s',
            ('q-mgr-events',),
        )
    ]
    actions = {e['action'] for e in events}
    assert 'manager_added' in actions
    assert 'manager_removed' in actions
    added = next(e for e in events if e['action'] == 'manager_added')
    assert json.loads(added['detail']) == 'manager'
    removed = next(e for e in events if e['action'] == 'manager_removed')
    assert json.loads(removed['detail']) == 'manager'


# ---------------------------------------------------------------------------
# get_quote_events
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'actions',
    [
        pytest.param(['quote_created'], id='single'),
        pytest.param(['quote_created', 'manager_added', 'manager_removed'], id='multiple'),
    ],
)
async def test_get_quote_events(db, actions):
    await create_quote(db, 'q-events', cost_object='CO', actor='admin')
    if 'manager_added' in actions or 'manager_removed' in actions:
        await add_quote_manager(db, 'q-events', 'grace', 'manager', actor='admin')
    if 'manager_removed' in actions:
        await remove_quote_manager(db, 'q-events', 'grace', actor='admin')
    events = await get_quote_events(db, 'q-events')
    assert events is not None
    event_actions = {e['action'] for e in events}
    for action in actions:
        assert action in event_actions


# ---------------------------------------------------------------------------
# Role resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'db_role,expected_role',
    [
        pytest.param('owner', 'quote_owner', id='owner'),
        pytest.param('manager', 'quote_manager', id='manager'),
    ],
)
async def test_get_billing_role_for_quote_manager(db, db_role, expected_role):
    await create_quote(db, 'q-role', cost_object='CO', actor='admin')
    await add_quote_manager(db, 'q-role', 'hank', db_role, actor='admin')
    role = await get_billing_role_for_quote(db, 'hank', False, 'q-role')
    assert role == expected_role


async def test_get_billing_role_for_quote_non_member(db):
    await create_quote(db, 'q-non-member', cost_object='CO', actor='admin')
    role = await get_billing_role_for_quote(db, 'outsider', False, 'q-non-member')
    assert role is None


async def test_get_billing_role_for_quote_global_bm_bypasses_db(db):
    role = await get_billing_role_for_quote(db, 'anyone', True, 'nonexistent-quote')
    assert role == 'global_bm'


async def test_get_billing_role_for_bp_member(db):
    await create_quote(db, 'q-bp-role', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-bp-role',))
    await create_billing_project(db, 'bp-role-test', q_row['id'], 50.0, None, 'admin', 'global_bm')
    async with db.start() as tx:
        await tx.execute_insertone(
            'INSERT INTO billing_project_users(billing_project, user, user_cs) VALUES (%s, %s, %s)',
            ('bp-role-test', 'irene', 'irene'),
        )
    role = await get_billing_role_for_bp(db, 'irene', False, 'bp-role-test')
    assert role == 'bp_member'


async def test_get_billing_role_for_bp_via_quote_manager(db):
    await create_quote(db, 'q-bp-qm', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-bp-qm',))
    await create_billing_project(db, 'bp-via-qm', q_row['id'], 50.0, None, 'admin', 'global_bm')
    await add_quote_manager(db, 'q-bp-qm', 'jack', 'owner', actor='admin')
    role = await get_billing_role_for_bp(db, 'jack', False, 'bp-via-qm')
    assert role == 'quote_owner'


async def test_get_billing_role_for_bp_non_member(db):
    await create_quote(db, 'q-bp-nm', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-bp-nm',))
    await create_billing_project(db, 'bp-non-member', q_row['id'], 50.0, None, 'admin', 'global_bm')
    role = await get_billing_role_for_bp(db, 'outsider', False, 'bp-non-member')
    assert role is None


async def test_get_billing_role_for_quote_id(db):
    await create_quote(db, 'q-id-role', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-id-role',))
    await add_quote_manager(db, 'q-id-role', 'kim', 'manager', actor='admin')
    role = await get_billing_role_for_quote_id(db, 'kim', False, q_row['id'])
    assert role == 'quote_manager'


# ---------------------------------------------------------------------------
# create_billing_project
# ---------------------------------------------------------------------------


async def test_create_billing_project_appears_in_quote(db):
    await create_quote(db, 'q-create-bp', cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-create-bp',))
    await create_billing_project(db, 'bp-new', q_row['id'], 100.0, None, 'admin', 'global_bm')
    q = await get_quote(db, 'q-create-bp')
    assert q is not None
    bp_names = [bp['billing_project'] for bp in q['billing_projects']]
    assert 'bp-new' in bp_names


async def test_create_billing_project_limit_exceeds_quote_raises(db):
    await create_quote(db, 'q-exceed', cost_object='CO', actor='admin', authorized_amount=200.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-exceed',))
    await create_billing_project(db, 'bp-exceed-1', q_row['id'], 150.0, None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='exceed quote authorized amount'):
        await create_billing_project(db, 'bp-exceed-2', q_row['id'], 100.0, None, 'admin', 'global_bm')


async def test_create_billing_project_duplicate_raises(db):
    await create_quote(db, 'q-dup-bp', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-dup-bp',))
    await create_billing_project(db, 'bp-dup', q_row['id'], None, None, 'admin', 'global_bm')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await create_billing_project(db, 'bp-dup', q_row['id'], None, None, 'admin', 'global_bm')


async def test_create_billing_project_logs_event(db):
    await create_quote(db, 'q-log-bp', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-log-bp',))
    await create_billing_project(db, 'bp-log', q_row['id'], 50.0, 25.0, 'admin', 'global_bm', comment='init')
    bp_events = await get_billing_project_events(db, 'bp-log')
    assert 'bp_created' in {e['action'] for e in bp_events}
    created_event = next(e for e in bp_events if e['action'] == 'bp_created')
    detail = json.loads(created_event['detail'])
    assert detail == {'limit': 50.0, 'low_budget_alert': 25.0}
    quote_events = await get_quote_events(db, 'q-log-bp')
    assert quote_events is not None
    assert any(e['action'] == 'bp_created' and e['target_project'] == 'bp-log' for e in quote_events)


async def test_create_billing_project_persists_description(db):
    await create_quote(db, 'q-bp-desc', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-bp-desc',))
    await create_billing_project(db, 'bp-desc', q_row['id'], None, None, 'admin', 'global_bm', description='A test BP')
    row = await db.select_and_fetchone('SELECT description FROM billing_projects WHERE name = %s', ('bp-desc',))
    assert row['description'] == 'A test BP'


# ---------------------------------------------------------------------------
# patch_billing_project
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'updates,col,expected',
    [
        pytest.param({'limit': 200.0}, 'limit', 200.0, id='limit'),
        pytest.param({'low_budget_alert': 50.0}, 'low_budget_alert', 50.0, id='alert'),
    ],
)
async def test_patch_billing_project_field(db, updates, col, expected):
    await create_quote(db, 'q-patch', cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-patch',))
    await create_billing_project(db, 'bp-patch', q_row['id'], 100.0, None, 'admin', 'global_bm')
    await patch_billing_project(db, 'bp-patch', updates, actor='admin', billing_role='global_bm')
    row = await db.select_and_fetchone(f'SELECT `{col}` FROM billing_projects WHERE name = %s', ('bp-patch',))
    assert row[col] == expected


async def test_patch_billing_project_limit_exceeds_quote_raises(db):
    await create_quote(db, 'q-patch-cap', cost_object='CO', actor='admin', authorized_amount=300.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-patch-cap',))
    await create_billing_project(db, 'bp-patch-1', q_row['id'], 200.0, None, 'admin', 'global_bm')
    await create_billing_project(db, 'bp-patch-2', q_row['id'], 50.0, None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='would exceed quote authorized amount'):
        await patch_billing_project(db, 'bp-patch-2', {'limit': 200.0}, actor='admin', billing_role='global_bm')


async def test_patch_billing_project_unlimited_requires_global_bm(db):
    await create_quote(db, 'q-patch-ul', cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-patch-ul',))
    await create_billing_project(db, 'bp-patch-ul', q_row['id'], 100.0, None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='Only global billing managers'):
        await patch_billing_project(db, 'bp-patch-ul', {'limit': None}, actor='admin', billing_role='quote_owner')


async def test_patch_billing_project_logs_limit_event_on_both_quote_and_bp(db):
    await create_quote(db, 'q-patch-log', cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-patch-log',))
    await create_billing_project(db, 'bp-patch-log', q_row['id'], 100.0, None, 'admin', 'global_bm')
    await patch_billing_project(db, 'bp-patch-log', {'limit': 200.0}, actor='admin', billing_role='global_bm')

    bp_events = await get_billing_project_events(db, 'bp-patch-log')
    bp_actions = {e['action'] for e in bp_events}
    assert 'limit_changed' in bp_actions
    limit_event = next(e for e in bp_events if e['action'] == 'limit_changed')
    assert json.loads(limit_event['detail']) == {'old': 100.0, 'new': 200.0}

    q_events = await get_quote_events(db, 'q-patch-log')
    assert q_events is not None
    q_actions = {e['action'] for e in q_events}
    assert 'bp_limit_changed' in q_actions


async def test_patch_billing_project_description(db):
    await create_quote(db, 'q-patch-desc', cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-patch-desc',))
    await create_billing_project(db, 'bp-patch-desc', q_row['id'], 100.0, None, 'admin', 'global_bm')
    await patch_billing_project(
        db, 'bp-patch-desc', {'description': 'updated'}, actor='admin', billing_role='global_bm'
    )
    row = await db.select_and_fetchone('SELECT description FROM billing_projects WHERE name = %s', ('bp-patch-desc',))
    assert row['description'] == 'updated'


# ---------------------------------------------------------------------------
# change_billing_project_quote
# ---------------------------------------------------------------------------


async def test_change_bp_quote_updates_quote_id(db):
    await create_quote(db, 'q-src', cost_object='CO1', actor='admin', authorized_amount=500.0)
    await create_quote(db, 'q-dest', cost_object='CO2', actor='admin', authorized_amount=500.0)
    q_src = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-src',))
    q_dest = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-dest',))
    await create_billing_project(db, 'bp-move', q_src['id'], 100.0, None, 'admin', 'global_bm')
    await change_billing_project_quote(db, 'bp-move', q_dest['id'], actor='admin')
    row = await db.select_and_fetchone('SELECT quote_id FROM billing_projects WHERE name = %s', ('bp-move',))
    assert row['quote_id'] == q_dest['id']


async def test_change_bp_quote_nonexistent_bp_raises(db):
    await create_quote(db, 'q-change-missing-bp', cost_object='CO', actor='admin', authorized_amount=500.0)
    q = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-change-missing-bp',))
    with pytest.raises(Exception):  # NonExistentBillingProjectError
        await change_billing_project_quote(db, 'no-such-bp', q['id'], actor='admin')


async def test_change_bp_quote_nonexistent_dest_quote_raises(db):
    await create_quote(db, 'q-change-bad-dest', cost_object='CO', actor='admin', authorized_amount=500.0)
    q = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-change-bad-dest',))
    await create_billing_project(db, 'bp-bad-dest-quote', q['id'], 100.0, None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='Unknown quote'):
        await change_billing_project_quote(db, 'bp-bad-dest-quote', 999999, actor='admin')


async def test_change_bp_quote_exceeds_dest_limit_raises(db):
    await create_quote(db, 'q-move-src', cost_object='CO1', actor='admin', authorized_amount=500.0)
    await create_quote(db, 'q-move-dest', cost_object='CO2', actor='admin', authorized_amount=200.0)
    q_src = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-move-src',))
    q_dest = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-move-dest',))
    await create_billing_project(db, 'bp-already', q_dest['id'], 150.0, None, 'admin', 'global_bm')
    await create_billing_project(db, 'bp-toobig', q_src['id'], 100.0, None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='exceed destination quote authorized amount'):
        await change_billing_project_quote(db, 'bp-toobig', q_dest['id'], actor='admin')


async def test_change_bp_quote_logs_events_on_src_dest_and_bp(db):
    await create_quote(db, 'q-log-src', cost_object='CO1', actor='admin', authorized_amount=500.0)
    await create_quote(db, 'q-log-dest', cost_object='CO2', actor='admin', authorized_amount=500.0)
    q_src = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-log-src',))
    q_dest = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-log-dest',))
    await create_billing_project(db, 'bp-log-move', q_src['id'], 100.0, None, 'admin', 'global_bm')
    await change_billing_project_quote(db, 'bp-log-move', q_dest['id'], actor='admin')

    src_events = await get_quote_events(db, 'q-log-src')
    assert src_events is not None
    assert any(e['action'] == 'bp_unassigned' for e in src_events)

    dest_events = await get_quote_events(db, 'q-log-dest')
    assert dest_events is not None
    assert any(e['action'] == 'bp_assigned' for e in dest_events)

    bp_events = await get_billing_project_events(db, 'bp-log-move')
    assert any(e['action'] == 'quote_changed' for e in bp_events)
    changed_event = next(e for e in bp_events if e['action'] == 'quote_changed')
    assert json.loads(changed_event['detail']) == {'old': 'q-log-src', 'new': 'q-log-dest'}


# ---------------------------------------------------------------------------
# get_billing_project_events
# ---------------------------------------------------------------------------


async def test_get_billing_project_events_returns_events(db):
    await create_quote(db, 'q-bpe', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-bpe',))
    await create_billing_project(db, 'bp-bpe', q_row['id'], None, None, 'admin', 'global_bm', comment='init')
    events = await get_billing_project_events(db, 'bp-bpe')
    assert len(events) >= 1
    assert any(e['action'] == 'bp_created' for e in events)


# ---------------------------------------------------------------------------
# billing_permission_required decorator
# ---------------------------------------------------------------------------


async def test_billing_permission_required_rejects_none_role():
    @billing_permission_required(BillingPermission.VIEW_QUOTE)
    async def handler(_request, _userdata):
        return web.Response(text='ok')

    request = MagicMock(spec=web.Request)
    request.app = {'db': MagicMock()}
    userdata = {'username': 'test', 'system_permissions': {}}  # type: ignore[arg-type]

    with patch('batch.billing_auth.resolve_billing_role', new_callable=AsyncMock, return_value=None):
        with pytest.raises(web.HTTPForbidden):
            await handler(request, userdata)  # type: ignore[arg-type]


async def test_billing_permission_required_calls_handler_with_sufficient_role():
    @billing_permission_required(BillingPermission.VIEW_QUOTE)
    async def handler(_request, _userdata):
        return web.Response(text='ok')

    request = MagicMock(spec=web.Request)
    request.app = {'db': MagicMock()}
    request.__setitem__ = MagicMock()
    userdata = {'username': 'test', 'system_permissions': {}}  # type: ignore[arg-type]

    with patch('batch.billing_auth.resolve_billing_role', new_callable=AsyncMock, return_value='global_bm'):
        response = await handler(request, userdata)  # type: ignore[arg-type]
    assert isinstance(response, web.Response)
    assert response.text == 'ok'


# ---------------------------------------------------------------------------
# add_billing_project_user / remove_billing_project_user
# ---------------------------------------------------------------------------


async def _make_bp(db, quote_name, bp_name, limit=None):
    """Helper: create a quote + billing project, return quote_id."""
    await create_quote(db, quote_name, cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (quote_name,))
    await create_billing_project(db, bp_name, q_row['id'], limit, None, 'admin', 'global_bm')
    return q_row['id']


async def test_add_billing_project_user_happy_path(db):
    await _make_bp(db, 'q-add-u', 'bp-add-u')
    await add_billing_project_user(db, 'bp-add-u', 'luna', 'admin')
    row = await db.select_and_fetchone(
        'SELECT user FROM billing_project_users WHERE billing_project = %s AND user = %s',
        ('bp-add-u', 'luna'),
    )
    assert row is not None


async def test_add_billing_project_user_duplicate_raises(db):
    await _make_bp(db, 'q-add-u-dup', 'bp-add-u-dup')
    await add_billing_project_user(db, 'bp-add-u-dup', 'mars', 'admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await add_billing_project_user(db, 'bp-add-u-dup', 'mars', 'admin')


async def test_add_billing_project_user_closed_raises(db):
    await _make_bp(db, 'q-add-u-cl', 'bp-add-u-cl')
    await close_billing_project(db, 'bp-add-u-cl', 'admin')
    with pytest.raises(Exception):  # ClosedBillingProjectError
        await add_billing_project_user(db, 'bp-add-u-cl', 'nova', 'admin')


async def test_add_billing_project_user_logs_event(db):
    await _make_bp(db, 'q-add-u-log', 'bp-add-u-log')
    await add_billing_project_user(db, 'bp-add-u-log', 'orion', 'admin', comment='welcome')
    events = await get_billing_project_events(db, 'bp-add-u-log')
    assert any(e['action'] == 'user_added' and e['target_user'] == 'orion' for e in events)


async def test_remove_billing_project_user_happy_path(db):
    await _make_bp(db, 'q-rm-u', 'bp-rm-u')
    await add_billing_project_user(db, 'bp-rm-u', 'pluto', 'admin')
    await remove_billing_project_user(db, 'bp-rm-u', 'pluto', 'admin')
    row = await db.select_and_fetchone(
        'SELECT user FROM billing_project_users WHERE billing_project = %s AND user = %s',
        ('bp-rm-u', 'pluto'),
    )
    assert row is None


async def test_remove_billing_project_user_not_member_raises(db):
    await _make_bp(db, 'q-rm-u-nm', 'bp-rm-u-nm')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await remove_billing_project_user(db, 'bp-rm-u-nm', 'nobody', 'admin')


async def test_remove_billing_project_user_logs_event(db):
    await _make_bp(db, 'q-rm-u-log', 'bp-rm-u-log')
    await add_billing_project_user(db, 'bp-rm-u-log', 'quasar', 'admin')
    await remove_billing_project_user(db, 'bp-rm-u-log', 'quasar', 'admin', comment='bye')
    events = await get_billing_project_events(db, 'bp-rm-u-log')
    assert any(e['action'] == 'user_removed' and e['target_user'] == 'quasar' for e in events)


# ---------------------------------------------------------------------------
# close_billing_project
# ---------------------------------------------------------------------------


async def test_close_billing_project_happy_path(db):
    await _make_bp(db, 'q-close', 'bp-close')
    await close_billing_project(db, 'bp-close', 'admin')
    row = await db.select_and_fetchone('SELECT `status` FROM billing_projects WHERE name = %s', ('bp-close',))
    assert row['status'] == 'closed'


async def test_close_billing_project_not_found_raises(db):
    with pytest.raises(Exception):  # NonExistentBillingProjectError
        await close_billing_project(db, 'no-such-bp', 'admin')


async def test_close_billing_project_already_closed_raises(db):
    await _make_bp(db, 'q-close-dup', 'bp-close-dup')
    await close_billing_project(db, 'bp-close-dup', 'admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await close_billing_project(db, 'bp-close-dup', 'admin')


async def test_close_billing_project_running_batch_raises(db):
    await _make_bp(db, 'q-close-run', 'bp-close-run')
    async with db.start() as tx:
        await tx.execute_insertone(
            'INSERT INTO batches(billing_project, time_completed, deleted) VALUES (%s, NULL, FALSE)',
            ('bp-close-run',),
        )
    with pytest.raises(BatchUserError, match='running batches'):
        await close_billing_project(db, 'bp-close-run', 'admin')


async def test_close_billing_project_logs_event(db):
    await _make_bp(db, 'q-close-log', 'bp-close-log')
    await close_billing_project(db, 'bp-close-log', 'admin', comment='eod')
    bp_events = await get_billing_project_events(db, 'bp-close-log')
    assert any(e['action'] == 'bp_closed' for e in bp_events)
    quote_events = await get_quote_events(db, 'q-close-log')
    assert quote_events is not None
    assert any(e['action'] == 'bp_closed' and e['target_project'] == 'bp-close-log' for e in quote_events)


# ---------------------------------------------------------------------------
# reopen_billing_project
# ---------------------------------------------------------------------------


async def test_reopen_billing_project_happy_path(db):
    await _make_bp(db, 'q-reopen', 'bp-reopen')
    await close_billing_project(db, 'bp-reopen', 'admin')
    await reopen_billing_project(db, 'bp-reopen', 'admin')
    row = await db.select_and_fetchone('SELECT `status` FROM billing_projects WHERE name = %s', ('bp-reopen',))
    assert row['status'] == 'open'


async def test_reopen_billing_project_already_open_raises(db):
    await _make_bp(db, 'q-reopen-open', 'bp-reopen-open')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await reopen_billing_project(db, 'bp-reopen-open', 'admin')


async def test_reopen_billing_project_deleted_raises(db):
    await _make_bp(db, 'q-reopen-del', 'bp-reopen-del')
    await close_billing_project(db, 'bp-reopen-del', 'admin')
    await delete_billing_project(db, 'bp-reopen-del')
    with pytest.raises(BatchUserError, match='deleted'):
        await reopen_billing_project(db, 'bp-reopen-del', 'admin')


async def test_reopen_billing_project_logs_event(db):
    await _make_bp(db, 'q-reopen-log', 'bp-reopen-log')
    await close_billing_project(db, 'bp-reopen-log', 'admin')
    await reopen_billing_project(db, 'bp-reopen-log', 'admin', comment='back')
    bp_events = await get_billing_project_events(db, 'bp-reopen-log')
    assert any(e['action'] == 'bp_reopened' for e in bp_events)
    quote_events = await get_quote_events(db, 'q-reopen-log')
    assert quote_events is not None
    assert any(e['action'] == 'bp_reopened' and e['target_project'] == 'bp-reopen-log' for e in quote_events)


# ---------------------------------------------------------------------------
# delete_billing_project
# ---------------------------------------------------------------------------


async def test_delete_billing_project_happy_path(db):
    await _make_bp(db, 'q-del', 'bp-del')
    await close_billing_project(db, 'bp-del', 'admin')
    await delete_billing_project(db, 'bp-del')
    row = await db.select_and_fetchone('SELECT `status` FROM billing_projects WHERE name = %s', ('bp-del',))
    assert row['status'] == 'deleted'


async def test_delete_billing_project_open_raises(db):
    await _make_bp(db, 'q-del-open', 'bp-del-open')
    with pytest.raises(BatchUserError, match='open'):
        await delete_billing_project(db, 'bp-del-open')


async def test_delete_billing_project_already_deleted_raises(db):
    await _make_bp(db, 'q-del-dup', 'bp-del-dup')
    await close_billing_project(db, 'bp-del-dup', 'admin')
    await delete_billing_project(db, 'bp-del-dup')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await delete_billing_project(db, 'bp-del-dup')


async def test_delete_billing_project_not_found_raises(db):
    with pytest.raises(Exception):  # NonExistentBillingProjectError
        await delete_billing_project(db, 'no-such-bp')


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
