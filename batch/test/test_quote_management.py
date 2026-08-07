"""Unit tests for quote management functions in billing_project_management.py.

Requires local MySQL (make local-mysql). The db fixture is provided by conftest.py.
"""

import json

import pytest
import pytest_asyncio

from batch.billing_project_management import (
    add_quote_manager,
    close_billing_project,
    close_quote,
    create_billing_project,
    create_quote,
    delete_billing_project,
    edit_quote,
    get_billing_role_for_quote,
    get_billing_role_for_quote_id,
    get_quote,
    get_quote_events,
    list_quotes_for_user,
    remove_quote_manager,
    reopen_quote,
)
from batch.exceptions import BatchOperationAlreadyCompletedError, BatchUserError


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
    await create_billing_project(db, 'bp-under-q', q_row['id'], 100.0, 'admin', 'global_bm')
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
    await create_billing_project(db, 'bp-cap', q_row['id'], 400.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='less than sum of BP limits'):
        await edit_quote(db, 'q-cap', {'authorized_amount': 300.0}, actor='admin', billing_role='global_bm')


async def test_edit_quote_rejects_finite_amount_when_unlimited_bp_exists(db):
    # A quote containing an unlimited BP cannot be given a finite cap — the BP
    # could keep accruing charges beyond the cap.
    await create_quote(db, 'q-ul-bp', cost_object='CO', actor='admin', authorized_amount=None)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-ul-bp',))
    await create_billing_project(db, 'bp-ul', q_row['id'], None, 'admin', 'global_bm')
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
    await create_billing_project(db, 'bp-open', q_row['id'], None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='bp-open'):
        await close_quote(db, 'q-open-bp', actor='admin')


async def test_close_quote_allowed_when_all_bps_closed(db):
    await create_quote(db, 'q-all-closed', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-all-closed',))
    await create_billing_project(db, 'bp-cl', q_row['id'], None, 'admin', 'global_bm')
    await close_billing_project(db, 'bp-cl', actor='admin')
    await close_quote(db, 'q-all-closed', actor='admin')
    row = await db.select_and_fetchone('SELECT state FROM quotes WHERE name = %s', ('q-all-closed',))
    assert row['state'] == 'closed'


async def test_close_quote_allowed_when_bps_deleted(db):
    await create_quote(db, 'q-bp-deleted', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-bp-deleted',))
    await create_billing_project(db, 'bp-del', q_row['id'], None, 'admin', 'global_bm')
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
# reopen_quote
# ---------------------------------------------------------------------------


async def test_reopen_quote_sets_state(db):
    await create_quote(db, 'q-reopen', cost_object='CO', actor='admin')
    await close_quote(db, 'q-reopen', actor='admin')
    await reopen_quote(db, 'q-reopen', actor='admin')
    row = await db.select_and_fetchone('SELECT state FROM quotes WHERE name = %s', ('q-reopen',))
    assert row['state'] == 'open'


async def test_reopen_quote_logs_event(db):
    await create_quote(db, 'q-reopen-log', cost_object='CO', actor='admin')
    await close_quote(db, 'q-reopen-log', actor='admin')
    await reopen_quote(db, 'q-reopen-log', actor='admin', comment='mistake')
    events = await get_quote_events(db, 'q-reopen-log')
    assert events is not None
    assert any(e['action'] == 'quote_reopened' for e in events)
    reopen_event = next(e for e in events if e['action'] == 'quote_reopened')
    assert reopen_event['comment'] == 'mistake'


async def test_reopen_quote_already_open_raises(db):
    await create_quote(db, 'q-reopen-dup', cost_object='CO', actor='admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await reopen_quote(db, 'q-reopen-dup', actor='admin')


async def test_reopen_quote_unknown_raises(db):
    with pytest.raises(BatchUserError):
        await reopen_quote(db, 'no-such-quote', actor='admin')


async def test_reopen_quote_allows_new_bps(db):
    await create_quote(db, 'q-reopen-bp', cost_object='CO', actor='admin')
    await close_quote(db, 'q-reopen-bp', actor='admin')
    await reopen_quote(db, 'q-reopen-bp', actor='admin')
    # Should be able to create a BP again after reopening
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-reopen-bp',))
    await create_billing_project(db, 'bp-after-reopen', q_row['id'], None, 'admin', 'global_bm')
    row = await db.select_and_fetchone('SELECT status FROM billing_projects WHERE name = %s', ('bp-after-reopen',))
    assert row['status'] == 'open'


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
# Role resolution — quote roles
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


async def test_get_billing_role_for_quote_id(db):
    await create_quote(db, 'q-id-role', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-id-role',))
    await add_quote_manager(db, 'q-id-role', 'kim', 'manager', actor='admin')
    role = await get_billing_role_for_quote_id(db, 'kim', False, q_row['id'])
    assert role == 'quote_manager'
