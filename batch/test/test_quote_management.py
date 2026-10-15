"""Unit tests for quote management functions in billing_project_management.py.

Requires local MySQL (make local-mysql). The db fixture is provided by conftest.py.
"""

import json

import pytest

from batch.billing_project_management import (
    INTERNAL_QUOTE_ID,
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

# ---------------------------------------------------------------------------
# create_quote
# ---------------------------------------------------------------------------


async def test_create_quote_returns_id(db, unique_name):
    q_basic = unique_name('q-basic')
    quote_id = await create_quote(db, q_basic, cost_object='CO-001', actor='admin', authorized_amount=1000.0)
    assert isinstance(quote_id, int)
    assert quote_id > 0


async def test_create_quote_persists_fields(db, unique_name):
    q_fields = unique_name('q-fields')
    await create_quote(db, q_fields, cost_object='CO-002', actor='admin', authorized_amount=5000.0, pi_name='Jane')
    row = await db.select_and_fetchone('SELECT * FROM quotes WHERE name = %s', (q_fields,))
    assert row['cost_object'] == 'CO-002'
    assert row['authorized_amount'] == 5000.0
    assert row['pi_name'] == 'Jane'


async def test_create_quote_logs_creation_event(db, unique_name):
    q_event = unique_name('q-event')
    await create_quote(db, q_event, cost_object='CO-003', actor='admin', authorized_amount=1000.0, comment='initial')
    row = await db.select_and_fetchone(
        'SELECT qe.* FROM quote_events qe JOIN quotes q ON q.id = qe.quote_id WHERE q.name = %s',
        (q_event,),
    )
    assert row['action'] == 'quote_created'
    assert row['actor'] == 'admin'
    assert json.loads(row['detail']) == 'CO-003'
    assert row['comment'] == 'initial'


async def test_create_quote_duplicate_raises(db, unique_name):
    q_dup = unique_name('q-dup')
    await create_quote(db, q_dup, cost_object='CO-004', actor='admin', authorized_amount=1000.0)
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await create_quote(db, q_dup, cost_object='CO-004b', actor='admin', authorized_amount=1000.0)


async def test_create_quote_requires_authorized_amount(db, unique_name):
    q_unlimited = unique_name('q-unlimited')
    with pytest.raises(BatchUserError, match='Only the INTERNAL quote is unlimited'):
        await create_quote(db, q_unlimited, cost_object='CO-005', actor='admin', authorized_amount=None)
    row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_unlimited,))
    assert row is None


async def test_create_quote_persists_description(db, unique_name):
    q_desc = unique_name('q-desc')
    await create_quote(
        db,
        q_desc,
        cost_object='CO',
        actor='admin',
        authorized_amount=1000.0,
        description='My test quote',
    )
    row = await db.select_and_fetchone('SELECT description FROM quotes WHERE name = %s', (q_desc,))
    assert row['description'] == 'My test quote'


async def test_create_quote_description_defaults_null(db, unique_name):
    q_no_desc = unique_name('q-no-desc')
    await create_quote(db, q_no_desc, cost_object='CO', actor='admin', authorized_amount=1000.0)
    row = await db.select_and_fetchone('SELECT description FROM quotes WHERE name = %s', (q_no_desc,))
    assert row['description'] is None


async def test_create_quote_persists_quote_number(db, unique_name):
    q_num = unique_name('q-num')
    await create_quote(db, q_num, cost_object='CO', actor='admin', authorized_amount=1000.0, quote_number='Q-2026-001')
    row = await db.select_and_fetchone('SELECT quote_number FROM quotes WHERE name = %s', (q_num,))
    assert row['quote_number'] == 'Q-2026-001'


async def test_create_quote_quote_number_defaults_null(db, unique_name):
    q_no_num = unique_name('q-no-num')
    await create_quote(db, q_no_num, cost_object='CO', actor='admin', authorized_amount=1000.0)
    row = await db.select_and_fetchone('SELECT quote_number FROM quotes WHERE name = %s', (q_no_num,))
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
async def test_list_quotes_for_user(db, unique_name, is_global_bm, add_manager, expected_count_delta):
    q_list = unique_name('q-list')
    before = await list_quotes_for_user(db, 'alice', is_global_bm)
    await create_quote(db, q_list, cost_object='co', actor='admin', authorized_amount=1000.0)
    if add_manager:
        await add_quote_manager(db, q_list, 'alice', 'manager', actor='admin')
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
    ],
)
async def test_get_quote_fields(db, unique_name, kwargs, expected):
    q_get = unique_name('q-get')
    await create_quote(db, q_get, cost_object='CO', actor='admin', **kwargs)
    q = await get_quote(db, q_get)
    assert q is not None
    for key, val in expected.items():
        assert q[key] == val


async def test_get_quote_internal_is_unlimited(db):
    q = await get_quote(db, 'INTERNAL')
    assert q is not None
    assert q['authorized_amount'] is None


async def test_get_quote_includes_managers(db, unique_name):
    q_mgrs = unique_name('q-mgrs')
    await create_quote(db, q_mgrs, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await add_quote_manager(db, q_mgrs, 'bob', 'owner', actor='admin')
    q = await get_quote(db, q_mgrs)
    assert q is not None
    assert any(m['user'] == 'bob' and m['role'] == 'owner' for m in q['managers'])


async def test_get_quote_includes_billing_projects(db, unique_name):
    q_bps = unique_name('q-bps')
    bp_under_q = unique_name('bp-under-q')
    await create_quote(db, q_bps, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_bps,))
    await create_billing_project(db, bp_under_q, q_row['id'], 100.0, 'admin', 'global_bm')
    q = await get_quote(db, q_bps)
    assert q is not None
    bp_names = [bp['billing_project'] for bp in q['billing_projects']]
    assert bp_under_q in bp_names


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
async def test_edit_quote_field(db, unique_name, field, new_value):
    q_edit = unique_name('q-edit')
    await create_quote(db, q_edit, cost_object='original', actor='admin', authorized_amount=1000.0)
    await edit_quote(db, q_edit, {field: new_value}, actor='admin', billing_role='global_bm')
    row = await db.select_and_fetchone(f'SELECT `{field}` FROM quotes WHERE name = %s', (q_edit,))
    assert row[field] == new_value


async def test_edit_quote_logs_event(db, unique_name):
    q_edit_log = unique_name('q-edit-log')
    await create_quote(db, q_edit_log, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await edit_quote(db, q_edit_log, {'cost_object': 'NEW'}, actor='admin', billing_role='global_bm', comment='upd')
    events = [
        r
        async for r in db.select_and_fetchall(
            'SELECT action, comment, detail FROM quote_events qe JOIN quotes q ON q.id = qe.quote_id WHERE q.name = %s',
            (q_edit_log,),
        )
    ]
    actions = [e['action'] for e in events]
    assert 'quote_edited' in actions
    edit_event = next(e for e in events if e['action'] == 'quote_edited')
    detail = json.loads(edit_event['detail'])
    assert detail['cost_object'] == {'old': 'CO', 'new': 'NEW'}


async def test_edit_quote_rejects_amount_below_bp_limits(db, unique_name):
    q_cap = unique_name('q-cap')
    bp_cap = unique_name('bp-cap')
    await create_quote(db, q_cap, cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_cap,))
    await create_billing_project(db, bp_cap, q_row['id'], 400.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='less than sum of BP limits'):
        await edit_quote(db, q_cap, {'authorized_amount': 300.0}, actor='admin', billing_role='global_bm')


async def test_edit_quote_rejects_finite_amount_when_unlimited_bp_exists(db, unique_name):
    # A quote containing an unlimited BP cannot be given a finite cap — the BP
    # could keep accruing charges beyond the cap. Only INTERNAL can hold unlimited BPs.
    bp_ul = unique_name('bp-ul')
    await create_billing_project(db, bp_ul, INTERNAL_QUOTE_ID, None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='unlimited billing project'):
        await edit_quote(db, 'INTERNAL', {'authorized_amount': 1000.0}, actor='admin', billing_role='global_bm')


async def test_edit_quote_unlimited_requires_global_bm(db):
    with pytest.raises(BatchUserError, match='Only global billing managers'):
        await edit_quote(db, 'INTERNAL', {'authorized_amount': None}, actor='admin', billing_role='quote_owner')


async def test_edit_quote_non_internal_cannot_be_unlimited(db, unique_name):
    q_ul_no = unique_name('q-ul-no')
    await create_quote(db, q_ul_no, cost_object='CO', actor='admin', authorized_amount=500.0)
    with pytest.raises(BatchUserError, match='Only the INTERNAL quote'):
        await edit_quote(db, q_ul_no, {'authorized_amount': None}, actor='admin', billing_role='global_bm')
    row = await db.select_and_fetchone('SELECT authorized_amount FROM quotes WHERE name = %s', (q_ul_no,))
    assert row['authorized_amount'] == 500.0


# ---------------------------------------------------------------------------
# close_quote
# ---------------------------------------------------------------------------


async def test_close_quote_sets_state(db, unique_name):
    q_close = unique_name('q-close')
    await create_quote(db, q_close, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await close_quote(db, q_close, actor='admin')
    row = await db.select_and_fetchone('SELECT state FROM quotes WHERE name = %s', (q_close,))
    assert row['state'] == 'closed'


async def test_close_quote_logs_event(db, unique_name):
    q_close_log = unique_name('q-close-log')
    await create_quote(db, q_close_log, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await close_quote(db, q_close_log, actor='admin', comment='done')
    events = [
        r
        async for r in db.select_and_fetchall(
            'SELECT action, comment FROM quote_events qe JOIN quotes q ON q.id = qe.quote_id WHERE q.name = %s',
            (q_close_log,),
        )
    ]
    actions = [e['action'] for e in events]
    assert 'quote_closed' in actions
    close_event = next(e for e in events if e['action'] == 'quote_closed')
    assert close_event['comment'] == 'done'


async def test_close_quote_already_closed_raises(db, unique_name):
    q_close_dup = unique_name('q-close-dup')
    await create_quote(db, q_close_dup, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await close_quote(db, q_close_dup, actor='admin')
    with pytest.raises(BatchOperationAlreadyCompletedError, match='already closed'):
        await close_quote(db, q_close_dup, actor='admin')


async def test_close_quote_unknown_raises(db):
    with pytest.raises(BatchUserError, match='Unknown quote'):
        await close_quote(db, 'no-such-quote', actor='admin')


async def test_close_quote_blocked_by_open_bp(db, unique_name):
    q_open_bp = unique_name('q-open-bp')
    bp_open = unique_name('bp-open')
    await create_quote(db, q_open_bp, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_open_bp,))
    await create_billing_project(db, bp_open, q_row['id'], 100.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match=bp_open):
        await close_quote(db, q_open_bp, actor='admin')


async def test_close_quote_allowed_when_all_bps_closed(db, unique_name):
    q_all_closed = unique_name('q-all-closed')
    bp_cl = unique_name('bp-cl')
    await create_quote(db, q_all_closed, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_all_closed,))
    await create_billing_project(db, bp_cl, q_row['id'], 100.0, 'admin', 'global_bm')
    await close_billing_project(db, bp_cl, actor='admin')
    await close_quote(db, q_all_closed, actor='admin')
    row = await db.select_and_fetchone('SELECT state FROM quotes WHERE name = %s', (q_all_closed,))
    assert row['state'] == 'closed'


async def test_close_quote_allowed_when_bps_deleted(db, unique_name):
    q_bp_deleted = unique_name('q-bp-deleted')
    bp_del = unique_name('bp-del')
    await create_quote(db, q_bp_deleted, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_bp_deleted,))
    await create_billing_project(db, bp_del, q_row['id'], 100.0, 'admin', 'global_bm')
    await close_billing_project(db, bp_del, actor='admin')
    await delete_billing_project(db, bp_del)
    await close_quote(db, q_bp_deleted, actor='admin')
    row = await db.select_and_fetchone('SELECT state FROM quotes WHERE name = %s', (q_bp_deleted,))
    assert row['state'] == 'closed'


async def test_get_quote_includes_state(db, unique_name):
    q_state_field = unique_name('q-state-field')
    await create_quote(db, q_state_field, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q = await get_quote(db, q_state_field)
    assert q is not None
    assert q['state'] == 'open'
    await close_quote(db, q_state_field, actor='admin')
    q2 = await get_quote(db, q_state_field)
    assert q2 is not None
    assert q2['state'] == 'closed'


# ---------------------------------------------------------------------------
# reopen_quote
# ---------------------------------------------------------------------------


async def test_reopen_quote_sets_state(db, unique_name):
    q_reopen = unique_name('q-reopen')
    await create_quote(db, q_reopen, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await close_quote(db, q_reopen, actor='admin')
    await reopen_quote(db, q_reopen, actor='admin')
    row = await db.select_and_fetchone('SELECT state FROM quotes WHERE name = %s', (q_reopen,))
    assert row['state'] == 'open'


async def test_reopen_quote_logs_event(db, unique_name):
    q_reopen_log = unique_name('q-reopen-log')
    await create_quote(db, q_reopen_log, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await close_quote(db, q_reopen_log, actor='admin')
    await reopen_quote(db, q_reopen_log, actor='admin', comment='mistake')
    events = await get_quote_events(db, q_reopen_log)
    assert events is not None
    assert any(e['action'] == 'quote_reopened' for e in events)
    reopen_event = next(e for e in events if e['action'] == 'quote_reopened')
    assert reopen_event['comment'] == 'mistake'


async def test_reopen_quote_already_open_raises(db, unique_name):
    q_reopen_dup = unique_name('q-reopen-dup')
    await create_quote(db, q_reopen_dup, cost_object='CO', actor='admin', authorized_amount=1000.0)
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await reopen_quote(db, q_reopen_dup, actor='admin')


async def test_reopen_quote_unknown_raises(db):
    with pytest.raises(BatchUserError):
        await reopen_quote(db, 'no-such-quote', actor='admin')


async def test_reopen_quote_allows_new_bps(db, unique_name):
    q_reopen_bp = unique_name('q-reopen-bp')
    bp_after_reopen = unique_name('bp-after-reopen')
    await create_quote(db, q_reopen_bp, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await close_quote(db, q_reopen_bp, actor='admin')
    await reopen_quote(db, q_reopen_bp, actor='admin')
    # Should be able to create a BP again after reopening
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_reopen_bp,))
    await create_billing_project(db, bp_after_reopen, q_row['id'], 100.0, 'admin', 'global_bm')
    row = await db.select_and_fetchone('SELECT status FROM billing_projects WHERE name = %s', (bp_after_reopen,))
    assert row['status'] == 'open'


# ---------------------------------------------------------------------------
# add_quote_manager / remove_quote_manager
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('role', ['owner', 'manager'])
async def test_add_quote_manager_role(db, unique_name, role):
    q_role = unique_name(f'q-role-{role}')
    await create_quote(db, q_role, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await add_quote_manager(db, q_role, 'carol', role, actor='admin')
    row = await db.select_and_fetchone(
        'SELECT qm.role FROM quote_managers qm JOIN quotes q ON q.id = qm.quote_id WHERE q.name = %s AND qm.user = %s',
        (q_role, 'carol'),
    )
    assert row['role'] == role


async def test_add_quote_manager_duplicate_raises(db, unique_name):
    q_dup_mgr = unique_name('q-dup-mgr')
    await create_quote(db, q_dup_mgr, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await add_quote_manager(db, q_dup_mgr, 'dave', 'manager', actor='admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await add_quote_manager(db, q_dup_mgr, 'dave', 'owner', actor='admin')


async def test_remove_quote_manager(db, unique_name):
    q_rm_mgr = unique_name('q-rm-mgr')
    await create_quote(db, q_rm_mgr, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await add_quote_manager(db, q_rm_mgr, 'eve', 'manager', actor='admin')
    await remove_quote_manager(db, q_rm_mgr, 'eve', actor='admin')
    row = await db.select_and_fetchone(
        'SELECT qm.role FROM quote_managers qm JOIN quotes q ON q.id = qm.quote_id WHERE q.name = %s AND qm.user = %s',
        (q_rm_mgr, 'eve'),
    )
    assert row is None


async def test_remove_quote_manager_not_member_raises(db, unique_name):
    q_rm_nonmember = unique_name('q-rm-nonmember')
    await create_quote(db, q_rm_nonmember, cost_object='CO', actor='admin', authorized_amount=1000.0)
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await remove_quote_manager(db, q_rm_nonmember, 'nobody', actor='admin')


async def test_add_remove_quote_manager_events_written(db, unique_name):
    q_mgr_events = unique_name('q-mgr-events')
    await create_quote(db, q_mgr_events, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await add_quote_manager(db, q_mgr_events, 'frank', 'manager', actor='admin')
    await remove_quote_manager(db, q_mgr_events, 'frank', actor='admin')
    events = [
        r
        async for r in db.select_and_fetchall(
            'SELECT action, detail FROM quote_events qe JOIN quotes q ON q.id = qe.quote_id WHERE q.name = %s',
            (q_mgr_events,),
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
async def test_get_quote_events(db, unique_name, actions):
    q_events = unique_name('q-events')
    await create_quote(db, q_events, cost_object='CO', actor='admin', authorized_amount=1000.0)
    if 'manager_added' in actions or 'manager_removed' in actions:
        await add_quote_manager(db, q_events, 'grace', 'manager', actor='admin')
    if 'manager_removed' in actions:
        await remove_quote_manager(db, q_events, 'grace', actor='admin')
    events = await get_quote_events(db, q_events)
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
async def test_get_billing_role_for_quote_manager(db, unique_name, db_role, expected_role):
    q_role = unique_name('q-role')
    await create_quote(db, q_role, cost_object='CO', actor='admin', authorized_amount=1000.0)
    await add_quote_manager(db, q_role, 'hank', db_role, actor='admin')
    role = await get_billing_role_for_quote(db, 'hank', False, q_role)
    assert role == expected_role


async def test_get_billing_role_for_quote_non_member(db, unique_name):
    q_non_member = unique_name('q-non-member')
    await create_quote(db, q_non_member, cost_object='CO', actor='admin', authorized_amount=1000.0)
    role = await get_billing_role_for_quote(db, 'outsider', False, q_non_member)
    assert role is None


async def test_get_billing_role_for_quote_global_bm_bypasses_db(db):
    role = await get_billing_role_for_quote(db, 'anyone', True, 'nonexistent-quote')
    assert role == 'global_bm'


async def test_get_billing_role_for_quote_id(db, unique_name):
    q_id_role = unique_name('q-id-role')
    await create_quote(db, q_id_role, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_id_role,))
    await add_quote_manager(db, q_id_role, 'kim', 'manager', actor='admin')
    role = await get_billing_role_for_quote_id(db, 'kim', False, q_row['id'])
    assert role == 'quote_manager'
