"""Unit tests for billing project management functions in billing_project_management.py.

Requires local MySQL (make local-mysql). The db fixture is provided by conftest.py.
"""

import json

import pytest

from batch.billing_auth import BILLING_ROLE_PERMISSIONS, BillingPermission
from batch.billing_grants import BillingAuthorizationError
from batch.billing_project_management import (
    INTERNAL_QUOTE_ID,
    get_billing_project_events,
    get_billing_role_for_bp,
    get_quote,
    get_quote_events,
)
from batch.exceptions import (
    BatchOperationAlreadyCompletedError,
    BatchUserError,
    BillingValidationError,
    ClosedBillingProjectError,
    NonExistentBillingProjectError,
    NonExistentQuoteError,
)

from .billing_writes import (
    add_billing_project_user,
    add_quote_manager,
    change_billing_project_quote,
    close_billing_project,
    close_quote,
    create_billing_project,
    create_quote,
    delete_billing_project,
    patch_billing_project,
    remove_billing_project_user,
    reopen_billing_project,
)


async def _make_bp(db, quote_name, bp_name, limit=100.0):
    """Helper: create a quote + billing project, return quote_id."""
    await create_quote(db, quote_name, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (quote_name,))
    await create_billing_project(db, bp_name, q_row['id'], limit, 'admin', 'global_bm')
    return q_row['id']


# ---------------------------------------------------------------------------
# Role resolution — billing project roles
# ---------------------------------------------------------------------------


async def test_get_billing_role_for_bp_member(db, unique_name):
    q_bp_role = unique_name('q-bp-role')
    bp_role_test = unique_name('bp-role-test')
    await create_quote(db, q_bp_role, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_bp_role,))
    await create_billing_project(db, bp_role_test, q_row['id'], 50.0, 'admin', 'global_bm')
    async with db.start() as tx:
        await tx.execute_insertone(
            'INSERT INTO billing_project_users(billing_project, user, user_cs) VALUES (%s, %s, %s)',
            (bp_role_test, 'irene', 'irene'),
        )
    role = await get_billing_role_for_bp(db, 'irene', False, bp_role_test)
    assert role == 'bp_member'


async def test_get_billing_role_for_bp_via_quote_manager(db, unique_name):
    q_bp_qm = unique_name('q-bp-qm')
    bp_via_qm = unique_name('bp-via-qm')
    await create_quote(db, q_bp_qm, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_bp_qm,))
    await create_billing_project(db, bp_via_qm, q_row['id'], 50.0, 'admin', 'global_bm')
    await add_quote_manager(db, q_bp_qm, 'jack', 'owner', actor='admin')
    role = await get_billing_role_for_bp(db, 'jack', False, bp_via_qm)
    assert role == 'quote_owner'


async def test_get_billing_role_for_bp_non_member(db, unique_name):
    q_bp_nm = unique_name('q-bp-nm')
    bp_non_member = unique_name('bp-non-member')
    await create_quote(db, q_bp_nm, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_bp_nm,))
    await create_billing_project(db, bp_non_member, q_row['id'], 50.0, 'admin', 'global_bm')
    role = await get_billing_role_for_bp(db, 'outsider', False, bp_non_member)
    assert role is None


# ---------------------------------------------------------------------------
# create_billing_project
# ---------------------------------------------------------------------------


async def test_create_billing_project_appears_in_quote(db, unique_name):
    q_create_bp = unique_name('q-create-bp')
    bp_new = unique_name('bp-new')
    await create_quote(db, q_create_bp, cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_create_bp,))
    await create_billing_project(db, bp_new, q_row['id'], 100.0, 'admin', 'global_bm')
    q = await get_quote(db, q_create_bp)
    assert q is not None
    bp_names = [bp['billing_project'] for bp in q['billing_projects']]
    assert bp_new in bp_names


async def test_create_billing_project_limit_exceeds_quote_raises(db, unique_name):
    q_exceed = unique_name('q-exceed')
    bp_exceed_1 = unique_name('bp-exceed-1')
    bp_exceed_2 = unique_name('bp-exceed-2')
    await create_quote(db, q_exceed, cost_object='CO', actor='admin', authorized_amount=200.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_exceed,))
    await create_billing_project(db, bp_exceed_1, q_row['id'], 150.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='exceed quote authorized amount'):
        await create_billing_project(db, bp_exceed_2, q_row['id'], 100.0, 'admin', 'global_bm')


async def test_create_billing_project_unlimited_rejected_under_limited_quote(db, unique_name):
    q_ul_reject = unique_name('q-ul-reject')
    bp_ul_reject = unique_name('bp-ul-reject')
    await create_quote(db, q_ul_reject, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_ul_reject,))
    with pytest.raises(BatchUserError, match='only be created under the INTERNAL quote'):
        await create_billing_project(db, bp_ul_reject, q_row['id'], None, 'admin', 'global_bm')


async def test_create_billing_project_unlimited_allowed_under_internal(db, unique_name):
    bp_ul_internal = unique_name('bp-ul-internal')
    await create_billing_project(db, bp_ul_internal, INTERNAL_QUOTE_ID, None, 'admin', 'global_bm')
    row = await db.select_and_fetchone(
        'SELECT quote_id, `limit` FROM billing_projects WHERE name = %s', (bp_ul_internal,)
    )
    assert row['quote_id'] == INTERNAL_QUOTE_ID
    assert row['limit'] is None


async def test_create_billing_project_unlimited_under_internal_requires_global_bm(db, unique_name):
    bp_ul_internal_qo = unique_name('bp-ul-internal-qo')
    with pytest.raises(BillingAuthorizationError):
        await create_billing_project(db, bp_ul_internal_qo, INTERNAL_QUOTE_ID, None, 'admin', 'quote_owner')


async def test_create_billing_project_unlimited_under_limited_quote_is_user_error_for_quote_owner(db, unique_name):
    # The quote rule is checked before the SET_UNLIMITED grant, so this is a 400, not a BillingAuthorizationError.
    q_ul_qo = unique_name('q-ul-qo')
    bp_ul_qo = unique_name('bp-ul-qo')
    await create_quote(db, q_ul_qo, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_ul_qo,))
    with pytest.raises(BillingValidationError, match='only be created under the INTERNAL quote'):
        await create_billing_project(db, bp_ul_qo, q_row['id'], None, 'alice', 'quote_owner')


async def test_create_billing_project_under_closed_quote_raises(db, unique_name):
    q_closed_create = unique_name('q-closed-create')
    bp_closed_create = unique_name('bp-closed-create')
    q_id = await create_quote(db, q_closed_create, cost_object='CO', actor='admin', authorized_amount=500.0)
    await close_quote(db, q_closed_create, actor='admin')
    with pytest.raises(BatchUserError, match='closed quote'):
        await create_billing_project(db, bp_closed_create, q_id, 100.0, 'admin', 'global_bm')


async def test_rule_violation_is_bad_request(db, unique_name):
    q_400 = unique_name('q-400')
    bp_400 = unique_name('bp-400')
    await create_quote(db, q_400, cost_object='CO', actor='admin', authorized_amount=100.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_400,))
    with pytest.raises(BillingValidationError) as exc_info:
        await create_billing_project(db, bp_400, q_row['id'], 200.0, 'admin', 'global_bm')
    assert exc_info.value.http_response().status == 400


async def test_unknown_quote_is_not_found(db, unique_name):
    bp_404 = unique_name('bp-404')
    with pytest.raises(NonExistentQuoteError) as exc_info:
        await create_billing_project(db, bp_404, 999999, 100.0, 'admin', 'global_bm')
    assert exc_info.value.http_response().status == 404


async def test_create_billing_project_duplicate_raises(db, unique_name):
    q_dup_bp = unique_name('q-dup-bp')
    bp_dup = unique_name('bp-dup')
    await create_quote(db, q_dup_bp, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_dup_bp,))
    await create_billing_project(db, bp_dup, q_row['id'], 100.0, 'admin', 'global_bm')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await create_billing_project(db, bp_dup, q_row['id'], 100.0, 'admin', 'global_bm')


async def test_create_billing_project_logs_event(db, unique_name):
    q_log_bp = unique_name('q-log-bp')
    bp_log = unique_name('bp-log')
    await create_quote(db, q_log_bp, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_log_bp,))
    await create_billing_project(db, bp_log, q_row['id'], 50.0, 'admin', 'global_bm', comment='init')
    bp_events = await get_billing_project_events(db, bp_log)
    assert 'bp_created' in {e['action'] for e in bp_events}
    created_event = next(e for e in bp_events if e['action'] == 'bp_created')
    detail = json.loads(created_event['detail'])
    assert detail == {'limit': 50.0}
    quote_events = await get_quote_events(db, q_log_bp)
    assert quote_events is not None
    assert any(e['action'] == 'bp_created' and e['target_project'] == bp_log for e in quote_events)


async def test_create_billing_project_logs_initial_users(db, unique_name):
    q_log_init_users = unique_name('q-log-init-users')
    bp_log_init_users = unique_name('bp-log-init-users')
    await create_quote(db, q_log_init_users, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_log_init_users,))
    await create_billing_project(
        db,
        bp_log_init_users,
        q_row['id'],
        50.0,
        'admin',
        'global_bm',
        initial_users=['ursa', 'vega'],
        comment='init',
    )
    bp_events = await get_billing_project_events(db, bp_log_init_users)
    user_added = [e for e in bp_events if e['action'] == 'user_added']
    assert {e['target_user'] for e in user_added} == {'ursa', 'vega'}
    assert all(e['actor'] == 'admin' and e['comment'] == 'init' for e in user_added)
    rows = [
        r
        async for r in db.select_and_fetchall(
            'SELECT user FROM billing_project_users WHERE billing_project = %s', (bp_log_init_users,)
        )
    ]
    assert {r['user'] for r in rows} == {'ursa', 'vega'}


async def test_create_billing_project_persists_description(db, unique_name):
    q_bp_desc = unique_name('q-bp-desc')
    bp_desc = unique_name('bp-desc')
    await create_quote(db, q_bp_desc, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_bp_desc,))
    await create_billing_project(db, bp_desc, q_row['id'], 100.0, 'admin', 'global_bm', description='A test BP')
    row = await db.select_and_fetchone('SELECT description FROM billing_projects WHERE name = %s', (bp_desc,))
    assert row['description'] == 'A test BP'


# ---------------------------------------------------------------------------
# patch_billing_project
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'updates,col,expected',
    [
        pytest.param({'limit': 200.0}, 'limit', 200.0, id='limit'),
    ],
)
async def test_patch_billing_project_field(db, unique_name, updates, col, expected):
    q_patch = unique_name('q-patch')
    bp_patch = unique_name('bp-patch')
    await create_quote(db, q_patch, cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_patch,))
    await create_billing_project(db, bp_patch, q_row['id'], 100.0, 'admin', 'global_bm')
    await patch_billing_project(db, bp_patch, updates, actor='admin', billing_role='global_bm')
    row = await db.select_and_fetchone(f'SELECT `{col}` FROM billing_projects WHERE name = %s', (bp_patch,))
    assert row[col] == expected


async def test_patch_billing_project_limit_exceeds_quote_raises(db, unique_name):
    q_patch_cap = unique_name('q-patch-cap')
    bp_patch_1 = unique_name('bp-patch-1')
    bp_patch_2 = unique_name('bp-patch-2')
    await create_quote(db, q_patch_cap, cost_object='CO', actor='admin', authorized_amount=300.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_patch_cap,))
    await create_billing_project(db, bp_patch_1, q_row['id'], 200.0, 'admin', 'global_bm')
    await create_billing_project(db, bp_patch_2, q_row['id'], 50.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='would exceed quote authorized amount'):
        await patch_billing_project(db, bp_patch_2, {'limit': 200.0}, actor='admin', billing_role='global_bm')


async def test_patch_billing_project_unlimited_requires_global_bm(db, unique_name):
    bp_patch_ul = unique_name('bp-patch-ul')
    await create_billing_project(db, bp_patch_ul, INTERNAL_QUOTE_ID, 100.0, 'admin', 'global_bm')
    with pytest.raises(BillingAuthorizationError):
        await patch_billing_project(db, bp_patch_ul, {'limit': None}, actor='admin', billing_role='quote_owner')


async def test_patch_billing_project_unlimited_under_limited_quote_is_user_error_for_quote_owner(db, unique_name):
    # The quote rule is checked before the SET_UNLIMITED grant, so this is a 400, not a BillingAuthorizationError.
    q_patch_ul_qo = unique_name('q-patch-ul-qo')
    bp_patch_ul_qo = unique_name('bp-patch-ul-qo')
    await create_quote(db, q_patch_ul_qo, cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_patch_ul_qo,))
    await create_billing_project(db, bp_patch_ul_qo, q_row['id'], 100.0, 'admin', 'global_bm')
    with pytest.raises(BillingValidationError, match='only exist under the INTERNAL quote'):
        await patch_billing_project(db, bp_patch_ul_qo, {'limit': None}, actor='alice', billing_role='quote_owner')


async def test_patch_billing_project_unlimited_rejected_under_limited_quote(db, unique_name):
    q_patch_ul_lim = unique_name('q-patch-ul-lim')
    bp_patch_ul_lim = unique_name('bp-patch-ul-lim')
    await create_quote(db, q_patch_ul_lim, cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_patch_ul_lim,))
    await create_billing_project(db, bp_patch_ul_lim, q_row['id'], 100.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='only exist under the INTERNAL quote'):
        await patch_billing_project(db, bp_patch_ul_lim, {'limit': None}, actor='admin', billing_role='global_bm')


async def test_patch_closed_billing_project_description_allowed(db, unique_name):
    bp_closed_desc = unique_name('bp-closed-desc')
    await create_billing_project(db, bp_closed_desc, INTERNAL_QUOTE_ID, 100.0, 'admin', 'global_bm')
    await close_billing_project(db, bp_closed_desc, 'admin')
    await patch_billing_project(
        db, bp_closed_desc, {'description': 'wrapped up'}, actor='admin', billing_role='bp_member'
    )
    row = await db.select_and_fetchone('SELECT description FROM billing_projects WHERE name = %s', (bp_closed_desc,))
    assert row['description'] == 'wrapped up'


async def test_patch_closed_billing_project_limit_rejected(db, unique_name):
    bp_closed_limit = unique_name('bp-closed-limit')
    await create_billing_project(db, bp_closed_limit, INTERNAL_QUOTE_ID, 100.0, 'admin', 'global_bm')
    await close_billing_project(db, bp_closed_limit, 'admin')
    with pytest.raises(ClosedBillingProjectError):
        await patch_billing_project(db, bp_closed_limit, {'limit': 50.0}, actor='admin', billing_role='global_bm')


async def test_patch_billing_project_logs_limit_event_on_both_quote_and_bp(db, unique_name):
    q_patch_log = unique_name('q-patch-log')
    bp_patch_log = unique_name('bp-patch-log')
    await create_quote(db, q_patch_log, cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_patch_log,))
    await create_billing_project(db, bp_patch_log, q_row['id'], 100.0, 'admin', 'global_bm')
    await patch_billing_project(db, bp_patch_log, {'limit': 200.0}, actor='admin', billing_role='global_bm')

    bp_events = await get_billing_project_events(db, bp_patch_log)
    bp_actions = {e['action'] for e in bp_events}
    assert 'limit_changed' in bp_actions
    limit_event = next(e for e in bp_events if e['action'] == 'limit_changed')
    assert json.loads(limit_event['detail']) == {'old': 100.0, 'new': 200.0}

    q_events = await get_quote_events(db, q_patch_log)
    assert q_events is not None
    q_actions = {e['action'] for e in q_events}
    assert 'bp_limit_changed' in q_actions


async def test_patch_billing_project_description(db, unique_name):
    q_patch_desc = unique_name('q-patch-desc')
    bp_patch_desc = unique_name('bp-patch-desc')
    await create_quote(db, q_patch_desc, cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_patch_desc,))
    await create_billing_project(db, bp_patch_desc, q_row['id'], 100.0, 'admin', 'global_bm')
    await patch_billing_project(db, bp_patch_desc, {'description': 'updated'}, actor='admin', billing_role='global_bm')
    row = await db.select_and_fetchone('SELECT description FROM billing_projects WHERE name = %s', (bp_patch_desc,))
    assert row['description'] == 'updated'


async def test_patch_billing_project_description_logs_event(db, unique_name):
    q_patch_desc_log = unique_name('q-patch-desc-log')
    bp_patch_desc_log = unique_name('bp-patch-desc-log')
    await create_quote(db, q_patch_desc_log, cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_patch_desc_log,))
    await create_billing_project(
        db, bp_patch_desc_log, q_row['id'], 100.0, 'admin', 'global_bm', description='original'
    )
    await patch_billing_project(
        db,
        bp_patch_desc_log,
        {'description': 'updated'},
        actor='admin',
        billing_role='global_bm',
        comment='why',
    )
    bp_events = await get_billing_project_events(db, bp_patch_desc_log)
    event = next(e for e in bp_events if e['action'] == 'description_changed')
    assert json.loads(event['detail']) == {'old': 'original', 'new': 'updated'}
    assert event['comment'] == 'why'


# ---------------------------------------------------------------------------
# change_billing_project_quote
# ---------------------------------------------------------------------------


async def test_change_bp_quote_updates_quote_id(db, unique_name):
    q_src = unique_name('q-src')
    q_dest = unique_name('q-dest')
    bp_move = unique_name('bp-move')
    await create_quote(db, q_src, cost_object='CO1', actor='admin', authorized_amount=500.0)
    await create_quote(db, q_dest, cost_object='CO2', actor='admin', authorized_amount=500.0)
    q_src = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_src,))
    q_dest = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_dest,))
    await create_billing_project(db, bp_move, q_src['id'], 100.0, 'admin', 'global_bm')
    await change_billing_project_quote(db, bp_move, q_dest['id'], actor='admin')
    row = await db.select_and_fetchone('SELECT quote_id FROM billing_projects WHERE name = %s', (bp_move,))
    assert row['quote_id'] == q_dest['id']


async def test_change_bp_quote_nonexistent_bp_raises(db, unique_name):
    q_change_missing_bp = unique_name('q-change-missing-bp')
    await create_quote(db, q_change_missing_bp, cost_object='CO', actor='admin', authorized_amount=500.0)
    q = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_change_missing_bp,))
    with pytest.raises(Exception):  # NonExistentBillingProjectError
        await change_billing_project_quote(db, 'no-such-bp', q['id'], actor='admin')


async def test_change_bp_quote_nonexistent_dest_quote_raises(db, unique_name):
    q_change_bad_dest = unique_name('q-change-bad-dest')
    bp_bad_dest_quote = unique_name('bp-bad-dest-quote')
    await create_quote(db, q_change_bad_dest, cost_object='CO', actor='admin', authorized_amount=500.0)
    q = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_change_bad_dest,))
    await create_billing_project(db, bp_bad_dest_quote, q['id'], 100.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='Unknown quote'):
        await change_billing_project_quote(db, bp_bad_dest_quote, 999999, actor='admin')


async def test_change_bp_quote_exceeds_dest_limit_raises(db, unique_name):
    q_move_src = unique_name('q-move-src')
    q_move_dest = unique_name('q-move-dest')
    bp_already = unique_name('bp-already')
    bp_toobig = unique_name('bp-toobig')
    await create_quote(db, q_move_src, cost_object='CO1', actor='admin', authorized_amount=500.0)
    await create_quote(db, q_move_dest, cost_object='CO2', actor='admin', authorized_amount=200.0)
    q_src = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_move_src,))
    q_dest = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_move_dest,))
    await create_billing_project(db, bp_already, q_dest['id'], 150.0, 'admin', 'global_bm')
    await create_billing_project(db, bp_toobig, q_src['id'], 100.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='exceed destination quote authorized amount'):
        await change_billing_project_quote(db, bp_toobig, q_dest['id'], actor='admin')


async def test_change_bp_quote_unlimited_bp_rejected_into_limited_quote(db, unique_name):
    q_ul_dest = unique_name('q-ul-dest')
    bp_ul_move = unique_name('bp-ul-move')
    await create_quote(db, q_ul_dest, cost_object='CO2', actor='admin', authorized_amount=500.0)
    q_dest = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_ul_dest,))
    await create_billing_project(db, bp_ul_move, INTERNAL_QUOTE_ID, None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='finite funding'):
        await change_billing_project_quote(db, bp_ul_move, q_dest['id'], actor='admin')


async def test_change_bp_quote_into_closed_quote_raises(db, unique_name):
    q_move_open = unique_name('q-move-open')
    q_move_closed = unique_name('q-move-closed')
    bp_move_to_closed = unique_name('bp-move-to-closed')
    await create_quote(db, q_move_open, cost_object='CO1', actor='admin', authorized_amount=500.0)
    dest_id = await create_quote(db, q_move_closed, cost_object='CO2', actor='admin', authorized_amount=500.0)
    await close_quote(db, q_move_closed, actor='admin')
    q_src = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_move_open,))
    await create_billing_project(db, bp_move_to_closed, q_src['id'], 100.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='closed quote'):
        await change_billing_project_quote(db, bp_move_to_closed, dest_id, actor='admin')


async def test_change_bp_quote_logs_events_on_src_dest_and_bp(db, unique_name):
    q_log_src = unique_name('q-log-src')
    q_log_dest = unique_name('q-log-dest')
    bp_log_move = unique_name('bp-log-move')
    await create_quote(db, q_log_src, cost_object='CO1', actor='admin', authorized_amount=500.0)
    await create_quote(db, q_log_dest, cost_object='CO2', actor='admin', authorized_amount=500.0)
    q_src = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_log_src,))
    q_dest = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_log_dest,))
    await create_billing_project(db, bp_log_move, q_src['id'], 100.0, 'admin', 'global_bm')
    await change_billing_project_quote(db, bp_log_move, q_dest['id'], actor='admin')

    src_events = await get_quote_events(db, q_log_src)
    assert src_events is not None
    assert any(e['action'] == 'bp_unassigned' for e in src_events)

    dest_events = await get_quote_events(db, q_log_dest)
    assert dest_events is not None
    assert any(e['action'] == 'bp_assigned' for e in dest_events)

    bp_events = await get_billing_project_events(db, bp_log_move)
    assert any(e['action'] == 'quote_changed' for e in bp_events)
    changed_event = next(e for e in bp_events if e['action'] == 'quote_changed')
    assert json.loads(changed_event['detail']) == {'old': q_log_src, 'new': q_log_dest}


# ---------------------------------------------------------------------------
# get_billing_project_events
# ---------------------------------------------------------------------------


async def test_get_billing_project_events_returns_events(db, unique_name):
    q_bpe = unique_name('q-bpe')
    bp_bpe = unique_name('bp-bpe')
    await create_quote(db, q_bpe, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (q_bpe,))
    await create_billing_project(db, bp_bpe, q_row['id'], 100.0, 'admin', 'global_bm', comment='init')
    events = await get_billing_project_events(db, bp_bpe)
    assert len(events) >= 1
    assert any(e['action'] == 'bp_created' for e in events)


# ---------------------------------------------------------------------------
# BILLING_ROLE_PERMISSIONS — user management restrictions
# ---------------------------------------------------------------------------


def test_add_bp_member_is_global_bm_only():
    for role in ('quote_owner', 'quote_manager', 'bp_member'):
        assert BillingPermission.ADD_BP_MEMBER not in BILLING_ROLE_PERMISSIONS[role], (
            f'{role} should not have ADD_BP_MEMBER'
        )
    assert BillingPermission.ADD_BP_MEMBER in BILLING_ROLE_PERMISSIONS['global_bm']


def test_add_manager_is_global_bm_only():
    for role in ('quote_owner', 'quote_manager', 'bp_member'):
        assert BillingPermission.ADD_MANAGER not in BILLING_ROLE_PERMISSIONS[role], (
            f'{role} should not have ADD_MANAGER'
        )
    assert BillingPermission.ADD_MANAGER in BILLING_ROLE_PERMISSIONS['global_bm']


def test_removing_other_members_is_global_bm_only():
    # Any member can still remove themselves; that is handled by the route, not this permission.
    for role in ('quote_owner', 'quote_manager', 'bp_member'):
        assert BillingPermission.MANAGE_BP_MEMBERS not in BILLING_ROLE_PERMISSIONS[role], (
            f'{role} should not have MANAGE_BP_MEMBERS'
        )
    assert BillingPermission.MANAGE_BP_MEMBERS in BILLING_ROLE_PERMISSIONS['global_bm']


def test_edit_quote_amount_is_global_bm_and_owner_only():
    assert BillingPermission.EDIT_QUOTE_AMOUNT in BILLING_ROLE_PERMISSIONS['global_bm']
    assert BillingPermission.EDIT_QUOTE_AMOUNT in BILLING_ROLE_PERMISSIONS['quote_owner']
    assert BillingPermission.EDIT_QUOTE_AMOUNT not in BILLING_ROLE_PERMISSIONS['quote_manager']
    assert BillingPermission.EDIT_QUOTE_AMOUNT not in BILLING_ROLE_PERMISSIONS['bp_member']


def test_manage_managers_available_to_quote_owner():
    assert BillingPermission.MANAGE_MANAGERS in BILLING_ROLE_PERMISSIONS['quote_owner']
    assert BillingPermission.MANAGE_MANAGERS not in BILLING_ROLE_PERMISSIONS['quote_manager']
    assert BillingPermission.MANAGE_MANAGERS not in BILLING_ROLE_PERMISSIONS['bp_member']


# ---------------------------------------------------------------------------
# add_billing_project_user / remove_billing_project_user
# ---------------------------------------------------------------------------


async def test_add_billing_project_user_happy_path(db, unique_name):
    q_add_u = unique_name('q-add-u')
    bp_add_u = unique_name('bp-add-u')
    await _make_bp(db, q_add_u, bp_add_u)
    await add_billing_project_user(db, bp_add_u, 'luna', 'admin')
    row = await db.select_and_fetchone(
        'SELECT user FROM billing_project_users WHERE billing_project = %s AND user = %s',
        (bp_add_u, 'luna'),
    )
    assert row is not None


async def test_add_billing_project_user_duplicate_raises(db, unique_name):
    q_add_u_dup = unique_name('q-add-u-dup')
    bp_add_u_dup = unique_name('bp-add-u-dup')
    await _make_bp(db, q_add_u_dup, bp_add_u_dup)
    await add_billing_project_user(db, bp_add_u_dup, 'mars', 'admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await add_billing_project_user(db, bp_add_u_dup, 'mars', 'admin')


async def test_add_billing_project_user_closed_raises(db, unique_name):
    q_add_u_cl = unique_name('q-add-u-cl')
    bp_add_u_cl = unique_name('bp-add-u-cl')
    await _make_bp(db, q_add_u_cl, bp_add_u_cl)
    await close_billing_project(db, bp_add_u_cl, 'admin')
    with pytest.raises(Exception):  # ClosedBillingProjectError
        await add_billing_project_user(db, bp_add_u_cl, 'nova', 'admin')


async def test_add_billing_project_user_logs_event(db, unique_name):
    q_add_u_log = unique_name('q-add-u-log')
    bp_add_u_log = unique_name('bp-add-u-log')
    await _make_bp(db, q_add_u_log, bp_add_u_log)
    await add_billing_project_user(db, bp_add_u_log, 'orion', 'admin', comment='welcome')
    events = await get_billing_project_events(db, bp_add_u_log)
    assert any(e['action'] == 'user_added' and e['target_user'] == 'orion' for e in events)


async def test_remove_billing_project_user_happy_path(db, unique_name):
    q_rm_u = unique_name('q-rm-u')
    bp_rm_u = unique_name('bp-rm-u')
    await _make_bp(db, q_rm_u, bp_rm_u)
    await add_billing_project_user(db, bp_rm_u, 'pluto', 'admin')
    await remove_billing_project_user(db, bp_rm_u, 'pluto', 'admin')
    row = await db.select_and_fetchone(
        'SELECT user FROM billing_project_users WHERE billing_project = %s AND user = %s',
        (bp_rm_u, 'pluto'),
    )
    assert row is None


async def test_remove_billing_project_user_not_member_raises(db, unique_name):
    q_rm_u_nm = unique_name('q-rm-u-nm')
    bp_rm_u_nm = unique_name('bp-rm-u-nm')
    await _make_bp(db, q_rm_u_nm, bp_rm_u_nm)
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await remove_billing_project_user(db, bp_rm_u_nm, 'nobody', 'admin')


async def test_remove_billing_project_user_logs_event(db, unique_name):
    q_rm_u_log = unique_name('q-rm-u-log')
    bp_rm_u_log = unique_name('bp-rm-u-log')
    await _make_bp(db, q_rm_u_log, bp_rm_u_log)
    await add_billing_project_user(db, bp_rm_u_log, 'quasar', 'admin')
    await remove_billing_project_user(db, bp_rm_u_log, 'quasar', 'admin', comment='bye')
    events = await get_billing_project_events(db, bp_rm_u_log)
    assert any(e['action'] == 'user_removed' and e['target_user'] == 'quasar' for e in events)


# ---------------------------------------------------------------------------
# close_billing_project
# ---------------------------------------------------------------------------


async def test_close_billing_project_happy_path(db, unique_name):
    q_close = unique_name('q-close')
    bp_close = unique_name('bp-close')
    await _make_bp(db, q_close, bp_close)
    await close_billing_project(db, bp_close, 'admin')
    row = await db.select_and_fetchone('SELECT `status` FROM billing_projects WHERE name = %s', (bp_close,))
    assert row['status'] == 'closed'


async def test_close_billing_project_not_found_raises(db):
    with pytest.raises(Exception):  # NonExistentBillingProjectError
        await close_billing_project(db, 'no-such-bp', 'admin')


async def test_close_billing_project_already_closed_raises(db, unique_name):
    q_close_dup = unique_name('q-close-dup')
    bp_close_dup = unique_name('bp-close-dup')
    await _make_bp(db, q_close_dup, bp_close_dup)
    await close_billing_project(db, bp_close_dup, 'admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await close_billing_project(db, bp_close_dup, 'admin')


async def test_close_billing_project_running_batch_raises(db, unique_name):
    q_close_run = unique_name('q-close-run')
    bp_close_run = unique_name('bp-close-run')
    await _make_bp(db, q_close_run, bp_close_run)
    async with db.start() as tx:
        await tx.execute_insertone(
            """INSERT INTO batches
               (billing_project, userdata, `user`, state, n_jobs, time_created, format_version, time_completed, deleted)
               VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, FALSE)""",
            (bp_close_run, '{}', 'testuser', 'running', 0, 0, 1),
        )
    with pytest.raises(BatchUserError, match='running batches'):
        await close_billing_project(db, bp_close_run, 'admin')


async def test_close_billing_project_logs_event(db, unique_name):
    q_close_log = unique_name('q-close-log')
    bp_close_log = unique_name('bp-close-log')
    await _make_bp(db, q_close_log, bp_close_log)
    await close_billing_project(db, bp_close_log, 'admin', comment='eod')
    bp_events = await get_billing_project_events(db, bp_close_log)
    assert any(e['action'] == 'bp_closed' for e in bp_events)
    quote_events = await get_quote_events(db, q_close_log)
    assert quote_events is not None
    assert any(e['action'] == 'bp_closed' and e['target_project'] == bp_close_log for e in quote_events)


# ---------------------------------------------------------------------------
# reopen_billing_project
# ---------------------------------------------------------------------------


async def test_reopen_billing_project_happy_path(db, unique_name):
    q_reopen = unique_name('q-reopen')
    bp_reopen = unique_name('bp-reopen')
    await _make_bp(db, q_reopen, bp_reopen)
    await close_billing_project(db, bp_reopen, 'admin')
    await reopen_billing_project(db, bp_reopen, 'admin')
    row = await db.select_and_fetchone('SELECT `status` FROM billing_projects WHERE name = %s', (bp_reopen,))
    assert row['status'] == 'open'


async def test_reopen_billing_project_already_open_raises(db, unique_name):
    q_reopen_open = unique_name('q-reopen-open')
    bp_reopen_open = unique_name('bp-reopen-open')
    await _make_bp(db, q_reopen_open, bp_reopen_open)
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await reopen_billing_project(db, bp_reopen_open, 'admin')


async def test_reopen_billing_project_deleted_raises(db, unique_name):
    q_reopen_del = unique_name('q-reopen-del')
    bp_reopen_del = unique_name('bp-reopen-del')
    await _make_bp(db, q_reopen_del, bp_reopen_del)
    await close_billing_project(db, bp_reopen_del, 'admin')
    await delete_billing_project(db, bp_reopen_del, 'admin')
    with pytest.raises(BatchUserError, match='deleted'):
        await reopen_billing_project(db, bp_reopen_del, 'admin')


async def test_reopen_billing_project_under_closed_quote_raises(db, unique_name):
    q_reopen_qclosed = unique_name('q-reopen-qclosed')
    bp_reopen_qclosed = unique_name('bp-reopen-qclosed')
    await _make_bp(db, q_reopen_qclosed, bp_reopen_qclosed)
    await close_billing_project(db, bp_reopen_qclosed, 'admin')
    await close_quote(db, q_reopen_qclosed, actor='admin')
    with pytest.raises(BatchUserError, match='closed quote'):
        await reopen_billing_project(db, bp_reopen_qclosed, 'admin')


async def test_reopen_billing_project_logs_event(db, unique_name):
    q_reopen_log = unique_name('q-reopen-log')
    bp_reopen_log = unique_name('bp-reopen-log')
    await _make_bp(db, q_reopen_log, bp_reopen_log)
    await close_billing_project(db, bp_reopen_log, 'admin')
    await reopen_billing_project(db, bp_reopen_log, 'admin', comment='back')
    bp_events = await get_billing_project_events(db, bp_reopen_log)
    assert any(e['action'] == 'bp_reopened' for e in bp_events)
    quote_events = await get_quote_events(db, q_reopen_log)
    assert quote_events is not None
    assert any(e['action'] == 'bp_reopened' and e['target_project'] == bp_reopen_log for e in quote_events)


# ---------------------------------------------------------------------------
# delete_billing_project
# ---------------------------------------------------------------------------


async def test_delete_billing_project_happy_path(db, unique_name):
    q_del = unique_name('q-del')
    bp_del = unique_name('bp-del')
    await _make_bp(db, q_del, bp_del)
    await close_billing_project(db, bp_del, 'admin')
    await delete_billing_project(db, bp_del, 'admin')
    row = await db.select_and_fetchone('SELECT `status` FROM billing_projects WHERE name = %s', (bp_del,))
    assert row['status'] == 'deleted'


async def test_delete_billing_project_open_raises(db, unique_name):
    q_del_open = unique_name('q-del-open')
    bp_del_open = unique_name('bp-del-open')
    await _make_bp(db, q_del_open, bp_del_open)
    with pytest.raises(BatchUserError, match='open'):
        await delete_billing_project(db, bp_del_open, 'admin')


async def test_delete_billing_project_already_deleted_raises(db, unique_name):
    q_del_dup = unique_name('q-del-dup')
    bp_del_dup = unique_name('bp-del-dup')
    await _make_bp(db, q_del_dup, bp_del_dup)
    await close_billing_project(db, bp_del_dup, 'admin')
    await delete_billing_project(db, bp_del_dup, 'admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await delete_billing_project(db, bp_del_dup, 'admin')


async def test_delete_billing_project_not_found_raises(db):
    with pytest.raises(NonExistentBillingProjectError):
        await delete_billing_project(db, 'no-such-bp', 'admin')
