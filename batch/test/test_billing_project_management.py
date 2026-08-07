"""Unit tests for billing project management functions in billing_project_management.py.

Requires local MySQL (make local-mysql). The db fixture is provided by conftest.py.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from aiohttp import web

from batch.billing_auth import BILLING_ROLE_PERMISSIONS, BillingPermission, billing_permission_required
from batch.billing_project_management import (
    add_billing_project_user,
    add_quote_manager,
    change_billing_project_quote,
    close_billing_project,
    create_billing_project,
    create_quote,
    delete_billing_project,
    get_billing_project_events,
    get_billing_role_for_bp,
    get_quote,
    get_quote_events,
    patch_billing_project,
    remove_billing_project_user,
    reopen_billing_project,
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


async def _make_bp(db, quote_name, bp_name, limit=None):
    """Helper: create a quote + billing project, return quote_id."""
    await create_quote(db, quote_name, cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (quote_name,))
    await create_billing_project(db, bp_name, q_row['id'], limit, 'admin', 'global_bm')
    return q_row['id']


# ---------------------------------------------------------------------------
# Role resolution — billing project roles
# ---------------------------------------------------------------------------


async def test_get_billing_role_for_bp_member(db):
    await create_quote(db, 'q-bp-role', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-bp-role',))
    await create_billing_project(db, 'bp-role-test', q_row['id'], 50.0, 'admin', 'global_bm')
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
    await create_billing_project(db, 'bp-via-qm', q_row['id'], 50.0, 'admin', 'global_bm')
    await add_quote_manager(db, 'q-bp-qm', 'jack', 'owner', actor='admin')
    role = await get_billing_role_for_bp(db, 'jack', False, 'bp-via-qm')
    assert role == 'quote_owner'


async def test_get_billing_role_for_bp_non_member(db):
    await create_quote(db, 'q-bp-nm', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-bp-nm',))
    await create_billing_project(db, 'bp-non-member', q_row['id'], 50.0, 'admin', 'global_bm')
    role = await get_billing_role_for_bp(db, 'outsider', False, 'bp-non-member')
    assert role is None


# ---------------------------------------------------------------------------
# create_billing_project
# ---------------------------------------------------------------------------


async def test_create_billing_project_appears_in_quote(db):
    await create_quote(db, 'q-create-bp', cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-create-bp',))
    await create_billing_project(db, 'bp-new', q_row['id'], 100.0, 'admin', 'global_bm')
    q = await get_quote(db, 'q-create-bp')
    assert q is not None
    bp_names = [bp['billing_project'] for bp in q['billing_projects']]
    assert 'bp-new' in bp_names


async def test_create_billing_project_limit_exceeds_quote_raises(db):
    await create_quote(db, 'q-exceed', cost_object='CO', actor='admin', authorized_amount=200.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-exceed',))
    await create_billing_project(db, 'bp-exceed-1', q_row['id'], 150.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='exceed quote authorized amount'):
        await create_billing_project(db, 'bp-exceed-2', q_row['id'], 100.0, 'admin', 'global_bm')


async def test_create_billing_project_unlimited_rejected_under_limited_quote(db):
    await create_quote(db, 'q-ul-reject', cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-ul-reject',))
    with pytest.raises(BatchUserError, match='only be created under unlimited quotes'):
        await create_billing_project(db, 'bp-ul-reject', q_row['id'], None, 'admin', 'global_bm')


async def test_create_billing_project_duplicate_raises(db):
    await create_quote(db, 'q-dup-bp', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-dup-bp',))
    await create_billing_project(db, 'bp-dup', q_row['id'], None, 'admin', 'global_bm')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await create_billing_project(db, 'bp-dup', q_row['id'], None, 'admin', 'global_bm')


async def test_create_billing_project_logs_event(db):
    await create_quote(db, 'q-log-bp', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-log-bp',))
    await create_billing_project(db, 'bp-log', q_row['id'], 50.0, 'admin', 'global_bm', comment='init')
    bp_events = await get_billing_project_events(db, 'bp-log')
    assert 'bp_created' in {e['action'] for e in bp_events}
    created_event = next(e for e in bp_events if e['action'] == 'bp_created')
    detail = json.loads(created_event['detail'])
    assert detail == {'limit': 50.0}
    quote_events = await get_quote_events(db, 'q-log-bp')
    assert quote_events is not None
    assert any(e['action'] == 'bp_created' and e['target_project'] == 'bp-log' for e in quote_events)


async def test_create_billing_project_persists_description(db):
    await create_quote(db, 'q-bp-desc', cost_object='CO', actor='admin')
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-bp-desc',))
    await create_billing_project(db, 'bp-desc', q_row['id'], None, 'admin', 'global_bm', description='A test BP')
    row = await db.select_and_fetchone('SELECT description FROM billing_projects WHERE name = %s', ('bp-desc',))
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
async def test_patch_billing_project_field(db, updates, col, expected):
    await create_quote(db, 'q-patch', cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-patch',))
    await create_billing_project(db, 'bp-patch', q_row['id'], 100.0, 'admin', 'global_bm')
    await patch_billing_project(db, 'bp-patch', updates, actor='admin', billing_role='global_bm')
    row = await db.select_and_fetchone(f'SELECT `{col}` FROM billing_projects WHERE name = %s', ('bp-patch',))
    assert row[col] == expected


async def test_patch_billing_project_limit_exceeds_quote_raises(db):
    await create_quote(db, 'q-patch-cap', cost_object='CO', actor='admin', authorized_amount=300.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-patch-cap',))
    await create_billing_project(db, 'bp-patch-1', q_row['id'], 200.0, 'admin', 'global_bm')
    await create_billing_project(db, 'bp-patch-2', q_row['id'], 50.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='would exceed quote authorized amount'):
        await patch_billing_project(db, 'bp-patch-2', {'limit': 200.0}, actor='admin', billing_role='global_bm')


async def test_patch_billing_project_unlimited_requires_global_bm(db):
    await create_quote(db, 'q-patch-ul', cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-patch-ul',))
    await create_billing_project(db, 'bp-patch-ul', q_row['id'], 100.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='Only global billing managers'):
        await patch_billing_project(db, 'bp-patch-ul', {'limit': None}, actor='admin', billing_role='quote_owner')


async def test_patch_billing_project_unlimited_rejected_under_limited_quote(db):
    await create_quote(db, 'q-patch-ul-lim', cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-patch-ul-lim',))
    await create_billing_project(db, 'bp-patch-ul-lim', q_row['id'], 100.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='only exist under unlimited quotes'):
        await patch_billing_project(db, 'bp-patch-ul-lim', {'limit': None}, actor='admin', billing_role='global_bm')


async def test_patch_billing_project_logs_limit_event_on_both_quote_and_bp(db):
    await create_quote(db, 'q-patch-log', cost_object='CO', actor='admin', authorized_amount=500.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-patch-log',))
    await create_billing_project(db, 'bp-patch-log', q_row['id'], 100.0, 'admin', 'global_bm')
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
    await create_billing_project(db, 'bp-patch-desc', q_row['id'], 100.0, 'admin', 'global_bm')
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
    await create_billing_project(db, 'bp-move', q_src['id'], 100.0, 'admin', 'global_bm')
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
    await create_billing_project(db, 'bp-bad-dest-quote', q['id'], 100.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='Unknown quote'):
        await change_billing_project_quote(db, 'bp-bad-dest-quote', 999999, actor='admin')


async def test_change_bp_quote_exceeds_dest_limit_raises(db):
    await create_quote(db, 'q-move-src', cost_object='CO1', actor='admin', authorized_amount=500.0)
    await create_quote(db, 'q-move-dest', cost_object='CO2', actor='admin', authorized_amount=200.0)
    q_src = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-move-src',))
    q_dest = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-move-dest',))
    await create_billing_project(db, 'bp-already', q_dest['id'], 150.0, 'admin', 'global_bm')
    await create_billing_project(db, 'bp-toobig', q_src['id'], 100.0, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='exceed destination quote authorized amount'):
        await change_billing_project_quote(db, 'bp-toobig', q_dest['id'], actor='admin')


async def test_change_bp_quote_unlimited_bp_rejected_into_limited_quote(db):
    await create_quote(db, 'q-ul-src', cost_object='CO1', actor='admin', authorized_amount=None)
    await create_quote(db, 'q-ul-dest', cost_object='CO2', actor='admin', authorized_amount=500.0)
    q_src = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-ul-src',))
    q_dest = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-ul-dest',))
    await create_billing_project(db, 'bp-ul-move', q_src['id'], None, 'admin', 'global_bm')
    with pytest.raises(BatchUserError, match='finite funding'):
        await change_billing_project_quote(db, 'bp-ul-move', q_dest['id'], actor='admin')


async def test_change_bp_quote_logs_events_on_src_dest_and_bp(db):
    await create_quote(db, 'q-log-src', cost_object='CO1', actor='admin', authorized_amount=500.0)
    await create_quote(db, 'q-log-dest', cost_object='CO2', actor='admin', authorized_amount=500.0)
    q_src = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-log-src',))
    q_dest = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', ('q-log-dest',))
    await create_billing_project(db, 'bp-log-move', q_src['id'], 100.0, 'admin', 'global_bm')
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
    await create_billing_project(db, 'bp-bpe', q_row['id'], None, 'admin', 'global_bm', comment='init')
    events = await get_billing_project_events(db, 'bp-bpe')
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


def test_manage_bp_members_available_to_non_global_bm():
    for role in ('quote_owner', 'quote_manager', 'bp_member'):
        assert BillingPermission.MANAGE_BP_MEMBERS in BILLING_ROLE_PERMISSIONS[role], (
            f'{role} should have MANAGE_BP_MEMBERS'
        )


def test_manage_managers_available_to_quote_owner():
    assert BillingPermission.MANAGE_MANAGERS in BILLING_ROLE_PERMISSIONS['quote_owner']
    assert BillingPermission.MANAGE_MANAGERS not in BILLING_ROLE_PERMISSIONS['quote_manager']
    assert BillingPermission.MANAGE_MANAGERS not in BILLING_ROLE_PERMISSIONS['bp_member']


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
            """INSERT INTO batches
               (billing_project, userdata, `user`, state, n_jobs, time_created, format_version, time_completed, deleted)
               VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, FALSE)""",
            ('bp-close-run', '{}', 'testuser', 'running', 0, 0, 1),
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
