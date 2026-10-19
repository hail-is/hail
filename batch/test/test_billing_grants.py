"""Tests for billing grants: issuing them in billing_auth, and the DB-layer writes refusing to run without them.

Requires local MySQL (make local-mysql). The db fixture is provided by conftest.py.
"""

import pytest
from aiohttp import web

from batch import billing_project_management as bpm
from batch.billing_auth import billing_permission_required, grant_from_system_permission, require
from batch.billing_grants import (
    BillingAuthorizationError,
    BillingGrant,
    BillingPermission,
    BillingProjectTarget,
    GlobalTarget,
    QuoteTarget,
    issue_grant,
)
from gear import SystemPermission, UserData

from .billing_writes import add_billing_project_user, add_quote_manager, create_billing_project, create_quote

P = BillingPermission


def _userdata(username: str, **system_permissions: bool) -> UserData:
    return UserData(
        id=0,
        state='active',
        username=username,
        login_id=username,
        namespace_name='default',
        is_service_account=False,
        hail_credentials_secret_name='',
        tokens_secret_name='',
        system_permissions=system_permissions,
    )


def _gbm(username: str) -> UserData:
    return _userdata(username, update_all_billing_projects=True)


class _Request(dict):
    def __init__(self, db, match_info):
        super().__init__()
        self.app = {'db': db}
        self.match_info = match_info


async def _quote_with_bp(db, unique_name):
    q, bp = unique_name('q-grant'), unique_name('bp-grant')
    await create_quote(db, q, 'CO', 'admin', authorized_amount=1000.0)
    quote_id = await bpm.get_quote_id(db, q)
    assert quote_id is not None
    await create_billing_project(db, bp, quote_id, 10.0, 'admin', 'global_bm')
    return q, quote_id, bp


# ---------------------------------------------------------------------------
# Grants themselves
# ---------------------------------------------------------------------------


def test_grants_cannot_be_constructed_directly():
    with pytest.raises(BillingAuthorizationError):
        BillingGrant('mallory', P.DELETE_BP, BillingProjectTarget('bp'))


async def test_write_without_grant_is_refused(db, unique_name):
    _, _, bp = await _quote_with_bp(db, unique_name)
    with pytest.raises(BillingAuthorizationError):
        await bpm.close_billing_project(db, bp, None)  # type: ignore[arg-type]


async def test_write_with_wrong_permission_is_refused(db, unique_name):
    _, _, bp = await _quote_with_bp(db, unique_name)
    with pytest.raises(BillingAuthorizationError):
        await bpm.close_billing_project(db, bp, issue_grant('admin', P.VIEW_BP, BillingProjectTarget(bp)))


async def test_write_with_grant_for_another_object_is_refused(db, unique_name):
    _, _, bp = await _quote_with_bp(db, unique_name)
    _, _, other_bp = await _quote_with_bp(db, lambda prefix: unique_name(prefix + '-other'))
    with pytest.raises(BillingAuthorizationError):
        await bpm.close_billing_project(db, bp, issue_grant('admin', P.CLOSE_REOPEN_BP, BillingProjectTarget(other_bp)))
    row = await db.select_and_fetchone('SELECT `status` FROM billing_projects WHERE name = %s', (bp,))
    assert row['status'] == 'open'


async def test_extra_grant_must_be_for_the_same_actor(db, unique_name):
    _, _, bp = await _quote_with_bp(db, unique_name)
    target = BillingProjectTarget(bp)
    with pytest.raises(BillingAuthorizationError):
        await bpm.patch_billing_project(
            db,
            bp,
            {'limit': 5.0},
            issue_grant('alice', P.EDIT_BP_METADATA, target),
            limit_grant=issue_grant('bob', P.EDIT_BP_LIMIT, target),
        )


async def test_sensitive_part_of_an_operation_needs_its_own_grant(db, unique_name):
    q, quote_id, _ = await _quote_with_bp(db, unique_name)
    with pytest.raises(BillingAuthorizationError):
        await bpm.edit_quote(
            db, q, {'authorized_amount': 2000.0}, issue_grant('admin', P.EDIT_QUOTE, QuoteTarget(quote_id))
        )


async def test_events_record_the_grant_actor(db, unique_name):
    _, _, bp = await _quote_with_bp(db, unique_name)
    await bpm.close_billing_project(db, bp, issue_grant('carol', P.CLOSE_REOPEN_BP, BillingProjectTarget(bp)))
    events = await bpm.get_billing_project_events(db, bp)
    assert next(e for e in events if e['action'] == 'bp_closed')['actor'] == 'carol'


async def test_leaving_needs_only_view_but_removing_others_needs_manage(db, unique_name):
    _, _, bp = await _quote_with_bp(db, unique_name)
    await add_billing_project_user(db, bp, 'dave', 'admin')
    await add_billing_project_user(db, bp, 'erin', 'admin')
    view = issue_grant('dave', P.VIEW_BP, BillingProjectTarget(bp))
    with pytest.raises(BillingAuthorizationError):
        await bpm.remove_billing_project_user(db, bp, 'erin', view)
    await bpm.remove_billing_project_user(db, bp, 'dave', view)


# ---------------------------------------------------------------------------
# Issuing grants
# ---------------------------------------------------------------------------


async def test_require_issues_grant_for_role_permissions(db, unique_name):
    q, quote_id, bp = await _quote_with_bp(db, unique_name)
    await add_quote_manager(db, q, 'mgr', 'manager', 'admin')
    await add_billing_project_user(db, bp, 'member', 'admin')

    grant = await require(db, _userdata('mgr'), P.EDIT_BP_LIMIT, BillingProjectTarget(bp))
    assert (grant.actor, grant.permission, grant.target) == ('mgr', P.EDIT_BP_LIMIT, BillingProjectTarget(bp))
    assert (await require(db, _userdata('mgr'), P.CREATE_BP, QuoteTarget(quote_id))).target == QuoteTarget(quote_id)
    await require(db, _userdata('member'), P.VIEW_BP, BillingProjectTarget(bp))

    for userdata, permission, target in [
        (_userdata('member'), P.CLOSE_REOPEN_BP, BillingProjectTarget(bp)),
        (_userdata('mgr'), P.EDIT_QUOTE_AMOUNT, QuoteTarget(quote_id)),
        (_userdata('mgr'), P.SET_UNLIMITED, GlobalTarget()),
        (_userdata('outsider'), P.VIEW_BP, BillingProjectTarget(bp)),
    ]:
        with pytest.raises(web.HTTPForbidden):
            await require(db, userdata, permission, target)


async def test_global_billing_managers_get_ordinary_scoped_grants(db, unique_name):
    _, quote_id, _ = await _quote_with_bp(db, unique_name)
    grant = await require(db, _gbm('gbm'), P.EDIT_QUOTE_AMOUNT, QuoteTarget(quote_id))
    assert grant.target == QuoteTarget(quote_id)
    assert (await require(db, _gbm('gbm'), P.SET_UNLIMITED, GlobalTarget())).actor == 'gbm'


def test_grant_from_system_permission():
    userdata = _userdata('admin', delete_all_billing_projects=True)
    grant = grant_from_system_permission(
        userdata, SystemPermission.DELETE_ALL_BILLING_PROJECTS, P.DELETE_BP, BillingProjectTarget('bp')
    )
    assert (grant.actor, grant.permission) == ('admin', P.DELETE_BP)
    with pytest.raises(web.HTTPForbidden):
        grant_from_system_permission(
            _userdata('nobody'), SystemPermission.DELETE_ALL_BILLING_PROJECTS, P.DELETE_BP, BillingProjectTarget('bp')
        )


# ---------------------------------------------------------------------------
# billing_permission_required decorator
# ---------------------------------------------------------------------------


@billing_permission_required(P.VIEW_QUOTE)
async def _view_quote_handler(_request, _userdata):
    return web.Response(text='ok')


async def test_decorator_issues_grant_for_route_quote(db, unique_name):
    q, quote_id, _ = await _quote_with_bp(db, unique_name)
    await add_quote_manager(db, q, 'qmgr', 'manager', 'admin')
    request = _Request(db, {'name': q})
    response = await _view_quote_handler(request, _userdata('qmgr'))  # type: ignore[arg-type]
    assert isinstance(response, web.Response) and response.text == 'ok'
    assert request['billing_role'] == 'quote_manager'
    assert request['billing_grant'] == issue_grant('qmgr', P.VIEW_QUOTE, QuoteTarget(quote_id))


async def test_decorator_rejects_caller_without_role(db, unique_name):
    q, _, _ = await _quote_with_bp(db, unique_name)
    with pytest.raises(web.HTTPForbidden):
        await _view_quote_handler(_Request(db, {'name': q}), _userdata('outsider'))  # type: ignore[arg-type]


async def test_decorator_only_reveals_unknown_quotes_to_global_billing_managers(db):
    with pytest.raises(web.HTTPForbidden):
        await _view_quote_handler(_Request(db, {'name': 'no-such-quote'}), _userdata('outsider'))  # type: ignore[arg-type]
    with pytest.raises(web.HTTPNotFound):
        await _view_quote_handler(_Request(db, {'name': 'no-such-quote'}), _gbm('gbm'))  # type: ignore[arg-type]
