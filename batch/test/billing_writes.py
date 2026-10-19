"""The billing_project_management write functions, called as a user with a given billing role.

Each wrapper issues the grants that role holds (global_bm unless a billing_role is given) and calls the real
function, so the DB-layer tests can focus on data behaviour. test_billing_grants covers the grant checks themselves.
"""

from typing import Optional

from batch import billing_project_management as bpm
from batch.billing_auth import BILLING_ROLE_PERMISSIONS
from batch.billing_grants import (
    BillingGrant,
    BillingPermission,
    BillingProjectTarget,
    BillingTarget,
    GlobalTarget,
    QuoteTarget,
    issue_grant,
)

P = BillingPermission


def _grant(actor: str, role: str, permission: BillingPermission, target: BillingTarget) -> Optional[BillingGrant]:
    if permission not in BILLING_ROLE_PERMISSIONS[role]:
        return None
    return issue_grant(actor, permission, target)


async def _quote_target(db, name: str) -> QuoteTarget:
    # An unknown quote gets a target that matches nothing; the function raises NonExistentQuoteError first.
    quote_id = await bpm.get_quote_id(db, name)
    return QuoteTarget(quote_id if quote_id is not None else -1)


async def create_quote(db, name, cost_object, actor, **kwargs):
    return await bpm.create_quote(db, name, cost_object, issue_grant(actor, P.CREATE_QUOTE, GlobalTarget()), **kwargs)


async def edit_quote(db, name, updates, actor, billing_role, comment=None):
    target = await _quote_target(db, name)
    return await bpm.edit_quote(
        db,
        name,
        updates,
        _grant(actor, billing_role, P.EDIT_QUOTE, target),  # type: ignore[arg-type]
        amount_grant=_grant(actor, billing_role, P.EDIT_QUOTE_AMOUNT, target),
        unlimited_grant=_grant(actor, billing_role, P.SET_UNLIMITED, GlobalTarget()),
        comment=comment,
    )


async def close_quote(db, name, actor, comment=None):
    grant = issue_grant(actor, P.CLOSE_QUOTE, await _quote_target(db, name))
    return await bpm.close_quote(db, name, grant, comment)


async def reopen_quote(db, name, actor, comment=None):
    grant = issue_grant(actor, P.CLOSE_QUOTE, await _quote_target(db, name))
    return await bpm.reopen_quote(db, name, grant, comment)


async def add_quote_manager(db, quote_name, target_user, role, actor, comment=None):
    grant = issue_grant(actor, P.ADD_MANAGER, await _quote_target(db, quote_name))
    return await bpm.add_quote_manager(db, quote_name, target_user, role, grant, comment)


async def remove_quote_manager(db, quote_name, target_user, actor, comment=None):
    grant = issue_grant(actor, P.MANAGE_MANAGERS, await _quote_target(db, quote_name))
    return await bpm.remove_quote_manager(db, quote_name, target_user, grant, comment)


async def create_billing_project(
    db, name, quote_id, limit, actor, billing_role, initial_users=None, description=None, comment=None
):
    return await bpm.create_billing_project(
        db,
        name,
        quote_id,
        limit,
        _grant(actor, billing_role, P.CREATE_BP, QuoteTarget(quote_id)),  # type: ignore[arg-type]
        unlimited_grant=_grant(actor, billing_role, P.SET_UNLIMITED, GlobalTarget()),
        members_grant=_grant(actor, billing_role, P.ADD_BP_MEMBER, BillingProjectTarget(name)),
        initial_users=initial_users,
        description=description,
        comment=comment,
    )


async def patch_billing_project(db, bp_name, updates, actor, billing_role, comment=None):
    target = BillingProjectTarget(bp_name)
    return await bpm.patch_billing_project(
        db,
        bp_name,
        updates,
        _grant(actor, billing_role, P.EDIT_BP_METADATA, target),  # type: ignore[arg-type]
        limit_grant=_grant(actor, billing_role, P.EDIT_BP_LIMIT, target),
        unlimited_grant=_grant(actor, billing_role, P.SET_UNLIMITED, GlobalTarget()),
        comment=comment,
    )


async def change_billing_project_quote(db, bp_name, dest_quote_id, actor, comment=None):
    return await bpm.change_billing_project_quote(
        db,
        bp_name,
        dest_quote_id,
        issue_grant(actor, P.CHANGE_BP_QUOTE, BillingProjectTarget(bp_name)),
        issue_grant(actor, P.CHANGE_BP_QUOTE, QuoteTarget(dest_quote_id)),
        comment,
    )


async def add_billing_project_user(db, bp_name, user, actor, comment=None):
    grant = issue_grant(actor, P.ADD_BP_MEMBER, BillingProjectTarget(bp_name))
    return await bpm.add_billing_project_user(db, bp_name, user, grant, comment)


async def remove_billing_project_user(db, bp_name, user, actor, comment=None):
    permission = P.VIEW_BP if user == actor else P.MANAGE_BP_MEMBERS
    grant = issue_grant(actor, permission, BillingProjectTarget(bp_name))
    return await bpm.remove_billing_project_user(db, bp_name, user, grant, comment)


async def close_billing_project(db, bp_name, actor, comment=None):
    grant = issue_grant(actor, P.CLOSE_REOPEN_BP, BillingProjectTarget(bp_name))
    return await bpm.close_billing_project(db, bp_name, grant, comment)


async def reopen_billing_project(db, bp_name, actor, comment=None):
    grant = issue_grant(actor, P.CLOSE_REOPEN_BP, BillingProjectTarget(bp_name))
    return await bpm.reopen_billing_project(db, bp_name, grant, comment)


async def delete_billing_project(db, bp_name, actor, comment=None):
    grant = issue_grant(actor, P.DELETE_BP, BillingProjectTarget(bp_name))
    return await bpm.delete_billing_project(db, bp_name, grant, comment)
