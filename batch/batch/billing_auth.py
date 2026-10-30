from functools import wraps
from typing import Any, Awaitable, Callable, Optional

from aiohttp import web

from gear import Database, SystemPermission, UserData

from . import billing_project_management as billing_dao
from .billing_grants import (
    BillingGrant,
    BillingPermission,
    BillingProjectTarget,
    BillingTarget,
    GlobalTarget,
    QuoteTarget,
    issue_grant,
)

_ALL_BILLING_PERMISSIONS: set[BillingPermission] = set(BillingPermission)

BILLING_ROLE_PERMISSIONS: dict[str, set[BillingPermission]] = {
    # Removing other users from a billing project (MANAGE_BP_MEMBERS) is global-billing-manager only; any member can
    # still remove themselves.
    'global_bm': _ALL_BILLING_PERMISSIONS,
    'quote_owner': {
        BillingPermission.VIEW_QUOTE,
        BillingPermission.EDIT_QUOTE,
        BillingPermission.EDIT_QUOTE_AMOUNT,
        BillingPermission.CLOSE_QUOTE,
        BillingPermission.MANAGE_MANAGERS,
        BillingPermission.VIEW_BP,
        BillingPermission.CREATE_BP,
        BillingPermission.EDIT_BP_LIMIT,
        BillingPermission.EDIT_BP_METADATA,
        BillingPermission.CLOSE_REOPEN_BP,
        BillingPermission.CHANGE_BP_QUOTE,
        BillingPermission.VIEW_EVENTS,
    },
    'quote_manager': {
        BillingPermission.VIEW_QUOTE,
        BillingPermission.EDIT_QUOTE,
        BillingPermission.VIEW_BP,
        BillingPermission.CREATE_BP,
        BillingPermission.EDIT_BP_LIMIT,
        BillingPermission.EDIT_BP_METADATA,
        BillingPermission.CLOSE_REOPEN_BP,
        BillingPermission.CHANGE_BP_QUOTE,
        BillingPermission.VIEW_EVENTS,
    },
    'bp_member': {
        BillingPermission.VIEW_BP,
        BillingPermission.EDIT_BP_METADATA,
        BillingPermission.VIEW_EVENTS,
    },
}


def is_global_billing_manager(userdata: UserData) -> bool:
    return userdata['system_permissions'].get(SystemPermission.UPDATE_ALL_BILLING_PROJECTS, False)


async def billing_role_on(db: Database, userdata: UserData, target: BillingTarget) -> Optional[str]:
    """The caller's billing role on target, or None if they have none."""
    username = userdata['username']
    is_gbm = is_global_billing_manager(userdata)
    if isinstance(target, QuoteTarget):
        return await billing_dao.get_billing_role_for_quote_id(db, username, is_gbm, target.quote_id)
    if isinstance(target, BillingProjectTarget):
        return await billing_dao.get_billing_role_for_bp(db, username, is_gbm, target.name)
    return 'global_bm' if is_gbm else None


def _forbidden(permission: BillingPermission) -> web.HTTPForbidden:
    return web.HTTPForbidden(reason=f'Insufficient billing permissions for {permission.value}')


async def require(
    db: Database, userdata: UserData, permission: BillingPermission, target: BillingTarget
) -> BillingGrant:
    """Check the caller's billing role grants permission on target, raising 403 if not."""
    role = await billing_role_on(db, userdata, target)
    if role is None or permission not in BILLING_ROLE_PERMISSIONS.get(role, set()):
        raise _forbidden(permission)
    return issue_grant(userdata['username'], permission, target)


def grant_from_system_permission(
    userdata: UserData, system_permission: SystemPermission, permission: BillingPermission, target: BillingTarget
) -> BillingGrant:
    """For routes authorized by a system permission rather than a billing role (legacy and admin-only routes)."""
    if not userdata['system_permissions'].get(system_permission, False):
        raise _forbidden(permission)
    return issue_grant(userdata['username'], permission, target)


async def _route_target(db: Database, userdata: UserData, request: web.Request) -> BillingTarget:
    quote_name = request.match_info.get('name')
    if quote_name:
        quote_id = await billing_dao.get_quote_id(db, quote_name)
        if quote_id is None:
            # Only global billing managers may learn that a quote doesn't exist.
            if is_global_billing_manager(userdata):
                raise web.HTTPNotFound(reason=f'Unknown quote {quote_name}.')
            raise web.HTTPForbidden(reason='Insufficient billing permissions')
        return QuoteTarget(quote_id)
    billing_project = request.match_info.get('billing_project')
    if billing_project:
        return BillingProjectTarget(billing_project)
    return GlobalTarget()


def billing_permission_required(
    permission: BillingPermission,
) -> Callable[[Callable[..., Any]], Callable[[web.Request, UserData], Awaitable[web.StreamResponse]]]:
    """Decorator that checks billing role permissions on the route's quote or billing project.

    Sets request['billing_role'] and request['billing_grant']. Must be stacked under @auth.authenticated_users_only().
    """

    def decorator(handler: Callable[..., Any]) -> Callable[[web.Request, UserData], Awaitable[web.StreamResponse]]:
        @wraps(handler)
        async def inner(request: web.Request, userdata: UserData) -> web.StreamResponse:
            db: Database = request.app['db']
            target = await _route_target(db, userdata, request)
            role = await billing_role_on(db, userdata, target)
            if role is None or permission not in BILLING_ROLE_PERMISSIONS.get(role, set()):
                raise _forbidden(permission)
            request['billing_role'] = role
            request['billing_grant'] = issue_grant(userdata['username'], permission, target)
            return await handler(request, userdata)

        return inner

    return decorator
