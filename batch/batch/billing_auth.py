from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Optional

from aiohttp import web

from gear import Database, SystemPermission, UserData

from . import billing_dao


class BillingPermission(str, Enum):
    VIEW_QUOTE = 'view_quote'
    EDIT_QUOTE = 'edit_quote'
    MANAGE_MANAGERS = 'manage_managers'
    VIEW_BP = 'view_bp'
    CREATE_BP = 'create_bp'
    EDIT_BP_LIMIT = 'edit_bp_limit'
    EDIT_BP_ALERT = 'edit_bp_alert'
    MANAGE_BP_MEMBERS = 'manage_bp_members'
    CLOSE_REOPEN_BP = 'close_reopen_bp'
    CHANGE_BP_QUOTE = 'change_bp_quote'
    VIEW_EVENTS = 'view_events'


_ALL_BILLING_PERMISSIONS: set['BillingPermission'] = set(BillingPermission)

BILLING_ROLE_PERMISSIONS: dict[str, set[BillingPermission]] = {
    'global_bm': _ALL_BILLING_PERMISSIONS,
    'quote_owner': {
        BillingPermission.VIEW_QUOTE,
        BillingPermission.EDIT_QUOTE,
        BillingPermission.MANAGE_MANAGERS,
        BillingPermission.VIEW_BP,
        BillingPermission.CREATE_BP,
        BillingPermission.EDIT_BP_LIMIT,
        BillingPermission.EDIT_BP_ALERT,
        BillingPermission.MANAGE_BP_MEMBERS,
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
        BillingPermission.EDIT_BP_ALERT,
        BillingPermission.MANAGE_BP_MEMBERS,
        BillingPermission.CLOSE_REOPEN_BP,
        BillingPermission.CHANGE_BP_QUOTE,
        BillingPermission.VIEW_EVENTS,
    },
    'bp_member': {
        BillingPermission.VIEW_BP,
        BillingPermission.EDIT_BP_ALERT,
        BillingPermission.MANAGE_BP_MEMBERS,
        BillingPermission.VIEW_EVENTS,
    },
}


def _is_global_bm(userdata: UserData) -> bool:
    return userdata['system_permissions'].get(SystemPermission.UPDATE_ALL_BILLING_PROJECTS, False)


async def resolve_billing_role(db: Database, username: str, userdata: UserData, request: web.Request) -> Optional[str]:
    """Determine the caller's billing role for this request's resource context."""
    is_gbm = _is_global_bm(userdata)
    quote_name = request.match_info.get('name')
    billing_project_name = request.match_info.get('billing_project')

    if quote_name:
        return await billing_dao.get_billing_role_for_quote(db, username, is_gbm, quote_name)
    if billing_project_name:
        return await billing_dao.get_billing_role_for_bp(db, username, is_gbm, billing_project_name)
    if is_gbm:
        return 'global_bm'
    return None


async def resolve_billing_role_for_quote_id(
    db: Database, username: str, userdata: UserData, quote_id: int
) -> Optional[str]:
    """Resolve billing role given an explicit quote_id (for create-BP flow)."""
    return await billing_dao.get_billing_role_for_quote_id(db, username, _is_global_bm(userdata), quote_id)


def billing_permission_required(
    permission: BillingPermission,
) -> Callable[[Callable[..., Any]], Callable[[web.Request, UserData], Awaitable[web.StreamResponse]]]:
    """Decorator that checks billing role permissions. Must be stacked under @auth.authenticated_users_only()."""

    def decorator(handler: Callable[..., Any]) -> Callable[[web.Request, UserData], Awaitable[web.StreamResponse]]:
        @wraps(handler)
        async def inner(request: web.Request, userdata: UserData) -> web.StreamResponse:
            db: Database = request.app['db']
            username = userdata['username']
            billing_role = await resolve_billing_role(db, username, userdata, request)
            if billing_role is None or permission not in BILLING_ROLE_PERMISSIONS.get(billing_role, set()):
                raise web.HTTPForbidden(reason=f'Insufficient billing permissions for {permission.value}')
            request['billing_role'] = billing_role
            return await handler(request, userdata)

        return inner

    return decorator
