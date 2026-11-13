"""Proof that a caller was checked for a billing permission on a particular quote or billing project.

billing_auth issues grants once it has checked a user's billing role (or, for legacy and admin-only routes, a
system permission). The write functions in billing_project_management require a grant for each permission an
operation needs, on the object it acts on, and take the acting user from the grant.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


class BillingPermission(str, Enum):
    VIEW_QUOTE = 'view_quote'
    CREATE_QUOTE = 'create_quote'
    EDIT_QUOTE = 'edit_quote'
    EDIT_QUOTE_AMOUNT = 'edit_quote_amount'
    CLOSE_QUOTE = 'close_quote'
    ADD_MANAGER = 'add_manager'
    MANAGE_MANAGERS = 'manage_managers'
    VIEW_BP = 'view_bp'
    CREATE_BP = 'create_bp'
    EDIT_BP_LIMIT = 'edit_bp_limit'
    EDIT_BP_METADATA = 'edit_bp_metadata'
    ADD_BP_MEMBER = 'add_bp_member'
    MANAGE_BP_MEMBERS = 'manage_bp_members'
    CLOSE_REOPEN_BP = 'close_reopen_bp'
    DELETE_BP = 'delete_bp'
    CHANGE_BP_QUOTE = 'change_bp_quote'
    # Making a quote or billing project unlimited (only possible within INTERNAL)
    SET_UNLIMITED = 'set_unlimited'
    VIEW_EVENTS = 'view_events'


@dataclass(frozen=True)
class GlobalTarget:
    """The deployment as a whole, for permissions that aren't about one existing quote or billing project."""


@dataclass(frozen=True)
class QuoteTarget:
    quote_id: int


@dataclass(frozen=True)
class BillingProjectTarget:
    name: str  # case-sensitive, as matched against billing_projects.name_cs


BillingTarget = Union[GlobalTarget, QuoteTarget, BillingProjectTarget]


class BillingAuthorizationError(Exception):
    """A billing write was called without a grant that covers it. A bug in the caller, not a user error."""


_ISSUER = object()


@dataclass(frozen=True)
class BillingGrant:
    actor: str
    permission: BillingPermission
    target: BillingTarget
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._issuer is not _ISSUER:
            raise BillingAuthorizationError('BillingGrants are only issued by issue_grant')


def issue_grant(actor: str, permission: BillingPermission, target: BillingTarget) -> BillingGrant:
    """Only for billing_auth, once it has checked the permission, and for tests."""
    return BillingGrant(actor, permission, target, _ISSUER)


def expect(
    grant: Optional[BillingGrant], permission: BillingPermission, target: BillingTarget, actor: Optional[str] = None
) -> str:
    """Check that grant covers permission on target (and was issued to actor, if given). Returns the grant's actor."""
    if grant is None or grant.permission is not permission or grant.target != target:
        raise BillingAuthorizationError(f'Expected a {permission.value} grant on {target}, got {grant}.')
    if actor is not None and grant.actor != actor:
        raise BillingAuthorizationError(f'Expected a grant issued to {actor}, got {grant}.')
    return grant.actor
