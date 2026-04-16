"""
Cross-service integration tests verifying that system role permissions are correctly enforced
at service API boundaries.

Test users:
- `test` (no roles initially): authenticated via Hail session token from /user-tokens/tokens.json
- `test-dev` (sysadmin): authenticated via GOOGLE_APPLICATION_CREDENTIALS (GSA key)

Caching note:
- Auth service uses LocalAuthenticator (direct DB) — role changes are visible immediately.
- Batch, CI, and monitoring use AuthServiceAuthenticator (10s TTL cache) — after a role
  change, 11s sleep is required before cross-service permission assertions are reliable.
"""

import asyncio
import json
import os
import time
from contextlib import contextmanager

from hailtop.auth import async_add_role, async_remove_role
from hailtop.config import get_deploy_config
from hailtop.utils import external_requests_client_session, retry_response_returning_functions

deploy_config = get_deploy_config()

# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------


def _load_test_user_token() -> str:
    tokens_file = '/user-tokens/tokens.json'
    with open(tokens_file, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    namespace = os.environ.get('HAIL_DEFAULT_NAMESPACE', deploy_config.default_namespace())
    return tokens[namespace]


# Unauthenticated session (no bearer token)
NO_AUTH_SESSION = external_requests_client_session()

# Session authenticated as the `test` user (no roles)
_test_token = _load_test_user_token()
TEST_USER_SESSION = external_requests_client_session(headers={'Authorization': f'Bearer {_test_token}'})

# ---------------------------------------------------------------------------
# Permission → endpoint mapping
#
# Only includes endpoints decorated with authenticated_users_with_permission
# (i.e. returns HTTP 401 when the caller lacks the required permission).
#
# Note: update_users is intentionally excluded — the only API endpoint for it
# (POST /api/v1alpha/invalidate_all_sessions) is too destructive to invoke
# in a shared test environment.
# ---------------------------------------------------------------------------

PERMISSION_ENDPOINTS: dict[str, list[tuple[str, str, str]]] = {
    # Auth service — LocalAuthenticator, no cache
    'read_users': [
        ('auth', 'GET', '/api/v1alpha/users'),
        ('auth', 'GET', '/api/v1alpha/users/test'),
    ],
    'delete_users': [
        ('auth', 'DELETE', '/api/v1alpha/users/nonexistent-permission-test-user'),
    ],
    'assign_system_roles': [
        ('auth', 'PATCH', '/api/v1alpha/system_roles/nonexistent-permission-test-user'),
    ],
    'read_system_roles': [
        ('auth', 'GET', '/api/v1alpha/system_roles/all'),
    ],
    # Batch service — AuthServiceAuthenticator, 10s TTL cache
    'create_billing_projects': [
        ('batch', 'POST', '/api/v1alpha/billing_projects/nonexistent-perm-test-project/create'),
    ],
    'delete_all_billing_projects': [
        ('batch', 'POST', '/api/v1alpha/billing_projects/nonexistent-perm-test-project/close'),
    ],
    'assign_users_to_all_billing_projects': [
        ('batch', 'POST', '/api/v1alpha/billing_projects/nonexistent-perm-test-project/users/test/add'),
    ],
    # CI service — AuthServiceAuthenticator, 10s TTL cache
    'read_ci': [
        ('ci', 'GET', '/api/v1alpha/deploy_status'),
        ('ci', 'GET', '/api/v1alpha/retried_tests'),
    ],
    'manage_ci': [
        ('ci', 'POST', '/api/v1alpha/update'),
    ],
    # Monitoring service — AuthServiceAuthenticator, 10s TTL cache
    'view_monitoring_dashboards': [
        ('monitoring', 'GET', '/api/v1alpha/billing'),
    ],
}

# Permissions whose endpoints live on services that cache userinfo (need sleep after role change)
_CROSS_SERVICE_PERMS = {
    'create_billing_projects',
    'delete_all_billing_projects',
    'assign_users_to_all_billing_projects',
    'read_ci',
    'manage_ci',
    'view_monitoring_dashboards',
}

# Permissions granted by each role (intersection with PERMISSION_ENDPOINTS keys)
ROLE_PERMISSIONS: dict[str, set[str]] = {
    'sysadmin-readonly': {'read_users', 'read_system_roles', 'read_ci', 'view_monitoring_dashboards'},
    'billing_manager': {
        'read_users',
        'create_billing_projects',
        'delete_all_billing_projects',
        'assign_users_to_all_billing_projects',
    },
    'developer': {'read_ci', 'manage_ci'},
    'sysadmin': {
        'read_users',
        'delete_users',
        'assign_system_roles',
        'read_system_roles',
        'view_monitoring_dashboards',
    },
}

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

# Extra request kwargs for endpoints that require a body
_ENDPOINT_KWARGS: dict[tuple[str, str, str], dict] = {
    ('auth', 'PATCH', '/api/v1alpha/system_roles/nonexistent-permission-test-user'): {
        'json': {'role_addition': 'developer'}
    },
}


def _request(session, service: str, method: str, path: str):
    url = deploy_config.url(service, path)
    extra = _ENDPOINT_KWARGS.get((service, method, path), {})
    return retry_response_returning_functions(getattr(session, method.lower()), url, allow_redirects=False, **extra)


def _assert_denied(session, service: str, method: str, path: str):
    r = _request(session, service, method, path)
    assert r.status_code == 401, f"{method} {service}{path}: expected 401 (permission denied), got {r.status_code}"


def _assert_granted(session, service: str, method: str, path: str):
    r = _request(session, service, method, path)
    assert r.status_code != 401, f"{method} {service}{path}: expected not-401 (permission granted), got {r.status_code}"


# ---------------------------------------------------------------------------
# Role fixture
# ---------------------------------------------------------------------------


@contextmanager
def _with_role(role: str):
    """Grant a role to the `test` user for the duration of the block, then revoke it.

    Sleeps 11 seconds after both granting and revoking whenever the role includes
    cross-service permissions, to let the AuthServiceAuthenticator cache expire.
    """
    granted_perms = ROLE_PERMISSIONS.get(role, set())
    needs_cross_service_sleep = bool(granted_perms & _CROSS_SERVICE_PERMS)

    asyncio.run(async_add_role('test', role))
    if needs_cross_service_sleep:
        time.sleep(11)
    try:
        yield
    finally:
        asyncio.run(async_remove_role('test', role))
        if needs_cross_service_sleep:
            time.sleep(11)


def _check_all_denied(session):
    """Assert every permission-gated endpoint returns 401 for this session."""
    for perm, endpoints in PERMISSION_ENDPOINTS.items():
        for service, method, path in endpoints:
            _assert_denied(session, service, method, path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_unauthenticated_denied():
    """Requests with no bearer token must be rejected (401) on all permission-gated endpoints."""
    _check_all_denied(NO_AUTH_SESSION)


def test_no_roles_denied():
    """Authenticated `test` user with no roles must be denied (401) on all permission-gated endpoints."""
    _check_all_denied(TEST_USER_SESSION)


def test_sysadmin_readonly():
    """sysadmin-readonly role grants: read_users, read_system_roles, read_ci, view_monitoring_dashboards."""
    role = 'sysadmin-readonly'
    granted = ROLE_PERMISSIONS[role]
    denied = set(PERMISSION_ENDPOINTS.keys()) - granted

    with _with_role(role):
        for perm in granted:
            for service, method, path in PERMISSION_ENDPOINTS[perm]:
                _assert_granted(TEST_USER_SESSION, service, method, path)
        for perm in denied:
            for service, method, path in PERMISSION_ENDPOINTS[perm]:
                _assert_denied(TEST_USER_SESSION, service, method, path)


def test_billing_manager():
    """billing_manager role grants: read_users, create/delete billing projects, assign users to billing projects."""
    role = 'billing_manager'
    granted = ROLE_PERMISSIONS[role]
    denied = set(PERMISSION_ENDPOINTS.keys()) - granted

    with _with_role(role):
        for perm in granted:
            for service, method, path in PERMISSION_ENDPOINTS[perm]:
                _assert_granted(TEST_USER_SESSION, service, method, path)
        for perm in denied:
            for service, method, path in PERMISSION_ENDPOINTS[perm]:
                _assert_denied(TEST_USER_SESSION, service, method, path)


def test_developer():
    """developer role grants: read_ci, manage_ci."""
    role = 'developer'
    granted = ROLE_PERMISSIONS[role]
    denied = set(PERMISSION_ENDPOINTS.keys()) - granted

    with _with_role(role):
        for perm in granted:
            for service, method, path in PERMISSION_ENDPOINTS[perm]:
                _assert_granted(TEST_USER_SESSION, service, method, path)
        for perm in denied:
            for service, method, path in PERMISSION_ENDPOINTS[perm]:
                _assert_denied(TEST_USER_SESSION, service, method, path)


def test_sysadmin():
    """sysadmin role grants: read_users, delete_users, assign_system_roles, read_system_roles, view_monitoring_dashboards."""
    role = 'sysadmin'
    granted = ROLE_PERMISSIONS[role]
    denied = set(PERMISSION_ENDPOINTS.keys()) - granted

    with _with_role(role):
        for perm in granted:
            for service, method, path in PERMISSION_ENDPOINTS[perm]:
                _assert_granted(TEST_USER_SESSION, service, method, path)
        for perm in denied:
            for service, method, path in PERMISSION_ENDPOINTS[perm]:
                _assert_denied(TEST_USER_SESSION, service, method, path)
