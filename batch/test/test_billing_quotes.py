"""Integration tests for billing quotes (Phase 1a).

Requires a running Batch deployment with billing-quote schema migrations applied.
Run via: pytest batch/test/test_billing_quotes.py
"""

import secrets

import pytest

from hailtop.batch_client.client import BatchClient


def random_name(prefix='test'):
    return f'{prefix}-{secrets.token_hex(6)}'


@pytest.fixture(scope='module')
def global_bm_client():
    """A BatchClient authenticated as a global billing manager."""
    client = BatchClient('test')
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Quote CRUD (global BM operations)
# ---------------------------------------------------------------------------


def test_create_and_get_quote(global_bm_client):
    name = random_name('q')
    global_bm_client.create_quote(name, cost_object='test-cost-object', authorized_amount=1000.0)
    q = global_bm_client.get_quote(name)
    assert q['name'] == name
    assert q['cost_object'] == 'test-cost-object'
    assert q['authorized_amount'] == 1000.0


def test_list_quotes_includes_new_quote(global_bm_client):
    name = random_name('q')
    global_bm_client.create_quote(name, cost_object='co-list')
    quotes = global_bm_client.list_quotes()
    names = [q['name'] for q in quotes]
    assert name in names


def test_create_quote_duplicate_fails(global_bm_client):
    name = random_name('q')
    global_bm_client.create_quote(name, cost_object='co1')
    with pytest.raises(Exception):
        global_bm_client.create_quote(name, cost_object='co2')


def test_edit_quote(global_bm_client):
    name = random_name('q')
    global_bm_client.create_quote(name, cost_object='original', authorized_amount=500.0)
    global_bm_client.edit_quote(name, cost_object='updated', authorized_amount=750.0)
    q = global_bm_client.get_quote(name)
    assert q['cost_object'] == 'updated'
    assert q['authorized_amount'] == 750.0


def test_edit_quote_authorized_amount_below_bp_limits_rejected(global_bm_client):
    name = random_name('q')
    bp_name = random_name('bp')
    global_bm_client.create_quote(name, cost_object='co', authorized_amount=500.0)
    global_bm_client.create_billing_project_v2(bp_name, quote_name=name, limit=400.0)
    with pytest.raises(Exception):
        global_bm_client.edit_quote(name, authorized_amount=300.0)
    global_bm_client.close_billing_project(bp_name)


def test_add_and_remove_quote_manager(global_bm_client):
    name = random_name('q')
    global_bm_client.create_quote(name, cost_object='co')
    global_bm_client.add_quote_manager(name, 'test', role='manager')
    q = global_bm_client.get_quote(name)
    managers = {m['user']: m['role'] for m in q['managers']}
    assert managers.get('test') == 'manager'

    global_bm_client.remove_quote_manager(name, 'test')
    q = global_bm_client.get_quote(name)
    managers = {m['user']: m['role'] for m in q['managers']}
    assert 'test' not in managers


def test_quote_events_written(global_bm_client):
    name = random_name('q')
    global_bm_client.create_quote(name, cost_object='co')
    global_bm_client.add_quote_manager(name, 'test', role='manager')
    global_bm_client.remove_quote_manager(name, 'test')
    events = global_bm_client.get_quote_events(name)
    actions = [e['action'] for e in events]
    assert 'quote_created' in actions
    assert 'manager_added' in actions
    assert 'manager_removed' in actions


# ---------------------------------------------------------------------------
# Billing project under a quote
# ---------------------------------------------------------------------------


def test_create_bp_under_quote(global_bm_client):
    q_name = random_name('q')
    bp_name = random_name('bp')
    global_bm_client.create_quote(q_name, cost_object='co', authorized_amount=1000.0)
    global_bm_client.create_billing_project_v2(bp_name, quote_name=q_name, limit=100.0)

    q = global_bm_client.get_quote(q_name)
    bp_names = [bp['billing_project'] for bp in q['billing_projects']]
    assert bp_name in bp_names

    global_bm_client.close_billing_project(bp_name)


def test_bp_limit_exceeding_quote_rejected(global_bm_client):
    q_name = random_name('q')
    bp_name = random_name('bp')
    global_bm_client.create_quote(q_name, cost_object='co', authorized_amount=200.0)
    global_bm_client.create_billing_project_v2(bp_name, quote_name=q_name, limit=150.0)
    with pytest.raises(Exception):
        global_bm_client.create_billing_project_v2(random_name('bp'), quote_name=q_name, limit=100.0)
    global_bm_client.close_billing_project(bp_name)


def test_patch_bp_limit(global_bm_client):
    q_name = random_name('q')
    bp_name = random_name('bp')
    global_bm_client.create_quote(q_name, cost_object='co', authorized_amount=500.0)
    global_bm_client.create_billing_project_v2(bp_name, quote_name=q_name, limit=100.0)
    global_bm_client.patch_billing_project(bp_name, limit=200.0)
    bp = global_bm_client.get_billing_project(bp_name)
    assert bp['limit'] == 200.0
    global_bm_client.close_billing_project(bp_name)


def test_patch_bp_limit_exceeding_quote_rejected(global_bm_client):
    q_name = random_name('q')
    bp_name1 = random_name('bp')
    bp_name2 = random_name('bp')
    global_bm_client.create_quote(q_name, cost_object='co', authorized_amount=300.0)
    global_bm_client.create_billing_project_v2(bp_name1, quote_name=q_name, limit=200.0)
    global_bm_client.create_billing_project_v2(bp_name2, quote_name=q_name, limit=50.0)
    with pytest.raises(Exception):
        global_bm_client.patch_billing_project(bp_name2, limit=200.0)
    global_bm_client.close_billing_project(bp_name1)
    global_bm_client.close_billing_project(bp_name2)


def test_patch_bp_alert(global_bm_client):
    q_name = random_name('q')
    bp_name = random_name('bp')
    global_bm_client.create_quote(q_name, cost_object='co')
    global_bm_client.create_billing_project_v2(bp_name, quote_name=q_name)
    global_bm_client.patch_billing_project(bp_name, low_budget_alert=50.0)
    bp = global_bm_client.get_billing_project(bp_name)
    assert bp['low_budget_alert'] == 50.0
    global_bm_client.close_billing_project(bp_name)


def test_change_bp_quote(global_bm_client):
    q1_name = random_name('q')
    q2_name = random_name('q')
    bp_name = random_name('bp')
    global_bm_client.create_quote(q1_name, cost_object='co1', authorized_amount=500.0)
    global_bm_client.create_quote(q2_name, cost_object='co2', authorized_amount=500.0)
    global_bm_client.create_billing_project_v2(bp_name, quote_name=q1_name, limit=100.0)

    global_bm_client.change_billing_project_quote(bp_name, q2_name)
    bp = global_bm_client.get_billing_project(bp_name)
    assert bp['quote_name'] == q2_name
    global_bm_client.close_billing_project(bp_name)


def test_bp_events_written(global_bm_client):
    q_name = random_name('q')
    bp_name = random_name('bp')
    global_bm_client.create_quote(q_name, cost_object='co')
    global_bm_client.create_billing_project_v2(bp_name, quote_name=q_name)
    global_bm_client.add_user('test', bp_name)
    global_bm_client.remove_user('test', bp_name)
    events = global_bm_client.get_billing_project_events(bp_name)
    actions = [e['action'] for e in events]
    assert len(actions) > 0
    global_bm_client.close_billing_project(bp_name)


# ---------------------------------------------------------------------------
# Role boundary checks
# ---------------------------------------------------------------------------


def test_internal_quote_accessible_via_internal_bm(global_bm_client):
    q = global_bm_client.get_quote('INTERNAL')
    assert q['name'] == 'INTERNAL'


def test_unlisted_quote_not_visible_to_non_member(global_bm_client):
    """A user with no role on a quote should get 403 on get_quote."""
    q_name = random_name('q')
    global_bm_client.create_quote(q_name, cost_object='co')

    non_member_client = BatchClient('test')
    try:
        with pytest.raises(Exception) as exc_info:
            non_member_client.get_quote(q_name)
        assert '403' in str(exc_info.value) or 'Forbidden' in str(exc_info.value)
    finally:
        non_member_client.close()
