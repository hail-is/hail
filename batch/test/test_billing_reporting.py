"""Unit tests for billing_reporting against a real MySQL instance.

Requires local MySQL (make local-mysql). The db fixture is provided by conftest.py.
"""

import datetime

import pytest

from batch.billing_project_management import (
    add_billing_project_user,
    add_quote_manager,
    create_billing_project,
    create_quote,
)
from batch.billing_reporting import (
    query_billing_breakdown,
    query_billing_history,
    query_billing_projects_with_cost,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_bp(db, quote_name, bp_name, limit=100.0):
    await create_quote(db, quote_name, cost_object='CO', actor='admin', authorized_amount=1000.0)
    q_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (quote_name,))
    await create_billing_project(db, bp_name, q_row['id'], limit, 'admin', 'global_bm')
    return q_row['id']


async def _insert_resource(db, resource_name, rate):
    async with db.start() as tx:
        return await tx.execute_insertone(
            'INSERT INTO resources (resource, rate) VALUES (%s, %s)',
            (resource_name, rate),
        )


async def _insert_spend(db, billing_date, billing_project, user, resource_id, usage):
    async with db.start() as tx:
        await tx.just_execute(
            """INSERT INTO aggregated_billing_project_user_resources_by_date_v3
               (billing_date, billing_project, `user`, resource_id, token, `usage`)
               VALUES (%s, %s, %s, %s, 0, %s) AS new_row
               ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_by_date_v3.`usage` + new_row.`usage`""",
            (billing_date, billing_project, user, resource_id, usage),
        )


_JAN1 = datetime.datetime(2026, 1, 1)
_JAN31 = datetime.datetime(2026, 1, 31)
_JAN15 = datetime.date(2026, 1, 15)


# ---------------------------------------------------------------------------
# query_billing_projects_with_cost — users and user_roles fields
# ---------------------------------------------------------------------------


async def test_query_bp_users_bp_member(db, unique_name):
    q_users_bpm = unique_name('q-users-bpm')
    bp_users_bpm = unique_name('bp-users-bpm')
    await _make_bp(db, q_users_bpm, bp_users_bpm)
    await add_billing_project_user(db, bp_users_bpm, 'luna', 'admin')
    results = await query_billing_projects_with_cost(db, billing_project=bp_users_bpm)
    assert len(results) == 1
    users = results[0]['user_roles']
    luna = next((u for u in users if u['user'] == 'luna'), None)
    assert luna is not None
    assert f'{bp_users_bpm}:member' in luna['roles']


async def test_query_bp_users_quote_manager_appears_without_direct_membership(db, unique_name):
    q_users_qm = unique_name('q-users-qm')
    bp_users_qm = unique_name('bp-users-qm')
    await _make_bp(db, q_users_qm, bp_users_qm)
    await add_quote_manager(db, q_users_qm, 'mars', 'owner', actor='admin')
    results = await query_billing_projects_with_cost(db, billing_project=bp_users_qm)
    assert len(results) == 1
    users = results[0]['user_roles']
    mars = next((u for u in users if u['user'] == 'mars'), None)
    assert mars is not None
    assert f'{q_users_qm}:owner' in mars['roles']


async def test_query_bp_users_dual_role(db, unique_name):
    q_users_dual = unique_name('q-users-dual')
    bp_users_dual = unique_name('bp-users-dual')
    await _make_bp(db, q_users_dual, bp_users_dual)
    await add_quote_manager(db, q_users_dual, 'nova', 'manager', actor='admin')
    await add_billing_project_user(db, bp_users_dual, 'nova', 'admin')
    results = await query_billing_projects_with_cost(db, billing_project=bp_users_dual)
    assert len(results) == 1
    users = results[0]['user_roles']
    nova = next((u for u in users if u['user'] == 'nova'), None)
    assert nova is not None
    assert f'{bp_users_dual}:member' in nova['roles']
    assert f'{q_users_dual}:manager' in nova['roles']


async def test_query_bp_users_is_member_names_only(db, unique_name):
    # `users` keeps its pre-quotes shape: plain member usernames, without quote managers.
    q_users_names = unique_name('q-users-names')
    bp_users_names = unique_name('bp-users-names')
    await _make_bp(db, q_users_names, bp_users_names)
    await add_billing_project_user(db, bp_users_names, 'orion', 'admin')
    await add_quote_manager(db, q_users_names, 'pax', 'owner', actor='admin')
    results = await query_billing_projects_with_cost(db, billing_project=bp_users_names)
    assert results[0]['users'] == ['orion']


async def test_query_bp_users_outsider_absent(db, unique_name):
    q_users_out = unique_name('q-users-out')
    bp_users_out = unique_name('bp-users-out')
    await _make_bp(db, q_users_out, bp_users_out)
    results = await query_billing_projects_with_cost(db, billing_project=bp_users_out)
    assert len(results) == 1
    users = results[0]['user_roles']
    assert not any(u['user'] == 'outsider' for u in users)


async def test_can_view_quote_global_bm(db, unique_name):
    q_cvq_gbm = unique_name('q-cvq-gbm')
    bp_cvq_gbm = unique_name('bp-cvq-gbm')
    await _make_bp(db, q_cvq_gbm, bp_cvq_gbm)
    results = await query_billing_projects_with_cost(db, billing_project=bp_cvq_gbm, quote_manager_user=None)
    assert results[0]['can_view_quote'] is True


async def test_can_view_quote_quote_manager(db, unique_name):
    q_cvq_qm = unique_name('q-cvq-qm')
    bp_cvq_qm = unique_name('bp-cvq-qm')
    await _make_bp(db, q_cvq_qm, bp_cvq_qm)
    await add_quote_manager(db, q_cvq_qm, 'piuser', 'manager', actor='admin')
    results = await query_billing_projects_with_cost(db, billing_project=bp_cvq_qm, quote_manager_user='piuser')
    assert results[0]['can_view_quote'] is True


async def test_can_view_quote_bp_member_only(db, unique_name):
    q_cvq_bpm = unique_name('q-cvq-bpm')
    bp_cvq_bpm = unique_name('bp-cvq-bpm')
    await _make_bp(db, q_cvq_bpm, bp_cvq_bpm)
    await add_billing_project_user(db, bp_cvq_bpm, 'regular', 'admin')
    results = await query_billing_projects_with_cost(
        db, user='regular', billing_project=bp_cvq_bpm, quote_manager_user='regular'
    )
    assert results[0]['can_view_quote'] is False


# ---------------------------------------------------------------------------
# query_billing_history — access control
# ---------------------------------------------------------------------------


async def test_billing_history_global_bm_sees_all(db, unique_name):
    q_hist_all = unique_name('q-hist-all')
    bp_hist_all = unique_name('bp-hist-all')
    cpu = unique_name('cpu')
    await _make_bp(db, q_hist_all, bp_hist_all)
    resource_id = await _insert_resource(db, cpu, 0.01)
    await _insert_spend(db, _JAN15, bp_hist_all, 'alice', resource_id, 100)
    await _insert_spend(db, _JAN15, bp_hist_all, 'bob', resource_id, 200)

    rows = await query_billing_history(db, _JAN1, _JAN31, user=None, quote_manager_user=None)
    users_seen = {r['user'] for r in rows if r['billing_project'] == bp_hist_all}
    assert 'alice' in users_seen
    assert 'bob' in users_seen


async def test_billing_history_regular_user_sees_only_own_spend(db, unique_name):
    q_hist_own = unique_name('q-hist-own')
    bp_hist_own = unique_name('bp-hist-own')
    cpu_own = unique_name('cpu-own')
    await _make_bp(db, q_hist_own, bp_hist_own)
    resource_id = await _insert_resource(db, cpu_own, 0.01)
    await _insert_spend(db, _JAN15, bp_hist_own, 'alice', resource_id, 100)
    await _insert_spend(db, _JAN15, bp_hist_own, 'bob', resource_id, 200)

    rows = await query_billing_history(db, _JAN1, _JAN31, user='alice', quote_manager_user=None)
    users_seen = {r['user'] for r in rows}
    assert 'alice' in users_seen
    assert 'bob' not in users_seen


async def test_billing_history_quote_manager_sees_all_spend_in_managed_quote(db, unique_name):
    q_hist_qm = unique_name('q-hist-qm')
    bp_hist_qm = unique_name('bp-hist-qm')
    cpu_qm = unique_name('cpu-qm')
    await _make_bp(db, q_hist_qm, bp_hist_qm)
    await add_quote_manager(db, q_hist_qm, 'manager', 'manager', actor='admin')
    resource_id = await _insert_resource(db, cpu_qm, 0.01)
    await _insert_spend(db, _JAN15, bp_hist_qm, 'alice', resource_id, 100)
    await _insert_spend(db, _JAN15, bp_hist_qm, 'bob', resource_id, 200)

    rows = await query_billing_history(db, _JAN1, _JAN31, user='manager', quote_manager_user='manager')
    users_seen = {r['user'] for r in rows if r['billing_project'] == bp_hist_qm}
    assert 'alice' in users_seen
    assert 'bob' in users_seen


async def test_billing_history_quote_manager_also_sees_own_spend_in_other_projects(db, unique_name):
    q_hist_other = unique_name('q-hist-other')
    bp_hist_other = unique_name('bp-hist-other')
    cpu_other = unique_name('cpu-other')
    await _make_bp(db, q_hist_other, bp_hist_other)
    resource_id = await _insert_resource(db, cpu_other, 0.01)
    await _insert_spend(db, _JAN15, bp_hist_other, 'manager', resource_id, 50)

    rows = await query_billing_history(db, _JAN1, _JAN31, user='manager', quote_manager_user='manager')
    projects_seen = {r['billing_project'] for r in rows if r['user'] == 'manager'}
    assert bp_hist_other in projects_seen


async def test_billing_history_non_manager_cannot_see_others_spend(db, unique_name):
    q_hist_excl = unique_name('q-hist-excl')
    bp_hist_excl = unique_name('bp-hist-excl')
    cpu_excl = unique_name('cpu-excl')
    await _make_bp(db, q_hist_excl, bp_hist_excl)
    resource_id = await _insert_resource(db, cpu_excl, 0.01)
    await _insert_spend(db, _JAN15, bp_hist_excl, 'alice', resource_id, 100)

    rows = await query_billing_history(db, _JAN1, _JAN31, user='outsider', quote_manager_user=None)
    users_seen = {r['user'] for r in rows if r['billing_project'] == bp_hist_excl}
    assert 'alice' not in users_seen


async def test_billing_history_date_filter_applied(db, unique_name):
    q_hist_date = unique_name('q-hist-date')
    bp_hist_date = unique_name('bp-hist-date')
    cpu_date = unique_name('cpu-date')
    await _make_bp(db, q_hist_date, bp_hist_date)
    resource_id = await _insert_resource(db, cpu_date, 0.01)
    await _insert_spend(db, datetime.date(2025, 12, 31), bp_hist_date, 'alice', resource_id, 999)
    await _insert_spend(db, _JAN15, bp_hist_date, 'alice', resource_id, 100)

    rows = await query_billing_history(db, _JAN1, _JAN31, user='alice', quote_manager_user=None)
    total = sum(r['cost'] for r in rows if r['billing_project'] == bp_hist_date)
    assert pytest.approx(total, abs=0.01) == 100 * 0.01


# ---------------------------------------------------------------------------
# query_billing_breakdown — access control mirrors query_billing_history
# ---------------------------------------------------------------------------


async def test_billing_breakdown_regular_user_sees_only_own(db, unique_name):
    q_brkdn_own = unique_name('q-brkdn-own')
    bp_brkdn_own = unique_name('bp-brkdn-own')
    mem_own = unique_name('mem-own')
    await _make_bp(db, q_brkdn_own, bp_brkdn_own)
    resource_id = await _insert_resource(db, mem_own, 0.005)
    await _insert_spend(db, _JAN15, bp_brkdn_own, 'alice', resource_id, 100)
    await _insert_spend(db, _JAN15, bp_brkdn_own, 'bob', resource_id, 200)

    rows = await query_billing_breakdown(db, _JAN1, _JAN31, user='alice', quote_manager_user=None)
    users_seen = {r['user'] for r in rows}
    assert 'alice' in users_seen
    assert 'bob' not in users_seen


async def test_billing_breakdown_quote_manager_sees_all_in_quote(db, unique_name):
    q_brkdn_qm = unique_name('q-brkdn-qm')
    bp_brkdn_qm = unique_name('bp-brkdn-qm')
    mem_qm = unique_name('mem-qm')
    await _make_bp(db, q_brkdn_qm, bp_brkdn_qm)
    await add_quote_manager(db, q_brkdn_qm, 'mgr', 'manager', actor='admin')
    resource_id = await _insert_resource(db, mem_qm, 0.005)
    await _insert_spend(db, _JAN15, bp_brkdn_qm, 'alice', resource_id, 100)
    await _insert_spend(db, _JAN15, bp_brkdn_qm, 'bob', resource_id, 200)

    rows = await query_billing_breakdown(db, _JAN1, _JAN31, user='mgr', quote_manager_user='mgr')
    users_seen = {r['user'] for r in rows if r['billing_project'] == bp_brkdn_qm}
    assert 'alice' in users_seen
    assert 'bob' in users_seen
