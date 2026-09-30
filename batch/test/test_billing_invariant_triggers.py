"""Tests for the quote / billing project invariant triggers in batch/sql/122-billing-quotes.sql.

These write directly to the database, bypassing the application-level checks in
billing_project_management.py, to show that the database rejects invalid states on its own.

Requires local MySQL (make local-mysql). The db fixture is provided by conftest.py.
"""

import pymysql
import pytest
import pytest_asyncio

from batch.billing_project_management import INTERNAL_QUOTE_ID

# 1644 ER_SIGNAL_EXCEPTION: raised by SIGNAL SQLSTATE '45000'
ER_SIGNAL_EXCEPTION = 1644


@pytest_asyncio.fixture(autouse=True)
async def clean_tables(db):
    yield
    async with db.start() as tx:
        await tx.just_execute('DELETE FROM billing_project_events')
        await tx.just_execute('DELETE FROM billing_project_users')
        await tx.just_execute('DELETE FROM billing_projects')
        await tx.just_execute('DELETE FROM quote_managers')
        await tx.just_execute('DELETE FROM quote_events')
        await tx.just_execute("DELETE FROM quotes WHERE name != 'INTERNAL'")


async def _execute(db, sql, args=()):
    async with db.start() as tx:
        await tx.just_execute(sql, args)


async def _insert_quote(db, name, authorized_amount, state='open') -> int:
    async with db.start() as tx:
        return await tx.execute_insertone(
            """
INSERT INTO quotes (name, name_cs, cost_object, authorized_amount, state, time_created)
VALUES (%s, %s, 'CO', %s, %s, 0);
""",
            (name, name, authorized_amount, state),
        )


async def _insert_bp(db, name, quote_id, limit, status='open'):
    await _execute(
        db,
        'INSERT INTO billing_projects (name, name_cs, quote_id, `limit`, `status`) VALUES (%s, %s, %s, %s, %s);',
        (name, name, quote_id, limit, status),
    )


def _assert_rejected(exc_info, message):
    err = exc_info.value
    assert err.args[0] == ER_SIGNAL_EXCEPTION, err.args
    assert message in err.args[1], err.args


# ---------------------------------------------------------------------------
# quotes
# ---------------------------------------------------------------------------


async def test_insert_quote_without_amount_rejected(db):
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _insert_quote(db, 'q-null', None)
    _assert_rejected(exc_info, 'only the INTERNAL quote may be unlimited')


async def test_insert_quote_negative_amount_rejected(db):
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _insert_quote(db, 'q-neg', -1.0)
    _assert_rejected(exc_info, 'must be non-negative')


async def test_update_quote_amount_to_null_rejected(db):
    quote_id = await _insert_quote(db, 'q-to-null', 100.0)
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _execute(db, 'UPDATE quotes SET authorized_amount = NULL WHERE id = %s;', (quote_id,))
    _assert_rejected(exc_info, 'only the INTERNAL quote may be unlimited')


async def test_rename_internal_rejected(db):
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _execute(
            db, "UPDATE quotes SET name = 'NOT-INTERNAL', name_cs = 'NOT-INTERNAL' WHERE id = %s;", (INTERNAL_QUOTE_ID,)
        )
    _assert_rejected(exc_info, 'cannot be renamed')


async def test_update_quote_amount_below_bp_limits_rejected(db):
    quote_id = await _insert_quote(db, 'q-shrink', 500.0)
    await _insert_bp(db, 'bp-shrink', quote_id, 400.0)
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _execute(db, 'UPDATE quotes SET authorized_amount = 300.0 WHERE id = %s;', (quote_id,))
    _assert_rejected(exc_info, 'cannot be less than the sum')


async def test_cap_internal_with_unlimited_bp_rejected(db):
    await _insert_bp(db, 'bp-internal-ul', INTERNAL_QUOTE_ID, None)
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _execute(db, 'UPDATE quotes SET authorized_amount = 1000000.0 WHERE id = %s;', (INTERNAL_QUOTE_ID,))
    _assert_rejected(exc_info, 'unlimited billing projects')


async def test_close_quote_with_open_bp_rejected(db):
    quote_id = await _insert_quote(db, 'q-close-open', 500.0)
    await _insert_bp(db, 'bp-close-open', quote_id, 100.0)
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _execute(db, "UPDATE quotes SET state = 'closed' WHERE id = %s;", (quote_id,))
    _assert_rejected(exc_info, 'cannot be closed while it has open billing projects')


async def test_close_quote_with_closed_bp_allowed(db):
    quote_id = await _insert_quote(db, 'q-close-closed', 500.0)
    await _insert_bp(db, 'bp-close-closed', quote_id, 100.0, status='closed')
    await _execute(db, "UPDATE quotes SET state = 'closed' WHERE id = %s;", (quote_id,))
    row = await db.select_and_fetchone('SELECT state FROM quotes WHERE id = %s;', (quote_id,))
    assert row['state'] == 'closed'


# ---------------------------------------------------------------------------
# billing_projects
# ---------------------------------------------------------------------------


async def test_insert_unlimited_bp_under_internal_allowed(db):
    await _insert_bp(db, 'bp-ul-ok', INTERNAL_QUOTE_ID, None)
    row = await db.select_and_fetchone('SELECT `limit` FROM billing_projects WHERE name = %s;', ('bp-ul-ok',))
    assert row['limit'] is None


async def test_insert_bp_without_quote_defaults_to_internal(db):
    # The legacy create path omits quote_id; it must land in INTERNAL, where unlimited is allowed.
    await _execute(db, 'INSERT INTO billing_projects (name, name_cs) VALUES (%s, %s);', ('bp-legacy', 'bp-legacy'))
    row = await db.select_and_fetchone('SELECT quote_id FROM billing_projects WHERE name = %s;', ('bp-legacy',))
    assert row['quote_id'] == INTERNAL_QUOTE_ID


async def test_insert_unlimited_bp_under_other_quote_rejected(db):
    quote_id = await _insert_quote(db, 'q-bp-ul', 500.0)
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _insert_bp(db, 'bp-ul-bad', quote_id, None)
    _assert_rejected(exc_info, 'may be unlimited')


async def test_update_bp_limit_to_null_under_other_quote_rejected(db):
    quote_id = await _insert_quote(db, 'q-bp-to-null', 500.0)
    await _insert_bp(db, 'bp-to-null', quote_id, 100.0)
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _execute(db, 'UPDATE billing_projects SET `limit` = NULL WHERE name = %s;', ('bp-to-null',))
    _assert_rejected(exc_info, 'may be unlimited')


async def test_insert_bp_with_missing_quote_rejected(db):
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _insert_bp(db, 'bp-no-quote', 999999, 100.0)
    _assert_rejected(exc_info, 'quote does not exist')


async def test_insert_bp_negative_limit_rejected(db):
    quote_id = await _insert_quote(db, 'q-bp-neg', 500.0)
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _insert_bp(db, 'bp-neg', quote_id, -1.0)
    _assert_rejected(exc_info, 'must be non-negative')


async def test_insert_bp_exceeding_quote_rejected(db):
    quote_id = await _insert_quote(db, 'q-bp-sum', 300.0)
    await _insert_bp(db, 'bp-sum-1', quote_id, 200.0)
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _insert_bp(db, 'bp-sum-2', quote_id, 150.0)
    _assert_rejected(exc_info, 'would exceed quote authorized_amount')


async def test_update_bp_limit_exceeding_quote_rejected(db):
    quote_id = await _insert_quote(db, 'q-bp-sum-upd', 300.0)
    await _insert_bp(db, 'bp-sum-upd-1', quote_id, 200.0)
    await _insert_bp(db, 'bp-sum-upd-2', quote_id, 50.0)
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _execute(db, 'UPDATE billing_projects SET `limit` = 150.0 WHERE name = %s;', ('bp-sum-upd-2',))
    _assert_rejected(exc_info, 'would exceed quote authorized_amount')


async def test_deleted_bp_limits_do_not_count(db):
    quote_id = await _insert_quote(db, 'q-bp-deleted', 300.0)
    await _insert_bp(db, 'bp-deleted', quote_id, 250.0, status='deleted')
    await _insert_bp(db, 'bp-live', quote_id, 250.0)
    row = await db.select_and_fetchone('SELECT `limit` FROM billing_projects WHERE name = %s;', ('bp-live',))
    assert row['limit'] == 250.0


async def test_move_bp_exceeding_dest_quote_rejected(db):
    src_id = await _insert_quote(db, 'q-move-src', 500.0)
    dest_id = await _insert_quote(db, 'q-move-dest', 100.0)
    await _insert_bp(db, 'bp-move', src_id, 200.0)
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _execute(db, 'UPDATE billing_projects SET quote_id = %s WHERE name = %s;', (dest_id, 'bp-move'))
    _assert_rejected(exc_info, 'would exceed quote authorized_amount')


async def test_insert_open_bp_under_closed_quote_rejected(db):
    quote_id = await _insert_quote(db, 'q-closed-insert', 500.0, state='closed')
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _insert_bp(db, 'bp-closed-insert', quote_id, 100.0)
    _assert_rejected(exc_info, 'under a closed quote')


async def test_reopen_bp_under_closed_quote_rejected(db):
    quote_id = await _insert_quote(db, 'q-closed-reopen', 500.0)
    await _insert_bp(db, 'bp-closed-reopen', quote_id, 100.0, status='closed')
    await _execute(db, "UPDATE quotes SET state = 'closed' WHERE id = %s;", (quote_id,))
    with pytest.raises(pymysql.err.OperationalError) as exc_info:
        await _execute(db, "UPDATE billing_projects SET `status` = 'open' WHERE name = %s;", ('bp-closed-reopen',))
    _assert_rejected(exc_info, 'under a closed quote')
