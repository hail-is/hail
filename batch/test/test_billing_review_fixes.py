"""Tests for the #15645 review fixes: lock ordering, close/move/delete edge cases, trigger error mapping, RESTRICT."""

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pymysql
import pytest

from batch import billing_project_management
from batch.billing_project_management import (
    get_billing_project_events,
    get_quote_events,
)
from batch.exceptions import BatchOperationAlreadyCompletedError, BillingValidationError

from .billing_writes import (
    change_billing_project_quote,
    close_billing_project,
    create_billing_project,
    create_quote,
    delete_billing_project,
    edit_quote,
    patch_billing_project,
)


async def _quote(db, name, authorized_amount=1000.0) -> int:
    await create_quote(db, name, cost_object='CO', actor='admin', authorized_amount=authorized_amount)
    row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name = %s', (name,))
    return row['id']


async def _status(db, bp) -> str:
    row = await db.select_and_fetchone('SELECT `status` FROM billing_projects WHERE name = %s', (bp,))
    return row['status']


async def test_close_does_not_undelete(db, unique_name):
    q, bp = unique_name('q-close-del'), unique_name('bp-close-del')
    await create_billing_project(db, bp, await _quote(db, q), 10.0, 'admin', 'global_bm')
    await close_billing_project(db, bp, 'admin')
    await delete_billing_project(db, bp, 'admin')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await close_billing_project(db, bp, 'admin')
    assert await _status(db, bp) == 'deleted'


async def test_change_quote_to_current_quote_is_noop(db, unique_name):
    q, bp = unique_name('q-same'), unique_name('bp-same')
    # The BP uses the whole quote, so double-counting its limit would fail the capacity check.
    quote_id = await _quote(db, q, authorized_amount=100.0)
    await create_billing_project(db, bp, quote_id, 100.0, 'admin', 'global_bm')
    with pytest.raises(BatchOperationAlreadyCompletedError):
        await change_billing_project_quote(db, bp, quote_id, 'admin')
    events = await get_quote_events(db, q)
    assert events is not None
    actions = [e['action'] for e in events]
    assert 'bp_assigned' not in actions and 'bp_unassigned' not in actions


async def test_delete_writes_audit_events(db, unique_name):
    q, bp = unique_name('q-del-audit'), unique_name('bp-del-audit')
    await create_billing_project(db, bp, await _quote(db, q), 10.0, 'admin', 'global_bm')
    await close_billing_project(db, bp, 'admin')
    await delete_billing_project(db, bp, 'alice', comment='done')
    bp_event = next(e for e in await get_billing_project_events(db, bp) if e['action'] == 'bp_deleted')
    assert (bp_event['actor'], bp_event['comment']) == ('alice', 'done')
    quote_events = await get_quote_events(db, q)
    assert quote_events is not None
    quote_event = next(e for e in quote_events if e['action'] == 'bp_deleted')
    assert (quote_event['actor'], quote_event['target_project']) == ('alice', bp)


@pytest.mark.parametrize('op', ['create', 'patch', 'move', 'edit_quote'])
async def test_trigger_violations_surface_as_user_errors(db, unique_name, op):
    """With the app-level sum check fooled, the DB trigger still rejects the write, as a 400 not a 500."""
    q, q2, bp, bp2 = unique_name('q-trig'), unique_name('q-trig2'), unique_name('bp-trig'), unique_name('bp-trig2')
    quote_id = await _quote(db, q, authorized_amount=100.0)
    await create_billing_project(db, bp, quote_id, 100.0, 'admin', 'global_bm')
    with patch.object(billing_project_management, '_sum_of_limits', AsyncMock(return_value=0)):
        with pytest.raises(BillingValidationError, match='Billing invariant violated'):
            if op == 'create':
                await create_billing_project(db, bp2, quote_id, 1.0, 'admin', 'global_bm')
            elif op == 'move':
                await create_billing_project(db, bp2, await _quote(db, q2), 1.0, 'admin', 'global_bm')
                await change_billing_project_quote(db, bp2, quote_id, 'admin')
            elif op == 'patch':
                await patch_billing_project(db, bp, {'limit': 50.0}, 'admin', 'global_bm')
                await create_billing_project(db, bp2, quote_id, 50.0, 'admin', 'global_bm')
                await patch_billing_project(db, bp, {'limit': 60.0}, 'admin', 'global_bm')
            else:
                await edit_quote(db, q, {'authorized_amount': 50.0}, 'admin', 'global_bm')


async def test_events_block_hard_delete(db, unique_name):
    q, bp = unique_name('q-restrict'), unique_name('bp-restrict')
    await create_billing_project(db, bp, await _quote(db, q), 10.0, 'admin', 'global_bm')
    with pytest.raises(pymysql.err.IntegrityError) as exc_info:
        await db.just_execute('DELETE FROM billing_projects WHERE name = %s', (bp,))
    assert exc_info.value.args[0] == 1451  # ER_ROW_IS_REFERENCED_2


async def test_concurrent_writes_under_one_quote_do_not_deadlock(db, unique_name, caplog):
    q = unique_name('q-deadlock')
    quote_id = await _quote(db, q, authorized_amount=100_000.0)
    bps = [unique_name(f'bp-deadlock-{i}') for i in range(4)]
    for bp in bps:
        await create_billing_project(db, bp, quote_id, 100.0, 'admin', 'global_bm')

    async def edit_limits(i):
        for j in range(15):
            await patch_billing_project(db, bps[(i + j) % len(bps)], {'limit': 100.0 + i + j}, 'admin', 'global_bm')

    async def edit_amount():
        for j in range(15):
            await edit_quote(db, q, {'authorized_amount': 100_000.0 + j}, 'admin', 'global_bm')

    with caplog.at_level(logging.INFO, logger='gear.database'):
        await asyncio.gather(*(edit_limits(i) for i in range(6)), edit_amount(), edit_amount())
    deadlocks = [r for r in caplog.records if 'Deadlock' in r.getMessage()]
    assert not deadlocks, f'{len(deadlocks)} deadlocks retried'

    total = await db.select_and_fetchone(
        "SELECT SUM(`limit`) AS s FROM billing_projects WHERE quote_id = %s AND `status` != 'deleted'", (quote_id,)
    )
    assert total['s'] <= 100_000.0
