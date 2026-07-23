from typing import Optional, Union

from gear import Database
from hailtop.utils import time_msecs

from .exceptions import (
    BatchOperationAlreadyCompletedError,
    BatchUserError,
    ClosedBillingProjectError,
    InvalidBillingLimitError,
    NonExistentBillingProjectError,
)


def _parse_billing_limit(limit: Optional[Union[str, float, int]]) -> Optional[float]:
    assert isinstance(limit, (str, float, int)) or limit is None, (limit, type(limit))

    if limit == 'None' or limit is None:
        return None
    try:
        parsed_limit = float(limit)
        assert parsed_limit >= 0
        return parsed_limit
    except (AssertionError, ValueError) as e:
        raise InvalidBillingLimitError(limit) from e


async def _log_quote_event(
    tx,
    quote_id: int,
    actor: str,
    action: str,
    target_user=None,
    target_project=None,
    detail=None,
    comment=None,
) -> None:
    await tx.execute_insertone(
        """
INSERT INTO quote_events (quote_id, timestamp, actor, action, target_user, target_project, detail, comment)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
""",
        (quote_id, time_msecs(), actor, action, target_user, target_project, detail, comment),
    )


async def _log_bp_event(
    tx,
    billing_project: str,
    actor: str,
    action: str,
    target_user=None,
    detail=None,
    comment=None,
) -> None:
    await tx.execute_insertone(
        """
INSERT INTO billing_project_events (billing_project, timestamp, actor, action, target_user, detail, comment)
VALUES (%s, %s, %s, %s, %s, %s, %s);
""",
        (billing_project, time_msecs(), actor, action, target_user, detail, comment),
    )


# ---------------------------------------------------------------------------
# Quote reads
# ---------------------------------------------------------------------------


async def list_quotes_for_user(db: Database, username: str, is_global_bm: bool) -> list:
    if is_global_bm:
        return [
            record
            async for record in db.select_and_fetchall(
                'SELECT id, name, cost_object, authorized_amount, pi_name, pm_designee, time_created FROM quotes;'
            )
        ]
    return [
        record
        async for record in db.select_and_fetchall(
            """
SELECT q.id, q.name, q.cost_object, q.authorized_amount, q.pi_name, q.pm_designee, q.time_created
FROM quotes q
INNER JOIN quote_managers qm ON qm.quote_id = q.id
WHERE qm.user = %s;
""",
            (username,),
        )
    ]


async def get_quote(db: Database, name: str) -> Optional[dict]:
    """Return the full quote dict (with managers and billing_projects), or None if not found."""
    quote_row = await db.select_and_fetchone(
        'SELECT id, name, cost_object, authorized_amount, pi_name, pm_designee, time_created FROM quotes WHERE name_cs = %s;',
        (name,),
    )
    if not quote_row:
        return None

    quote_id = quote_row['id']

    managers = [
        record
        async for record in db.select_and_fetchall(
            'SELECT user, role FROM quote_managers WHERE quote_id = %s;', (quote_id,)
        )
    ]

    bps = [
        record
        async for record in db.select_and_fetchall(
            """
SELECT bp.name AS billing_project, bp.`status`, bp.`limit`, bp.low_budget_alert,
  COALESCE(SUM(agg.`usage` * resources.rate), 0) AS accrued_cost,
  IF(bp.`limit` IS NULL, NULL, bp.`limit` - COALESCE(SUM(agg.`usage` * resources.rate), 0)) AS remaining
FROM billing_projects bp
LEFT JOIN aggregated_billing_project_user_resources_v3 agg ON agg.billing_project = bp.name
LEFT JOIN resources ON resources.resource_id = agg.resource_id
WHERE bp.quote_id = %s AND bp.`status` != 'deleted'
GROUP BY bp.name, bp.`status`, bp.`limit`, bp.low_budget_alert;
""",
            (quote_id,),
        )
    ]

    return {**quote_row, 'managers': managers, 'billing_projects': bps}


async def get_quote_events(db: Database, name: str) -> Optional[list]:
    """Return events for the named quote, or None if the quote doesn't exist."""
    quote_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name_cs = %s;', (name,))
    if not quote_row:
        return None

    return [
        record
        async for record in db.select_and_fetchall(
            """
SELECT id, timestamp, actor, action, target_user, target_project, detail, comment
FROM quote_events
WHERE quote_id = %s
ORDER BY timestamp DESC
LIMIT 1000;
""",
            (quote_row['id'],),
        )
    ]


async def get_billing_project_events(db: Database, bp_name: str) -> list:
    return [
        record
        async for record in db.select_and_fetchall(
            """
SELECT id, timestamp, actor, action, target_user, detail, comment
FROM billing_project_events
WHERE billing_project = %s
ORDER BY timestamp DESC
LIMIT 1000;
""",
            (bp_name,),
        )
    ]


# ---------------------------------------------------------------------------
# Role resolution
# ---------------------------------------------------------------------------


async def get_billing_role_for_quote(db: Database, username: str, is_global_bm: bool, quote_name: str) -> Optional[str]:
    if is_global_bm:
        return 'global_bm'
    quote_row = await db.select_and_fetchone('SELECT id FROM quotes WHERE name_cs = %s', (quote_name,))
    if not quote_row:
        return None
    mgr_row = await db.select_and_fetchone(
        'SELECT `role` FROM quote_managers WHERE quote_id = %s AND `user` = %s',
        (quote_row['id'], username),
    )
    if mgr_row:
        return f'quote_{mgr_row["role"]}'
    return None


async def get_billing_role_for_bp(db: Database, username: str, is_global_bm: bool, bp_name: str) -> Optional[str]:
    if is_global_bm:
        return 'global_bm'
    row = await db.select_and_fetchone(
        """
SELECT billing_projects.quote_id,
  qm.role AS qm_role,
  bpu.user AS bpu_user
FROM billing_projects
LEFT JOIN quote_managers qm
  ON qm.quote_id = billing_projects.quote_id AND qm.user = %s
LEFT JOIN billing_project_users bpu
  ON bpu.billing_project = billing_projects.name AND bpu.user_cs = %s
WHERE billing_projects.name_cs = %s AND billing_projects.`status` != 'deleted'
""",
        (username, username, bp_name),
    )
    if not row:
        return None
    if row['qm_role']:
        return f'quote_{row["qm_role"]}'
    if row['bpu_user']:
        return 'bp_member'
    return None


async def get_billing_role_for_quote_id(
    db: Database, username: str, is_global_bm: bool, quote_id: int
) -> Optional[str]:
    if is_global_bm:
        return 'global_bm'
    mgr_row = await db.select_and_fetchone(
        'SELECT `role` FROM quote_managers WHERE quote_id = %s AND `user` = %s',
        (quote_id, username),
    )
    if mgr_row:
        return f'quote_{mgr_row["role"]}'
    return None


# ---------------------------------------------------------------------------
# Quote writes
# ---------------------------------------------------------------------------


async def create_quote(
    db: Database,
    name: str,
    cost_object: str,
    actor: str,
    authorized_amount: Optional[float] = None,
    pi_name: Optional[str] = None,
    pm_designee: Optional[str] = None,
    comment: Optional[str] = None,
) -> int:
    """Insert a new quote and log the creation event. Returns the new quote id.

    Raises BatchOperationAlreadyCompletedError if the name already exists.
    """
    async with db.start() as tx:
        row = await tx.execute_and_fetchone('SELECT id FROM quotes WHERE name = %s FOR UPDATE;', (name,))
        if row is not None:
            raise BatchOperationAlreadyCompletedError(f'Quote {name} already exists.', 'info')

        now = time_msecs()
        quote_id = await tx.execute_insertone(
            """
INSERT INTO quotes (name, name_cs, cost_object, authorized_amount, pi_name, pm_designee, time_created)
VALUES (%s, %s, %s, %s, %s, %s, %s);
""",
            (name, name, cost_object, authorized_amount, pi_name, pm_designee, now),
        )
        assert quote_id is not None
        await _log_quote_event(tx, quote_id, actor, 'quote_created', detail=cost_object, comment=comment)
        return quote_id


async def edit_quote(
    db: Database,
    name: str,
    updates: dict,
    actor: str,
    billing_role: str,
    comment: Optional[str] = None,
) -> None:
    """Edit quote fields. updates keys: cost_object, pi_name, pm_designee, authorized_amount (float|None).

    Raises BatchUserError for authorization or capacity violations.
    """
    async with db.start() as tx:
        row = await tx.execute_and_fetchone(
            'SELECT id, authorized_amount FROM quotes WHERE name_cs = %s FOR UPDATE;', (name,)
        )
        if not row:
            raise BatchUserError(f'Unknown quote {name}.', 'error')

        quote_id = row['id']
        db_updates: dict = {}

        if 'cost_object' in updates:
            db_updates['cost_object'] = updates['cost_object']
        if 'pi_name' in updates:
            db_updates['pi_name'] = updates['pi_name']
        if 'pm_designee' in updates:
            db_updates['pm_designee'] = updates['pm_designee']
        if 'authorized_amount' in updates:
            new_amount = updates['authorized_amount']
            if new_amount is None and billing_role != 'global_bm':
                raise BatchUserError('Only global billing managers can set a quote to unlimited funding.', 'error')
            if new_amount is not None:
                limits_sum_row = await tx.execute_and_fetchone(
                    """
SELECT COALESCE(SUM(`limit`), 0) AS total_limit
FROM billing_projects
WHERE quote_id = %s AND `status` != 'deleted' AND `limit` IS NOT NULL;
""",
                    (quote_id,),
                )
                total = limits_sum_row['total_limit'] if limits_sum_row else 0
                if total > new_amount:
                    raise BatchUserError(
                        f'New authorized_amount {new_amount} is less than sum of BP limits {total}.',
                        'error',
                    )
            db_updates['authorized_amount'] = new_amount

        if db_updates:
            set_clause = ', '.join(f'`{k}` = %s' for k in db_updates)
            await tx.execute_update(
                f'UPDATE quotes SET {set_clause} WHERE id = %s;',
                (*db_updates.values(), quote_id),
            )
            await _log_quote_event(tx, quote_id, actor, 'quote_edited', comment=comment)


async def add_quote_manager(
    db: Database,
    quote_name: str,
    target_user: str,
    role: str,
    actor: str,
    comment: Optional[str] = None,
) -> None:
    """Add a user as owner or manager of a quote.

    Raises BatchOperationAlreadyCompletedError if user is already a manager.
    """
    async with db.start() as tx:
        quote_row = await tx.execute_and_fetchone('SELECT id FROM quotes WHERE name_cs = %s FOR UPDATE;', (quote_name,))
        if not quote_row:
            raise BatchUserError(f'Unknown quote {quote_name}.', 'error')
        quote_id = quote_row['id']

        existing = await tx.execute_and_fetchone(
            'SELECT role FROM quote_managers WHERE quote_id = %s AND user = %s;',
            (quote_id, target_user),
        )
        if existing:
            raise BatchOperationAlreadyCompletedError(
                f'User {target_user} is already a {existing["role"]} of quote {quote_name}.', 'info'
            )

        await tx.execute_insertone(
            'INSERT INTO quote_managers (quote_id, user, role) VALUES (%s, %s, %s);',
            (quote_id, target_user, role),
        )
        await _log_quote_event(
            tx, quote_id, actor, 'manager_added', target_user=target_user, detail=role, comment=comment
        )


async def remove_quote_manager(
    db: Database,
    quote_name: str,
    target_user: str,
    actor: str,
    comment: Optional[str] = None,
) -> None:
    """Remove a user from a quote's managers.

    Raises BatchOperationAlreadyCompletedError if user is not a manager.
    """
    async with db.start() as tx:
        quote_row = await tx.execute_and_fetchone('SELECT id FROM quotes WHERE name_cs = %s FOR UPDATE;', (quote_name,))
        if not quote_row:
            raise BatchUserError(f'Unknown quote {quote_name}.', 'error')
        quote_id = quote_row['id']

        existing = await tx.execute_and_fetchone(
            'SELECT role FROM quote_managers WHERE quote_id = %s AND user = %s FOR UPDATE;',
            (quote_id, target_user),
        )
        if not existing:
            raise BatchOperationAlreadyCompletedError(
                f'User {target_user} is not a manager of quote {quote_name}.', 'info'
            )

        await tx.just_execute(
            'DELETE FROM quote_managers WHERE quote_id = %s AND user = %s;',
            (quote_id, target_user),
        )
        await _log_quote_event(tx, quote_id, actor, 'manager_removed', target_user=target_user, comment=comment)


# ---------------------------------------------------------------------------
# Billing project writes
# ---------------------------------------------------------------------------


async def create_billing_project(
    db: Database,
    name: str,
    quote_id: int,
    limit: Optional[float],
    low_budget_alert: Optional[float],
    actor: str,
    billing_role: str,
    initial_users: Optional[list] = None,
    comment: Optional[str] = None,
) -> None:
    """Create a billing project under a quote.

    Raises BatchOperationAlreadyCompletedError if name already exists.
    Raises BatchUserError for authorization or capacity violations.
    """
    if initial_users is None:
        initial_users = []

    async with db.start() as tx:
        quote_row = await tx.execute_and_fetchone(
            'SELECT authorized_amount FROM quotes WHERE id = %s FOR UPDATE;', (quote_id,)
        )
        if quote_row is None:
            raise BatchUserError(f'Quote with id {quote_id} does not exist.', 'error')

        quote_authorized = quote_row['authorized_amount']

        if limit is None:
            if billing_role != 'global_bm':
                raise BatchUserError('Only global billing managers can create unlimited billing projects.', 'error')
            if quote_authorized is not None:
                raise BatchUserError('Unlimited billing projects can only be created under unlimited quotes.', 'error')

        if limit is not None and quote_authorized is not None:
            existing_sum_row = await tx.execute_and_fetchone(
                """
SELECT COALESCE(SUM(`limit`), 0) AS total_limit
FROM billing_projects
WHERE quote_id = %s AND `status` != 'deleted' AND `limit` IS NOT NULL;
""",
                (quote_id,),
            )
            existing_sum = existing_sum_row['total_limit'] if existing_sum_row else 0
            if existing_sum + limit > quote_authorized:
                raise BatchUserError(
                    f'Creating billing project with limit {limit} would exceed quote authorized amount {quote_authorized} '
                    f'(existing BPs use {existing_sum}).',
                    'error',
                )

        row = await tx.execute_and_fetchone('SELECT name_cs FROM billing_projects WHERE name = %s FOR UPDATE;', (name,))
        if row is not None:
            raise BatchOperationAlreadyCompletedError(f'Billing project {row["name_cs"]} already exists.', 'info')

        await tx.execute_insertone(
            'INSERT INTO billing_projects(name, name_cs, quote_id, `limit`, low_budget_alert) VALUES (%s, %s, %s, %s, %s);',
            (name, name, quote_id, limit, low_budget_alert),
        )

        for user in initial_users:
            await tx.execute_insertone(
                'INSERT INTO billing_project_users(billing_project, user, user_cs) VALUES (%s, %s, %s);',
                (name, user, user),
            )

        await _log_bp_event(tx, name, actor, 'bp_created', comment=comment)
        await _log_quote_event(tx, quote_id, actor, 'bp_created', target_project=name, comment=comment)


async def patch_billing_project(
    db: Database,
    bp_name: str,
    updates: dict,
    actor: str,
    billing_role: str,
    comment: Optional[str] = None,
) -> None:
    """Patch billing project limit and/or low_budget_alert.

    updates keys: 'limit' (raw value passed to _parse_billing_limit), 'low_budget_alert' (float|None).
    Raises BatchUserError for authorization or capacity violations.
    """
    async with db.start() as tx:
        row = await tx.execute_and_fetchone(
            """
SELECT billing_projects.name_cs, billing_projects.`status`, billing_projects.quote_id,
  q.authorized_amount,
  COALESCE((SELECT SUM(other_bp.`limit`) FROM billing_projects other_bp
    WHERE other_bp.quote_id = billing_projects.quote_id
      AND other_bp.name_cs != billing_projects.name_cs
      AND other_bp.`status` != 'deleted'), 0) AS other_bp_limits_sum
FROM billing_projects
JOIN quotes q ON q.id = billing_projects.quote_id
WHERE billing_projects.name_cs = %s AND billing_projects.`status` != 'deleted'
FOR UPDATE;
""",
            (bp_name,),
        )
        if not row:
            raise NonExistentBillingProjectError(bp_name)
        if row['status'] == 'closed':
            raise ClosedBillingProjectError(bp_name)

        db_updates: dict = {}

        if 'limit' in updates:
            new_limit = _parse_billing_limit(updates['limit'])
            if new_limit is None:
                if billing_role != 'global_bm':
                    raise BatchUserError(
                        'Only global billing managers can set a billing project to unlimited.', 'error'
                    )
                if row['authorized_amount'] is not None:
                    raise BatchUserError('Unlimited billing projects can only exist under unlimited quotes.', 'error')
            elif row['authorized_amount'] is not None:
                proposed_sum = row['other_bp_limits_sum'] + new_limit
                if proposed_sum > row['authorized_amount']:
                    raise BatchUserError(
                        f'Setting limit to {new_limit} would exceed quote authorized amount {row["authorized_amount"]} '
                        f'(other BPs use {row["other_bp_limits_sum"]}).',
                        'error',
                    )
            db_updates['limit'] = new_limit
            await _log_bp_event(tx, bp_name, actor, 'limit_changed', detail=str(new_limit), comment=comment)
            await _log_quote_event(
                tx,
                row['quote_id'],
                actor,
                'bp_limit_changed',
                target_project=bp_name,
                detail=str(new_limit),
                comment=comment,
            )

        if 'low_budget_alert' in updates:
            db_updates['low_budget_alert'] = updates['low_budget_alert']
            await _log_bp_event(
                tx, bp_name, actor, 'alert_threshold_changed', detail=str(updates['low_budget_alert']), comment=comment
            )

        if db_updates:
            set_clause = ', '.join(f'`{k}` = %s' for k in db_updates)
            await tx.execute_update(
                f'UPDATE billing_projects SET {set_clause} WHERE name_cs = %s;',
                (*db_updates.values(), bp_name),
            )


async def change_billing_project_quote(
    db: Database,
    bp_name: str,
    dest_quote_id: int,
    actor: str,
    comment: Optional[str] = None,
) -> None:
    """Move a billing project to a different quote.

    Raises BatchUserError for capacity violations or invalid state.
    """
    async with db.start() as tx:
        row = await tx.execute_and_fetchone(
            """
SELECT billing_projects.name_cs, billing_projects.`status`,
  billing_projects.quote_id AS src_quote_id,
  billing_projects.`limit`,
  q_dest.name AS dest_quote_name,
  q_dest.authorized_amount AS dest_authorized_amount,
  COALESCE((SELECT SUM(other_bp.`limit`) FROM billing_projects other_bp
    WHERE other_bp.quote_id = %s
      AND other_bp.`status` != 'deleted'), 0) AS dest_bp_limits_sum
FROM billing_projects
LEFT JOIN quotes q_dest ON q_dest.id = %s
WHERE billing_projects.name_cs = %s AND billing_projects.`status` != 'deleted'
FOR UPDATE;
""",
            (dest_quote_id, dest_quote_id, bp_name),
        )
        if not row:
            raise NonExistentBillingProjectError(bp_name)
        if row['dest_quote_name'] is None:
            raise BatchUserError(f'Unknown quote {dest_quote_id}.', 'error')
        if row['status'] == 'closed':
            raise ClosedBillingProjectError(bp_name)

        if row['limit'] is None and row['dest_authorized_amount'] is not None:
            raise BatchUserError(
                'An unlimited billing project cannot be moved to a quote with finite funding.', 'error'
            )

        if row['limit'] is not None and row['dest_authorized_amount'] is not None:
            new_sum = row['dest_bp_limits_sum'] + row['limit']
            if new_sum > row['dest_authorized_amount']:
                raise BatchUserError(
                    f'Moving billing project would exceed destination quote authorized amount '
                    f'{row["dest_authorized_amount"]} (existing BPs use {row["dest_bp_limits_sum"]}).',
                    'error',
                )

        src_quote_id = row['src_quote_id']
        await tx.execute_update(
            'UPDATE billing_projects SET quote_id = %s WHERE name_cs = %s;',
            (dest_quote_id, bp_name),
        )
        await _log_quote_event(tx, src_quote_id, actor, 'bp_unassigned', target_project=bp_name, comment=comment)
        await _log_quote_event(tx, dest_quote_id, actor, 'bp_assigned', target_project=bp_name, comment=comment)
        await _log_bp_event(tx, bp_name, actor, 'quote_changed', detail=row['dest_quote_name'], comment=comment)


# ---------------------------------------------------------------------------
# Billing project user management
# ---------------------------------------------------------------------------


async def add_billing_project_user(
    db: Database,
    bp_name: str,
    user: str,
    actor: str,
    comment: Optional[str] = None,
) -> None:
    """Add a user to a billing project.

    Raises BatchOperationAlreadyCompletedError if already a member.
    Raises NonExistentBillingProjectError / ClosedBillingProjectError on bad state.
    """
    async with db.start() as tx:
        row = await tx.execute_and_fetchone(
            """
SELECT billing_projects.name as billing_project,
    billing_projects.`status` as `status`,
    user
FROM billing_projects
LEFT JOIN (
  SELECT *
  FROM billing_project_users
  LEFT JOIN billing_projects ON billing_projects.name = billing_project_users.billing_project
  WHERE billing_projects.name_cs = %s AND user = %s
  FOR UPDATE
) AS t
ON billing_projects.name = t.billing_project
WHERE billing_projects.name_cs = %s AND billing_projects.`status` != 'deleted' LOCK IN SHARE MODE;
""",
            (bp_name, user, bp_name),
        )
        if row is None:
            raise NonExistentBillingProjectError(bp_name)
        if row['status'] == 'closed':
            raise ClosedBillingProjectError(bp_name)
        if row['user'] is not None:
            raise BatchOperationAlreadyCompletedError(
                f'User {user} is already member of billing project {bp_name}.', 'info'
            )

        await tx.execute_insertone(
            'INSERT INTO billing_project_users(billing_project, user, user_cs) VALUES (%s, %s, %s);',
            (bp_name, user, user),
        )
        await _log_bp_event(tx, bp_name, actor, 'user_added', target_user=user, comment=comment)


async def remove_billing_project_user(
    db: Database,
    bp_name: str,
    user: str,
    actor: str,
    comment: Optional[str] = None,
) -> None:
    """Remove a user from a billing project.

    Raises BatchOperationAlreadyCompletedError if user is not a member.
    Raises NonExistentBillingProjectError / BatchUserError on bad state.
    """
    async with db.start() as tx:
        row = await tx.execute_and_fetchone(
            """
SELECT billing_projects.name_cs as billing_project,
  billing_projects.`status` as `status`,
  `user`
FROM billing_projects
LEFT JOIN (
  SELECT billing_project_users.* FROM billing_project_users
  LEFT JOIN billing_projects ON billing_projects.name = billing_project_users.billing_project
  WHERE billing_projects.name_cs = %s AND user_cs = %s
  FOR UPDATE
) AS t ON billing_projects.name = t.billing_project
WHERE billing_projects.name_cs = %s;
""",
            (bp_name, user, bp_name),
        )
        if not row:
            raise NonExistentBillingProjectError(bp_name)
        if row['status'] in {'closed', 'deleted'}:
            raise BatchUserError(
                f'Billing project {bp_name} has been closed or deleted and cannot be modified.', 'error'
            )
        if row['user'] is None:
            raise BatchOperationAlreadyCompletedError(f'User {user} is not in billing project {bp_name}.', 'info')

        await tx.just_execute(
            """
DELETE billing_project_users FROM billing_project_users
LEFT JOIN billing_projects ON billing_projects.name = billing_project_users.billing_project
WHERE billing_projects.name_cs = %s AND user_cs = %s;
""",
            (bp_name, user),
        )
        await _log_bp_event(tx, bp_name, actor, 'user_removed', target_user=user, comment=comment)


# ---------------------------------------------------------------------------
# Billing project lifecycle
# ---------------------------------------------------------------------------


async def close_billing_project(
    db: Database,
    bp_name: str,
    actor: str,
    comment: Optional[str] = None,
) -> None:
    """Close a billing project.

    Raises BatchOperationAlreadyCompletedError if already closed.
    Raises BatchUserError if there are running batches.
    """
    async with db.start() as tx:
        row = await tx.execute_and_fetchone(
            """
SELECT name_cs, `status`, quote_id, batches.id as batch_id
FROM billing_projects
LEFT JOIN batches
ON billing_projects.name = batches.billing_project
AND billing_projects.`status` != 'deleted'
AND batches.time_completed IS NULL
AND NOT batches.deleted
WHERE name_cs = %s
LIMIT 1
FOR UPDATE;
""",
            (bp_name,),
        )
        if not row:
            raise NonExistentBillingProjectError(bp_name)
        assert row['name_cs'] == bp_name
        if row['status'] == 'closed':
            raise BatchOperationAlreadyCompletedError(
                f'Billing project {bp_name} is already closed or deleted.', 'info'
            )
        if row['batch_id'] is not None:
            raise BatchUserError(f'Billing project {bp_name} has running batches.', 'error')

        await tx.execute_update("UPDATE billing_projects SET `status` = 'closed' WHERE name_cs = %s;", (bp_name,))
        await _log_bp_event(tx, bp_name, actor, 'bp_closed', comment=comment)
        await _log_quote_event(tx, row['quote_id'], actor, 'bp_closed', target_project=bp_name, comment=comment)


async def reopen_billing_project(
    db: Database,
    bp_name: str,
    actor: str,
    comment: Optional[str] = None,
) -> None:
    """Reopen a closed billing project.

    Raises BatchOperationAlreadyCompletedError if already open.
    Raises BatchUserError if deleted.
    """
    async with db.start() as tx:
        row = await tx.execute_and_fetchone(
            'SELECT name_cs, `status`, quote_id FROM billing_projects WHERE name_cs = %s FOR UPDATE;', (bp_name,)
        )
        if not row:
            raise NonExistentBillingProjectError(bp_name)
        assert row['name_cs'] == bp_name
        if row['status'] == 'deleted':
            raise BatchUserError(f'Billing project {bp_name} has been deleted and cannot be reopened.', 'error')
        if row['status'] == 'open':
            raise BatchOperationAlreadyCompletedError(f'Billing project {bp_name} is already open.', 'info')

        await tx.execute_update("UPDATE billing_projects SET `status` = 'open' WHERE name_cs = %s;", (bp_name,))
        await _log_bp_event(tx, bp_name, actor, 'bp_reopened', comment=comment)
        await _log_quote_event(tx, row['quote_id'], actor, 'bp_reopened', target_project=bp_name, comment=comment)


async def delete_billing_project(
    db: Database,
    bp_name: str,
) -> None:
    """Delete (soft-delete) a closed billing project.

    Raises BatchOperationAlreadyCompletedError if already deleted.
    Raises BatchUserError if open.
    """
    async with db.start() as tx:
        row = await tx.execute_and_fetchone(
            'SELECT name_cs, `status` FROM billing_projects WHERE name_cs = %s FOR UPDATE;', (bp_name,)
        )
        if not row:
            raise NonExistentBillingProjectError(bp_name)
        assert row['name_cs'] == bp_name, row
        if row['status'] == 'deleted':
            raise BatchOperationAlreadyCompletedError(f'Billing project {bp_name} is already deleted.', 'info')
        if row['status'] == 'open':
            raise BatchUserError(f'Billing project {bp_name} is open and cannot be deleted.', 'error')

        await tx.execute_update("UPDATE billing_projects SET `status` = 'deleted' WHERE name_cs = %s;", (bp_name,))
